"""OpenAI vision based section extraction pipeline."""

import base64
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import APIError, OpenAI, RateLimitError

from .prompts import SECTION_TYPE_TO_PROMPT
from .utils import (
    detect_section_type,
    extract_json_from_text,
    get_section_number,
    load_images,
    sort_results,
)


load_dotenv()

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TIMEOUT_SECONDS = 90
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 20

_EASTERN_TO_WESTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _to_western_digits(text: str) -> str:
    """Convert Eastern Arabic digits to Western digits."""
    return text.translate(_EASTERN_TO_WESTERN_DIGITS)


def _parse_number(value: str) -> Optional[float]:
    """Parse integer/decimal number from text segment."""
    try:
        return float(value.strip())
    except ValueError:
        return None


def _fix_math_line(line: str) -> str:
    """Normalize one arithmetic line to canonical math order."""
    cleaned = _to_western_digits(line)
    cleaned = cleaned.replace("−", "-").replace("×", "*").replace("÷", "/")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if "=" not in cleaned:
        return cleaned

    lhs, rhs = [part.strip() for part in cleaned.split("=", 1)]
    has_expr_rhs = bool(re.search(r"\d+\s*[+\-*/]\s*\d+", rhs))
    lhs_is_number = bool(re.fullmatch(r"-?\d+(?:\.\d+)?", lhs))
    if lhs_is_number and has_expr_rhs:
        cleaned = f"{rhs} = {lhs}"

    match = re.fullmatch(
        (
            r"\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*"
            r"(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)\s*"
        ),
        cleaned,
    )
    if not match:
        return cleaned

    a_txt, op, b_txt, c_txt = match.groups()
    a = _parse_number(a_txt)
    b = _parse_number(b_txt)
    c = _parse_number(c_txt)
    if a is None or b is None or c is None:
        return cleaned

    eps = 1e-6
    if op == "-" and abs((a - b) - c) > eps and abs((b - a) - c) <= eps:
        return f"{b_txt} - {a_txt} = {c_txt}"
    if op == "/" and abs((a / b) - c) > eps and abs((b / a) - c) <= eps:
        return f"{b_txt} / {a_txt} = {c_txt}"

    return f"{a_txt} {op} {b_txt} = {c_txt}"


def _normalize_math_text(text: Optional[str]) -> Optional[str]:
    """Normalize multi-line extracted text while preserving line breaks."""
    if not text:
        return None
    lines = text.splitlines()
    normalized = [_fix_math_line(line) for line in lines]
    return "\n".join(normalized)


class OpenAISectionExtractor:
    """Extracts question text and handwritten answers from section images."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_VISION_MODEL") or os.getenv(
            "OPENAI_MODEL_NAME",
            DEFAULT_MODEL,
        )
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI vision extraction"
            )

        self.client = OpenAI(api_key=self.api_key, timeout=timeout_seconds)

    def extract_folder(self, folder_path: str) -> List[Dict]:
        """Run extraction on all section images and return sorted results."""
        image_paths = load_images(Path(folder_path))

        results: List[Dict] = []
        for image_path in image_paths:
            section_number = get_section_number(image_path.name)
            section_type = detect_section_type(image_path)
            prompt = SECTION_TYPE_TO_PROMPT.get(
                section_type,
                SECTION_TYPE_TO_PROMPT["unknown"],
            )
            results.append(
                self._extract_single_image(
                    image_path=image_path,
                    prompt=prompt,
                    section_number=section_number,
                )
            )

        return sort_results(results)

    def _extract_single_image(
        self,
        image_path: Path,
        prompt: str,
        section_number: int,
    ) -> Dict:
        """Send one section image to OpenAI and parse the JSON response."""
        image_url = self._image_to_data_url(image_path)
        instruction = self._build_instruction(prompt, section_number)

        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You read scanned school exams, including "
                                "Arabic text and children's handwriting. "
                                "Preserve wording exactly as written and "
                                "return only valid JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                        "detail": "high",
                                    },
                                },
                            ],
                        },
                    ],
                )
                raw_text = response.choices[0].message.content or "{}"
                parsed = extract_json_from_text(raw_text)
                return self._normalize_result(parsed, section_number)
            except RateLimitError:
                attempt += 1
                if attempt > MAX_RETRIES:
                    return self._error_result(
                        section_number,
                        "OpenAI rate limit exceeded",
                    )
                time.sleep(RETRY_DELAY_SECONDS)
            except APIError as exc:
                return self._error_result(section_number, str(exc))
            except Exception as exc:
                return self._error_result(section_number, str(exc))

    @staticmethod
    def _build_instruction(prompt: str, section_number: int) -> str:
        """Combine the prompt template with stable JSON output rules."""
        return (
            f"{prompt}\n\n"
            f"This is section_number={section_number}.\n"
            "Return valid JSON only using this schema:\n"
            "{\n"
            '  "section_number": integer,\n'
            '  "question": string or null,\n'
            '  "student_answer": string or null,\n'
            '  "confidence": number\n'
            "}\n"
            "If the student answer does not exist, use null."
        )

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        """Convert an image file to a base64 data URL for OpenAI vision."""
        suffix = image_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        elif suffix == ".png":
            mime_type = "image/png"
        elif suffix == ".webp":
            mime_type = "image/webp"
        elif suffix == ".bmp":
            mime_type = "image/bmp"
        else:
            mime_type = "image/jpeg"

        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _normalize_result(parsed: Dict, section_number: int) -> Dict:
        """Normalize model output to the required schema."""
        question = parsed.get("question")
        student_answer = parsed.get("student_answer")
        confidence = parsed.get("confidence", 0.0)

        question_norm = _normalize_math_text(question) if question else None
        answer_norm = (
            _normalize_math_text(student_answer)
            if student_answer
            else None
        )

        return {
            "section_number": section_number,
            "question": question_norm,
            "student_answer": answer_norm,
            "confidence": float(confidence or 0.0),
        }

    @staticmethod
    def _error_result(section_number: int, error_message: str) -> Dict:
        """Return a stable failure payload for a single section."""
        return {
            "section_number": section_number,
            "question": None,
            "student_answer": None,
            "confidence": 0.0,
            "error": error_message,
        }
