from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from logger_manager import LoggerManager
from agents_module.prompts import (
    BASE_PROMPT_SUBMISSION_EXTRACTION,
    GENERIC_TEMPLATE_SUBMISSION,
    TEMPLATES_SUBMISSIONS_PROMPT,
)

try:
    from agents_module.question_extractor_google_cloud import (
        CLASSIFICATION_PROMPT,
        QuestionExtractorGoogleCloud,
    )
except Exception:
    CLASSIFICATION_PROMPT = None
    QuestionExtractorGoogleCloud = None

try:
    from agents_module.ocr_gemini import run_ocr as legacy_run_ocr
except Exception:
    legacy_run_ocr = None

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(vertexai=True, api_key=GEMINI_API_KEY)

_OUTPUT_DIR = Path("Exams") / "content_extraction_jsons"
_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ContentCorrectionExtraction:
    def __init__(self) -> None:
        self.logger = LoggerManager.get_logger(__name__)
        self._question_type_classifier = self._build_question_type_classifier()

    def _build_question_type_classifier(self):
        if QuestionExtractorGoogleCloud is None:
            return None

        try:
            if os.getenv("MISTRAL_API_KEY"):
                classifier = QuestionExtractorGoogleCloud.__new__(
                    QuestionExtractorGoogleCloud
                )
                classifier.logger = self.logger
                return classifier
        except Exception as exc:
            self.logger.warning(
                "QuestionExtractorGoogleCloud classifier unavailable: %s",
                exc,
            )
        return None

    def process_path(self, input_path: Path) -> Dict[str, Any]:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        image_paths = self._collect_image_paths(input_path)
        if not image_paths:
            return {"submission_content": []}

        results: List[Dict[str, Any]] = []
        for image_path in image_paths:
            try:
                results.append(self.process_image(image_path))
            except Exception as exc:
                self.logger.error(
                    "Failed to process %s: %s",
                    image_path.name,
                    exc,
                    exc_info=True,
                )
                results.append(self._error_item(image_path, str(exc)))

        return {"submission_content": results}

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        question_type, confidence = self._classify_question_type(image_path)
        structured = self._extract_by_type(image_path, question_type)
        return {
            "question_type": question_type,
            "confidence": confidence,
            "content": {
                "content": structured.get("content", {}),
                "student_answer": structured.get("student_answer", {}),
                "confidence": structured.get("confidence", confidence),
            },
        }

    def _collect_image_paths(self, input_path: Path) -> List[Path]:
        if input_path.is_file():
            return [input_path] if input_path.suffix.lower() in _SUPPORTED_SUFFIXES else []

        paths = [
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
        ]
        return sorted(paths, key=self._extract_section_number)

    def _classify_question_type(self, image_path: Path) -> Tuple[str, float]:
        ocr_text = self._extract_text(image_path)

        if self._question_type_classifier is not None:
            try:
                question_type, confidence = self._question_type_classifier._classify(
                    ocr_text
                )
                return question_type, confidence
            except Exception as exc:
                self.logger.warning(
                    "Classifier from question_extractor_google_cloud failed; using fallback: %s",
                    exc,
                )

        if CLASSIFICATION_PROMPT:
            raw = self._gemini_classify(image_path)
            parsed = self._parse_json(raw)
            return (
                str(parsed.get("question_type", "UNKNOWN")).upper(),
                float(parsed.get("confidence", 0.5)),
            )

        return "UNKNOWN", 0.0

    def _extract_text(self, image_path: Path) -> str:
        if legacy_run_ocr is not None:
            try:
                return legacy_run_ocr(str(image_path)) or ""
            except Exception as exc:
                self.logger.warning(
                    "Legacy OCR helper failed for %s, using Gemini OCR fallback: %s",
                    image_path.name,
                    exc,
                )

        with open(image_path, "rb") as f:
            file_bytes = f.read()

        image_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type=self._mime_type(image_path),
        )
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[
                "Extract all visible text from this exam page as raw OCR text.",
                image_part,
            ],
        )
        return response.text or ""

    def _gemini_classify(self, image_path: Path) -> str:
        if not CLASSIFICATION_PROMPT:
            return "{}"

        with open(image_path, "rb") as f:
            file_bytes = f.read()

        image_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type=self._mime_type(image_path),
        )
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[CLASSIFICATION_PROMPT, image_part],
        )
        return response.text or "{}"

    def _extract_by_type(self, image_path: Path, question_type: str) -> Dict[str, Any]:
        template = TEMPLATES_SUBMISSIONS_PROMPT.get(
            question_type,
            GENERIC_TEMPLATE_SUBMISSION,
        )
        prompt = BASE_PROMPT_SUBMISSION_EXTRACTION.format(
            structure_placeholder=template
        )

        with open(image_path, "rb") as f:
            file_bytes = f.read()

        image_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type=self._mime_type(image_path),
        )
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=[prompt, image_part],
        )
        return self._parse_json(response.text or "{}")

    def _parse_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            self.logger.warning("Could not parse JSON response")
            return {"raw_text": cleaned, "error": "Failed to parse JSON"}

    def _extract_section_number(self, path: Path) -> int:
        match = re.search(r"section[_\s-](\d+)", path.name, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _mime_type(self, path: Path) -> str:
        return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    def _error_item(self, image_path: Path, error_message: str) -> Dict[str, Any]:
        return {
            "question_type": "UNKNOWN",
            "confidence": 0.0,
            "content": {
                "content": {},
                "student_answer": {},
                "confidence": 0.0,
                "error": error_message,
                "image_name": image_path.name,
            },
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract question type then structured submission content."
    )
    parser.add_argument("input_path", type=Path, help="Image file or folder path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    pipeline = ContentCorrectionExtraction()
    result = pipeline.process_path(args.input_path)

    output_path = args.output
    if output_path is None:
        if args.input_path.is_dir():
            output_name = args.input_path.name + "_submission_content.json"
        else:
            output_name = args.input_path.stem + "_submission_content.json"
        output_path = _OUTPUT_DIR / output_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
