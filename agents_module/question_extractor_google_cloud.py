"""
question_extractor_google_cloud.py
────────────────────────────────────────────────────────────────────────────────
Pipeline
  Step 1 — Google Cloud Vision   : OCR the exam image  →  raw text
  Step 2 — GPT-4o (OpenAI)       : CLASSIFICATION_PROMPT + OCR text
                                    →  question_type + confidence
  Step 3 — GPT-4o (OpenAI)       : extraction prompt + OCR text
                                    →  structured content dict

Why this combination?
  • Cloud Vision gives higher-quality OCR on printed/mixed Arabic exams than
    Gemini's built-in vision, especially with diacritics (tashkeel).
  • GPT-4o gives LLM-quality classification identical in behaviour to the
    Gemini image path, using the same CLASSIFICATION_PROMPT and per-type
    extraction prompts already defined in prompts.py.

Public API is identical to QuestionExtractor (Gemini image version):
  • process_exam(folder_path, is_submission, save_results, output_path)
  • process_image(image_path, is_submission)
  • get_statistics(results)
  • Context-manager support (__enter__ / __exit__)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from google.api_core import exceptions as gcp_exceptions
from google.cloud import vision
from openai import OpenAI, RateLimitError, APIError

# ── project-root path fix 
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from logger_manager import LoggerManager
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))
    from logger_manager import LoggerManager

try:
    from agents_module.agents_config import AgentsConfig
    from agents_module.prompts import CLASSIFICATION_PROMPT
except ImportError:
    from agents_config import AgentsConfig      # type: ignore
    from prompts import CLASSIFICATION_PROMPT   # type: ignore

import os
from dotenv import load_dotenv
load_dotenv()   # reads .env from cwd — same pattern as AgentsConfig

try:
    from Layout_module.layout_config import LayoutConfig
    _GCV_CREDENTIALS: Optional[str] = LayoutConfig.CREDENTIALS_PATH
except Exception:
    _GCV_CREDENTIALS = None

# OpenAI key — add OPENAI_API_KEY=sk-... to your .env file
_OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

# ── constants ─────────────────────────────────────────────────────────────────
_OPENAI_MODEL     = "gpt-4o"
_MAX_TOKENS       = 2048
_TEMPERATURE      = 0.1          # low temp → consistent, deterministic outputs

_MAX_OCR_RETRIES  = 5
_OCR_BASE_DELAY_S = 2
_OCR_MAX_DELAY_S  = 60

_MAX_LLM_RETRIES  = 3
_LLM_RETRY_DELAY  = 60                          # seconds to wait on rate-limit


class QuestionExtractorGoogleCloud:
    """
    Classify and extract exam-question images using:
      • Google Cloud Vision  for OCR
      • GPT-4o (OpenAI)      for classification + structured extraction

    Usage
    ─────
    with QuestionExtractorGoogleCloud() as extractor:
        result = extractor.process_image(Path("section_1.jpg"), is_submission=False)

    extractor = QuestionExtractorGoogleCloud()
    results   = extractor.process_exam(Path("exam_folder/"))
    extractor.close()
    """

    # ── init / teardown ───────────────────────────────────────────────────────

    def __init__(self, max_workers: int = 5) -> None:
        self.logger      = LoggerManager.get_logger(__name__)
        self.max_workers = max_workers
        self.executor    = ThreadPoolExecutor(max_workers=self.max_workers)

        # Google Cloud Vision client
        if AgentsConfig.GOOGLE_API_KEY:
            self.vision_client = vision.ImageAnnotatorClient(
                client_options={"api_key": AgentsConfig.GOOGLE_API_KEY}
            )
            self.logger.info(f"Cloud Vision client initialised with API key from AgentsConfig")
        else:
            self.vision_client = vision.ImageAnnotatorClient()
            self.logger.info("Cloud Vision client initialised via Application Default Credentials")

        # OpenAI client
        if not _OPENAI_API_KEY:
            raise EnvironmentError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY as an environment variable "
                "or add it to LayoutConfig.OPENAI_API_KEY."
            )
        self.openai_client = OpenAI(api_key=_OPENAI_API_KEY)
        self.logger.info(f"OpenAI client initialised (model: {_OPENAI_MODEL})")

        self.logger.info("QuestionExtractorGoogleCloud ready")

    def __enter__(self) -> "QuestionExtractorGoogleCloud":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self.executor:
            self.executor.shutdown(wait=True)

    # ── public API ─────

    def process_exam(
        self,
        folder_path: Path,
        is_submission: bool = False,
        save_results: bool = False,
        output_path: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Process every image in *folder_path* in parallel.

        Args:
            folder_path:   Directory containing exam-section images.
            is_submission: True → extract student answers; False → correct answers.
            save_results:  Write results to a JSON file when True.
            output_path:   JSON destination (defaults to cwd).

        Returns:
            List of per-image result dicts sorted by section number.
        """
        if self.executor is None:
            raise RuntimeError("Executor is not initialised — was close() already called?")
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder_path}")

        image_paths = [
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        total = len(image_paths)
        if total == 0:
            self.logger.warning(f"No image files found in {folder_path}")
            return []

        self.logger.info(f"Submitting {total} image(s) to thread pool …")
        futures = [
            (self.executor.submit(self.process_image, p, is_submission), p)
            for p in image_paths
        ]

        results: List[Dict] = []
        for future, image_path in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                self.logger.error(f"Task failed for {image_path.name}: {exc}")
                results.append(self._error_result(image_path, exc))

        results.sort(
            key=lambda r: self._extract_section_number(
                r.get("meta_data", {}).get("image_name", "")
            )
        )

        successful = sum(1 for r in results if "error" not in r)
        self.logger.info(f"Batch complete — {successful}/{total} successful")

        if save_results:
            self._save_results(
                results,
                output_path or Path("question_analysis_results_google_cloud.json"),
            )
        return results

    def process_image(self, image_path: Path, is_submission: bool) -> Dict:
        """
        Full pipeline for a single exam-section image.

        Steps:
            1. Cloud Vision OCR  → ocr_text
            2. GPT-4o classify   → question_type, confidence
            3. GPT-4o extract    → structured content dict

        Args:
            image_path:    Path to the image file.
            is_submission: True → submission mode; False → correction mode.

        Returns:
            Dict with keys: question_type, confidence, content, meta_data.
        """
        self.logger.info(f"Processing: {image_path.name}")
        try:
            with Image.open(image_path) as image:

                # ── Step 1: OCR ───────────────────────────────────────────────
                ocr_text = self._ocr_extract_text(image)

                if not ocr_text.strip():
                    self.logger.warning(
                        f"{image_path.name}: Cloud Vision returned empty text — "
                        "image may be handwritten-only or unreadable."
                    )
                    return {
                        "question_type": "UNKNOWN",
                        "confidence":    0.0,
                        "content":       {"error": "OCR returned empty text"},
                        "meta_data": {
                            "image_path": str(image_path),
                            "image_name": image_path.name,
                        },
                    }

                # ── Step 2: Classify with GPT-4o ─────────────────────────────
                question_type, confidence = self._classify_question_type(ocr_text)

                # ── Step 3: Fetch extraction prompt ───────────────────────────
                extraction_prompt = AgentsConfig.get_extraction_prompt(
                    question_type, is_submission
                )
                if extraction_prompt is None:
                    self.logger.warning(
                        f"No extraction prompt for type '{question_type}'."
                    )
                    return {
                        "question_type": question_type,
                        "confidence":    confidence,
                        "content": {
                            "error": "No extraction prompt available for this question type"
                        },
                        "meta_data": {
                            "image_path": str(image_path),
                            "image_name": image_path.name,
                        },
                    }

                # ── Step 4: Extract content with GPT-4o ───────────────────────
                content = self._extract_content(ocr_text, extraction_prompt)

            return {
                "question_type": question_type,
                "confidence":    confidence,
                "content":       content,
                "meta_data": {
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                },
            }

        except Exception as exc:
            self.logger.error(f"Error processing {image_path}: {exc}", exc_info=True)
            return self._error_result(image_path, exc)

    def get_statistics(self, results: List[Dict]) -> Dict:
        """Aggregate success rate, confidence, and type distribution."""
        if not results:
            return {"error": "No results to analyse"}

        total      = len(results)
        successful = sum(1 for r in results if "error" not in r)
        failed     = total - successful

        type_counts: Dict[str, int] = {}
        conf_sum, conf_count = 0.0, 0

        for r in results:
            if "error" in r:
                continue
            q_type = r.get("question_type", "UNKNOWN")
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
            conf_sum   += float(r.get("confidence", 0))
            conf_count += 1

        return {
            "total_processed":            total,
            "successful":                 successful,
            "failed":                     failed,
            "success_rate":               (successful / total * 100) if total else 0,
            "average_confidence":         conf_sum / conf_count if conf_count else 0.0,
            "question_type_distribution": type_counts,
            "most_common_type": (
                max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
            ),
        }

    # ── Step 1 : Cloud Vision OCR ─────────────────────────────────────────────

    def _ocr_extract_text(self, image: Image.Image) -> str:
        """
        Send image to Google Cloud Vision document_text_detection.
        Retries on transient quota / availability errors with exponential back-off.
        """
        attempt = 0
        while True:
            try:
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=95)
                response = self.vision_client.document_text_detection(
                    image=vision.Image(content=buf.getvalue())
                )
                if response.error.message:
                    raise RuntimeError(f"Vision API error: {response.error.message}")
                ann = response.full_text_annotation
                text = ann.text if ann and ann.text else ""
                self.logger.debug(f"OCR extracted {len(text)} characters")
                return text

            except (
                gcp_exceptions.ResourceExhausted,
                gcp_exceptions.DeadlineExceeded,
                gcp_exceptions.ServiceUnavailable,
            ) as exc:
                attempt += 1
                if attempt > _MAX_OCR_RETRIES:
                    self.logger.error(f"OCR failed after {_MAX_OCR_RETRIES} retries: {exc}")
                    raise
                delay = min(_OCR_MAX_DELAY_S, _OCR_BASE_DELAY_S ** attempt)
                self.logger.warning(
                    f"Transient OCR error — retrying in {delay}s "
                    f"(attempt {attempt}/{_MAX_OCR_RETRIES}): {exc}"
                )
                time.sleep(delay)

    # ── Step 2 : GPT-4o classification ───────────────────────────────────────

    def _classify_question_type(self, ocr_text: str) -> Tuple[str, float]:
        """
        Send OCR text + CLASSIFICATION_PROMPT to GPT-4o.

        The prompt already contains full instructions and output format,
        so we just append the extracted text as additional context.

        Returns:
            (question_type: str, confidence: float)
        """
        # Build the message: the existing CLASSIFICATION_PROMPT describes the task
        # and output format; we append the OCR text so the model classifies it.
        user_message = (
            f"{CLASSIFICATION_PROMPT}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "NOTE: You are classifying based on OCR-extracted text (not an image).\n"
            "Apply the same classification rules to the text below.\n\n"
            "OCR TEXT:\n"
            f"{ocr_text}"
        )

        attempt = 0
        while True:
            try:
                response = self.openai_client.chat.completions.create(
                    model=_OPENAI_MODEL,
                    temperature=_TEMPERATURE,
                    max_tokens=_MAX_TOKENS,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert exam question classifier. "
                                "Respond ONLY with valid JSON — no markdown, no extra text."
                            ),
                        },
                        {"role": "user", "content": user_message},
                    ],
                )

                raw = response.choices[0].message.content or ""
                result = self._parse_json(raw)

                question_type = str(result.get("question_type", "UNKNOWN")).upper()
                confidence    = float(result.get("confidence", 0.5))

                self.logger.info(
                    f"[GPT-4o classify] {question_type}  confidence={confidence:.2f}"
                )
                return question_type, confidence

            except RateLimitError as exc:
                attempt += 1
                if attempt > _MAX_LLM_RETRIES:
                    self.logger.error(
                        f"GPT-4o rate limit exceeded after {_MAX_LLM_RETRIES} retries"
                    )
                    return "UNKNOWN", 0.0
                delay = self._parse_retry_delay(str(exc)) or _LLM_RETRY_DELAY
                self.logger.warning(
                    f"GPT-4o rate limit — retrying in {delay}s "
                    f"(attempt {attempt}/{_MAX_LLM_RETRIES})"
                )
                time.sleep(delay)

            except APIError as exc:
                self.logger.error(f"GPT-4o API error during classification: {exc}")
                return "UNKNOWN", 0.0

            except Exception as exc:
                self.logger.error(f"Unexpected classification error: {exc}", exc_info=True)
                return "UNKNOWN", 0.0

    # ── Step 3 : GPT-4o content extraction ───────────────────────────────────

    def _extract_content(self, ocr_text: str, extraction_prompt: str) -> Dict:
        """
        Send OCR text + per-type extraction prompt to GPT-4o.

        The extraction_prompt already specifies the exact JSON structure expected.
        We append the OCR text and ask for structured output.

        Returns:
            Parsed content dict (mirrors what the Gemini image path returns).
        """
        user_message = (
            f"{extraction_prompt}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "NOTE: Extract from the OCR text below (not an image).\n"
            "Apply the same extraction rules — the text was produced by Google Cloud Vision OCR.\n\n"
            "OCR TEXT:\n"
            f"{ocr_text}"
        )

        attempt = 0
        while True:
            try:
                response = self.openai_client.chat.completions.create(
                    model=_OPENAI_MODEL,
                    temperature=_TEMPERATURE,
                    max_tokens=_MAX_TOKENS,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert exam content extractor. "
                                "Respond ONLY with valid JSON — no markdown, no extra text."
                            ),
                        },
                        {"role": "user", "content": user_message},
                    ],
                )

                raw     = response.choices[0].message.content or ""
                content = self._parse_json(raw)

                if not content or content.get("error"):
                    self.logger.warning("GPT-4o extraction returned an error or empty result")

                return content

            except RateLimitError as exc:
                attempt += 1
                if attempt > _MAX_LLM_RETRIES:
                    self.logger.error(
                        f"GPT-4o rate limit exceeded after {_MAX_LLM_RETRIES} retries "
                        "during extraction"
                    )
                    return {"error": "Rate limit exceeded", "raw_text": ocr_text}
                delay = self._parse_retry_delay(str(exc)) or _LLM_RETRY_DELAY
                self.logger.warning(
                    f"GPT-4o rate limit during extraction — retrying in {delay}s "
                    f"(attempt {attempt}/{_MAX_LLM_RETRIES})"
                )
                time.sleep(delay)

            except APIError as exc:
                self.logger.error(f"GPT-4o API error during extraction: {exc}")
                return {"error": str(exc), "raw_text": ocr_text}

            except Exception as exc:
                self.logger.error(f"Unexpected extraction error: {exc}", exc_info=True)
                return {"error": str(exc), "raw_text": ocr_text}

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> Dict:
        """Strip markdown fences and parse JSON from LLM response."""
        text = text.strip()
        # Remove ```json ... ``` or ``` ... ``` wrappers
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            return {"raw_text": text, "error": f"Failed to parse JSON: {exc}"}

    @staticmethod
    def _parse_retry_delay(error_str: str) -> float:
        """Extract wait time from a rate-limit error message."""
        match = re.search(r"retry after (\d+(?:\.\d+)?)", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"(\d+(?:\.\d+)?)\s*s", error_str, re.IGNORECASE)
        return float(match.group(1)) if match else _LLM_RETRY_DELAY

    @staticmethod
    def _extract_section_number(image_name: str) -> int:
        m = re.search(r"section[_\s-](\d+)", image_name, re.IGNORECASE)
        return int(m.group(1)) if m else -1

    @staticmethod
    def _error_result(image_path: Path, exc: Exception) -> Dict:
        return {
            "meta_data": {
                "image_path": str(image_path),
                "image_name": image_path.name,
            },
            "error":         str(exc),
            "question_type": "ERROR",
            "confidence":    0.0,
        }

    def _save_results(self, results: List[Dict], output_path: Path) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved → {output_path}")
        except Exception as exc:
            self.logger.error(f"Failed to save results: {exc}", exc_info=True)


# ── backward-compatible alias ─────────────────────────────────────────────────
QuestionExtractor = QuestionExtractorGoogleCloud