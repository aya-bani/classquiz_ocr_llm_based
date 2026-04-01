
from __future__ import annotations
# === CLI for batch classification ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch classify section images and output question type & confidence.")
    parser.add_argument("folder", type=str, help="Folder path containing section images")
    parser.add_argument("--is_submission", action="store_true", help="Set if images are student submissions (default: correction mode)")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    is_submission = args.is_submission

    with QuestionExtractorGoogleCloud() as extractor:
        results = extractor.process_exam(folder_path, is_submission=is_submission)
    print("\n===== Classification Results =====\n")
    for r in results:
        image_name = r.get("meta_data", {}).get("image_name", "?")
        qtype = r.get("question_type", "UNKNOWN")
        conf = r.get("confidence", 0)
        print(f"{image_name}: {qtype} (confidence: {conf})")

import io
import json
import re
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from google.api_core import exceptions as gcp_exceptions
from google.cloud import vision

# ── project-root path fix
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from logger_manager import LoggerManager
from agents_module.agents_config import AgentsConfig
from agents_module.prompts import (
    CLASSIFICATION_PROMPT,
    TEMPLATES_CORRECTION_PROMPT,
    TEMPLATES_SUBMISSIONS_PROMPT,
)

import os
from dotenv import load_dotenv
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL     = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL   = "mistral-small-latest"

_MAX_TOKENS       = 2048
_TEMPERATURE      = 0.1

_MAX_OCR_RETRIES  = 5
_OCR_BASE_DELAY_S = 2
_OCR_MAX_DELAY_S  = 60

_MAX_LLM_RETRIES  = 3
_LLM_RETRY_DELAY  = 15


class QuestionExtractorGoogleCloud:

    def __init__(self, max_workers: int = 5) -> None:
        self.logger      = LoggerManager.get_logger(__name__)
        self.max_workers = max_workers
        self.executor    = ThreadPoolExecutor(max_workers=self.max_workers)

        # Google Cloud Vision
        if AgentsConfig.GOOGLE_API_KEY:
            self.vision_client = vision.ImageAnnotatorClient(
                client_options={"api_key": AgentsConfig.GOOGLE_API_KEY}
            )
            self.logger.info("Cloud Vision client initialised with API key from AgentsConfig")
        else:
            self.vision_client = vision.ImageAnnotatorClient()
            self.logger.info("Cloud Vision client initialised via Application Default Credentials")

        if not MISTRAL_API_KEY:
            raise EnvironmentError("MISTRAL_API_KEY not found in .env")

        self.logger.info(f"Mistral ready (model: {MISTRAL_MODEL})")
        self.logger.info("QuestionExtractorGoogleCloud ready")

    def __enter__(self) -> "QuestionExtractorGoogleCloud":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self.executor:
            self.executor.shutdown(wait=True)

    # ── public API ────────────────────────────────────────────────────────────

    def process_exam(
        self,
        folder_path: Path,
        is_submission: bool = False,
        save_results: bool = False,
        output_path: Optional[Path] = None,
    ) -> List[Dict]:
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
                output_path or Path("question_analysis_results.json"),
            )
        return results

    def process_image(self, image_path: Path, is_submission: bool) -> Dict:
        self.logger.info(f"Processing: {image_path.name}")
        try:
            with Image.open(image_path) as image:

                # ── Step 1: OCR ───────────────────────────────────────────────
                ocr_text = self._ocr_extract_text(image)

                if not ocr_text.strip():
                    self.logger.warning(f"{image_path.name}: OCR returned empty text")
                    return self._empty_result(image_path)

                # ── Step 2: Classify using CLASSIFICATION_PROMPT from prompts.py
                question_type, confidence = self._classify(ocr_text)

                # ── Step 3: Pick template from prompts.py ─────────────────────
                templates = (
                    TEMPLATES_SUBMISSIONS_PROMPT
                    if is_submission
                    else TEMPLATES_CORRECTION_PROMPT
                )
                # Fall back to UNKNOWN template if type not in mapping
                template = templates.get(question_type) or templates.get("UNKNOWN")

                # ── Step 4: Extract using template from prompts.py ────────────
                content = self._extract(ocr_text, template)

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

    # ── Step 1: OCR ───────────────────────────────────────────────────────────

    def _ocr_extract_text(self, image: Image.Image) -> str:
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
                ann  = response.full_text_annotation
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
                delay = min(_OCR_MAX_DELAY_S, _OCR_BASE_DELAY_S * (2 ** attempt))
                self.logger.warning(
                    f"OCR transient error — retrying in {delay}s "
                    f"({attempt}/{_MAX_OCR_RETRIES})"
                )
                time.sleep(delay)

    # ── Mistral call ──────────────────────────────────────────────────────────

    def _mistral_call(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       MISTRAL_MODEL,
            "messages":    [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": _TEMPERATURE,
            "max_tokens":  _MAX_TOKENS,
        }

        attempt = 0
        while True:
            try:
                response = requests.post(
                    MISTRAL_URL, headers=headers, json=payload, timeout=60
                )
                if response.status_code == 429:
                    attempt += 1
                    if attempt > _MAX_LLM_RETRIES:
                        raise RuntimeError("Mistral rate limit exceeded after retries")
                    delay = _LLM_RETRY_DELAY * attempt
                    self.logger.warning(
                        f"Rate limit — retrying in {delay}s ({attempt}/{_MAX_LLM_RETRIES})"
                    )
                    time.sleep(delay)
                    continue

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Mistral error {response.status_code}: {response.text}"
                    )

                return response.json()["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout:
                attempt += 1
                if attempt > _MAX_LLM_RETRIES:
                    raise RuntimeError("Mistral request timed out after retries")
                self.logger.warning(f"Timeout — retrying ({attempt}/{_MAX_LLM_RETRIES})")
                time.sleep(_LLM_RETRY_DELAY)

    # ── Step 2: Classify ─────────────────────────────────────────────────────

    def _classify(self, ocr_text: str) -> Tuple[str, float]:
        system = (
            "You are an expert exam question classifier. "
            "Return ONLY valid JSON: "
            "{\"question_type\": \"...\", \"confidence\": 0.0, \"reasoning\": \"...\"}"
        )
        user = (
            f"{CLASSIFICATION_PROMPT}\n\n"
            "NOTE: Classify based on OCR-extracted text below (no image available).\n\n"
            f"OCR TEXT:\n{ocr_text}"
        )

        raw  = self._mistral_call(system, user)
        data = self._parse_json(raw)

        question_type = str(data.get("question_type", "UNKNOWN")).upper()
        confidence    = float(data.get("confidence", 0.5))

        self.logger.info(
            f"[classify] {question_type} ({confidence:.2f}) "
            f"— {data.get('reasoning', '')[:80]}"
        )
        return question_type, confidence

    # ── Step 3: Extract ───────────────────────────────────────────────────────

    def _extract(self, ocr_text: str, template: str) -> Dict:
        system = (
            "You are an expert exam content extractor. "
            "Return ONLY valid JSON — no markdown, no extra text."
        )
        user = (
            f"{template}\n\n"
            "NOTE: Extract from OCR text below (no image available).\n\n"
            f"OCR TEXT:\n{ocr_text}"
        )

        raw     = self._mistral_call(system, user)
        content = self._parse_json(raw)

        if content.get("error"):
            self.logger.warning(f"Extraction issue: {content.get('error')}")

        return content

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> Dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$",          "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            return {"raw_text": text, "error": f"Failed to parse JSON: {exc}"}

    @staticmethod
    def _extract_section_number(image_name: str) -> int:
        m = re.search(r"section[_\s-](\d+)", image_name, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)", image_name)
        return int(m.group(1)) if m else 999

    @staticmethod
    def _empty_result(image_path: Path) -> Dict:
        return {
            "question_type": "UNKNOWN",
            "confidence":    0.0,
            "content":       {"error": "OCR returned empty text"},
            "meta_data": {
                "image_path": str(image_path),
                "image_name": image_path.name,
            },
        }

    @staticmethod
    def _error_result(image_path: Path, exc: Exception) -> Dict:
        return {
            "error":         str(exc),
            "question_type": "ERROR",
            "confidence":    0.0,
            "meta_data": {
                "image_path": str(image_path),
                "image_name": image_path.name,
            },
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