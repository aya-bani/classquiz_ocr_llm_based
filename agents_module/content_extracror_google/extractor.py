# extractor.py

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import dotenv
from google.cloud import vision
from google.oauth2 import service_account

dotenv.load_dotenv()

try:
    from .prompt import SYSTEM_PROMPT, build_user_prompt
    from .utils import (
        OcrBlock,
        assemble_text_rtl,
        blocks_in_zone,
        clean_arabic_text,
        confidence_emoji,
        confidence_label,
        filter_by_confidence,
        is_handwritten_candidate,
        is_placeholder_only,
        remove_placeholders,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompt import SYSTEM_PROMPT, build_user_prompt
    from utils import (
        OcrBlock,
        assemble_text_rtl,
        blocks_in_zone,
        clean_arabic_text,
        confidence_emoji,
        confidence_label,
        filter_by_confidence,
        is_handwritten_candidate,
        is_placeholder_only,
        remove_placeholders,
    )

GOOGLE_CREDENTIALS_PATH: Optional[str] = None
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

@dataclass
class ExtractionResult:
    student_answer: str = ""
    raw_vision_text: str = ""
    confidence: float = 0.0
    blocks: List[OcrBlock] = field(default_factory=list)

def _build_vision_client():
    if GOOGLE_CREDENTIALS_PATH:
        credentials = service_account.Credentials.from_service_account_file(
            GOOGLE_CREDENTIALS_PATH
        )
        return vision.ImageAnnotatorClient(credentials=credentials)

    if GOOGLE_API_KEY:
        return vision.ImageAnnotatorClient(
            client_options={"api_key": GOOGLE_API_KEY}
        )

    return vision.ImageAnnotatorClient()

class ArabicHandwrittenExtractor:
    def extract_folder(
        self,
        folder_path: str | Path,
        section_type: str = "unknown",
        question_text: str = "",
        output_path: str | Path = None,
    ):
        """Extract all images in a folder and save results to JSON."""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
        if self.verbose:
            print(f"[INFO] Found {len(images)} image(s) in {folder}")

        results = []
        for img_path in images:
            if self.verbose:
                print(f"[→] {img_path.name}")
            result = self.extract_image(img_path)
            results.append(result)

        # Determine output path
        if output_path is None:
            out_path = folder.parent / "exam_1.json"
        else:
            out_path = Path(output_path)
        print(f"[DEBUG] Attempting to write output to: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize results
        def result_to_dict(result, idx):
            return {
                "section_number": idx + 1,
                "question_type": section_type if section_type else "UNKNOWN",
                "question": question_text if question_text else "",
                "options": None,
                "student_answer": result.student_answer,
                "metadata": None,
                "confidence": round(result.confidence, 4),
            }

        # Only include results with a non-empty student_answer (handwritten detected)
        filtered_results = [r for r in results if r.student_answer and r.student_answer.strip()]
        output_list = [result_to_dict(r, i) for i, r in enumerate(filtered_results)]
        try:
            out_path.write_text(json.dumps(output_list, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[INFO] Results saved → {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write output file: {e}")
        return results

    def __init__(self, min_block_confidence=0.4, verbose=False):
        self.min_block_confidence = min_block_confidence
        self.verbose = verbose
        self._vision_client = _build_vision_client()

    def extract_image(self, image_path: str | Path) -> ExtractionResult:
        path = Path(image_path)
        result = ExtractionResult()

        raw_text, blocks = self._run_vision(path)
        result.raw_vision_text = raw_text

        if not blocks and not raw_text:
            return result

        blocks = filter_by_confidence(blocks, self.min_block_confidence)
        blocks = [b for b in blocks if is_handwritten_candidate(b.text, b.confidence)]

        # Always use LLM post-processing to extract only handwritten answer
        assembled = remove_placeholders(clean_arabic_text(assemble_text_rtl(blocks)))
        llm_answer = ""
        try:
            from .prompt import build_user_prompt, SYSTEM_PROMPT
            import openai
            # Use OpenAI API key from env
            openai.api_key = os.getenv("OPENAI_API_KEY")
            user_prompt = build_user_prompt(
                raw_ocr_text=assembled or raw_text,
                section_type="unknown",
                question_text="",
                avg_confidence=sum(b.confidence for b in blocks) / len(blocks) if blocks else 0.0,
                has_dotted_line=False,
            )
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=300,
            )
            llm_answer = (response["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            if self.verbose:
                print(f"[LLM ERROR] {e}")
            llm_answer = assembled

        result.student_answer = llm_answer
        result.confidence = sum(b.confidence for b in blocks) / len(blocks) if blocks else 0.0
        result.blocks = blocks

        if self.verbose:
            print(f"[RESULT] {result.student_answer}")

        return result

    def _run_vision(self, image_path: Path):

        with open(image_path, "rb") as f:
            content = f.read()

        image = vision.Image(content=content)

        response = self._vision_client.document_text_detection(
            image=image,
            image_context={"language_hints": ["ar"]}  # 🔥 FIX
        )

        if response.error.message:
            raise RuntimeError(response.error.message)

        annotation = response.full_text_annotation

        if not annotation or not annotation.text:  # 🔥 FIX
            return "", []

        if self.verbose:
            print(f"[RAW OCR]: {annotation.text[:200]}...")

        return annotation.text, self._parse(annotation)

    def _parse(self, annotation):

        blocks = []

        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:

                        text = "".join([s.text for s in word.symbols])

                        sym_confs = [
                            s.confidence for s in word.symbols
                            if hasattr(s, "confidence")  # 🔥 FIX
                        ]

                        conf = sum(sym_confs) / len(sym_confs) if sym_confs else 0.0

                        bbox = [
                            {"x": v.x, "y": v.y}
                            for v in word.bounding_box.vertices
                        ]

                        blocks.append(OcrBlock(text, conf, bbox))

        return blocks