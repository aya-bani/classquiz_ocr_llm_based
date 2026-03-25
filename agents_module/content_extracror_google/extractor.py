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

        if not blocks:
            cleaned = remove_placeholders(clean_arabic_text(raw_text))
            if cleaned and not is_placeholder_only(cleaned):
                result.student_answer = cleaned
                result.confidence = 0.4
            return result

        assembled = remove_placeholders(clean_arabic_text(assemble_text_rtl(blocks)))

        result.student_answer = assembled
        result.confidence = sum(b.confidence for b in blocks) / len(blocks)
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