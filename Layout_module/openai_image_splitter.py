"""
openai_image_splitter.py
------------------------
Splits exam pages into sections using OpenAI Vision API + OpenCV.

Structure mirrors image_splitter.py (Google Vision version):
  - detect_section_lines()  →  runs OpenAI Vision OCR (y_percent, proven working),
                                snaps to OpenCV bands, returns sorted
                                (x_min, y_min, x_max, y_max) list
  - split_image()           →  identical cropping loop to ImageSplitter.split_image()
  - split_and_save()        →  identical to ImageSplitter.split_and_save()

Detection uses the original y_percent prompt (not pixel bounding boxes) because
OpenAI Vision reliably estimates relative position but struggles with exact pixels.
The y_percent is converted to absolute Y then snapped to the nearest OpenCV text band.
"""

from __future__ import annotations

import re
import base64
import json
import sys
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_manager import LoggerManager
from Layout_module.layout_config import LayoutConfig


# ---------------------------------------------------------------------------
# Arabic normalization
# ---------------------------------------------------------------------------

_DIACRITICS = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670'
    r'\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]'
)
_TATWEEL = re.compile(r'\u0640')
_ALEF    = re.compile(r'[\u0622\u0623\u0625\u0671\u0627]')
_YA      = re.compile(r'\u0649')
_AL      = re.compile(r'^ال')


def _normalize(text: str) -> str:
    text = _DIACRITICS.sub('', text)
    text = _TATWEEL.sub('', text)
    text = _ALEF.sub('ا', text)
    text = _YA.sub('ي', text)
    return unicodedata.normalize('NFC', text)


def _root(kw: str) -> str:
    return _AL.sub('', _normalize(kw))


# ---------------------------------------------------------------------------
# OpenCV text-band helpers
# ---------------------------------------------------------------------------

def find_text_bands(image: np.ndarray, min_height: int = 8) -> List[tuple]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
           if len(image.shape) == 3 else image.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    h_proj = np.sum(dilated, axis=1)
    threshold = image.shape[1] * 3
    bands, in_band, band_start = [], False, 0
    for y, val in enumerate(h_proj):
        if not in_band and val > threshold:
            in_band, band_start = True, y
        elif in_band and val <= threshold:
            in_band = False
            if y - band_start >= min_height:
                bands.append((band_start, y))
    if in_band and (len(h_proj) - band_start) >= min_height:
        bands.append((band_start, len(h_proj)))
    return bands


def anchor_to_nearest_band(
    estimated_y: int,
    bands: List[tuple],
    search_radius: int = 300,
) -> Optional[int]:
    best_dist, best_y = search_radius + 1, None
    for y0, y1 in bands:
        dist = abs((y0 + y1) // 2 - estimated_y)
        if dist < best_dist:
            best_dist, best_y = dist, y0
    return best_y


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------

class OpenAIImageSplitter:

    CHUNK_HEIGHT  = 1500
    CHUNK_OVERLAP = 200

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.logger = LoggerManager.get_logger(__name__)

        self.api_key    = api_key    or LayoutConfig.OPENAI_API_KEY
        self.model_name = model_name or LayoutConfig.OPENAI_MODEL_NAME

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")

        self.client = OpenAI(api_key=self.api_key)

        self.keywords          = LayoutConfig.KEY_WORDS
        self.excluded_keywords = LayoutConfig.EXCLUDED_KEYWORDS

        self._kw_roots   = {kw: _root(kw) for kw in self.keywords}
        self._excl_roots = [_root(e) for e in self.excluded_keywords]

        self.logger.info(
            f"OpenAIImageSplitter ready | model={self.model_name} "
            f"| keywords={self.keywords} "
            f"| chunk_height={self.CHUNK_HEIGHT}px overlap={self.CHUNK_OVERLAP}px"
        )

    # ------------------------------------------------------------------
    # 1. detect_section_lines
    #    Mirrors: ImageSplitter.detect_section_lines()
    #    Returns: List of (x_min, y_min, x_max, y_max) sorted top→bottom
    # ------------------------------------------------------------------

    def detect_section_lines(
        self, image: Image.Image
    ) -> List[Tuple[int, int, int, int]]:
        """
        Run OpenAI Vision OCR on *image* in overlapping chunks.
        Each chunk uses the original y_percent prompt (proven reliable).
        y_percent is converted to absolute Y then snapped to nearest OpenCV band.
        Returns sorted [(x_min, y_min, x_max, y_max)] — same as Google Vision version.
        """
        self.logger.info("Running OpenAI OCR to detect section markers")

        cv_image      = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        bands         = find_text_bands(cv_image)
        width, total_height = image.size

        self.logger.debug(f"OpenCV found {len(bands)} text bands")

        raw_coords: List[Tuple[int, int, int, int]] = []
        chunk_idx = 0
        y_top     = 0

        while y_top < total_height:
            y_bot   = min(y_top + self.CHUNK_HEIGHT, total_height)
            chunk   = image.crop((0, y_top, width, y_bot))
            chunk_h = y_bot - y_top

            self.logger.info(f"Chunk {chunk_idx}: y={y_top}–{y_bot} ({chunk_h}px)")

            # original y_percent detection — returns (keyword, text, estimated_local_y)
            chunk_matches = self._openai_extract(chunk, chunk_h)

            for keyword, word_text, local_y in chunk_matches:
                abs_y = y_top + local_y

                # snap to nearest OpenCV text band
                snapped_y = anchor_to_nearest_band(abs_y, bands, search_radius=300)
                if snapped_y is None:
                    self.logger.warning(
                        f"Could not snap '{keyword}' y={abs_y} – using raw value"
                    )
                    snapped_y = abs_y

                # use full image width for x extent (we only care about y for splitting)
                raw_coords.append((0, snapped_y, width, snapped_y + 30))

                self.logger.debug(
                    f"  Chunk {chunk_idx}: '{keyword}' "
                    f"local_y={local_y} → abs_y={abs_y} → snapped={snapped_y}"
                )
                print("-----------------Word detected:-------------------")
                print(f"  keyword='{keyword}'  text='{word_text}'  y={snapped_y}")

            if y_bot >= total_height:
                break
            y_top     = y_bot - self.CHUNK_OVERLAP
            chunk_idx += 1

        # deduplicate and sort top→bottom by y_min
        section_coords = self._deduplicate_coords(raw_coords, proximity=60)
        section_coords.sort(key=lambda c: c[1])

        self.logger.info(f"Detected {len(section_coords)} section lines")
        print("-----------------OCR Response-------------------")
        for coord in section_coords:
            print(f"  Section line bbox: {coord}")

        return section_coords

    # ------------------------------------------------------------------
    # 2. split_image
    #    Mirrors: ImageSplitter.split_image() — identical cropping loop
    # ------------------------------------------------------------------

    def split_image(self, image: Image.Image) -> List[Image.Image]:
        """
        Split *image* into sections based on keyword bounding boxes.
        Cropping logic is identical to ImageSplitter.split_image().
        """
        self.logger.info("Splitting image into sections")

        line_coords = self.detect_section_lines(image)

        if not line_coords:
            self.logger.warning("No section lines detected; returning full image")
            return [image]

        img_array     = np.array(image)
        height, width = img_array.shape[:2]
        sections      = []
        y_start       = 0

        for i, coords in enumerate(line_coords):
            x_min, y_min, x_max, y_max = coords

            if i == 0:
                # slice from top to first keyword (header, may be empty)
                crop = img_array[y_start:y_min, 0:width]
                if crop.size != 0:
                    sections.append(Image.fromarray(crop))
                y_start = y_min
            else:
                # slice from previous keyword to this keyword
                crop = img_array[y_start:y_min, 0:width]
                if crop.size != 0:
                    sections.append(Image.fromarray(crop))
                y_start = y_min

        # last section: final keyword → bottom of image
        crop = img_array[y_start:height, 0:width]
        if crop.size != 0:
            sections.append(Image.fromarray(crop))
            self.logger.debug(
                f"Created section {len(sections)} from y={y_start} to y={height}"
            )

        self.logger.info(f"Total sections created: {len(sections)}")
        return sections

    # ------------------------------------------------------------------
    # 3. split_and_save
    #    Mirrors: ImageSplitter.split_and_save() — identical signature + return
    # ------------------------------------------------------------------

    def split_and_save(
        self,
        image: Image.Image,
        exam_id: int,
        submission_id: Optional[int] = None,
    ) -> dict:
        self.logger.info("Starting split_and_save")
        sections = self.split_image(image)

        if submission_id is not None:
            output_prefix = f"exam_{exam_id}_submission_{submission_id}"
            directory     = LayoutConfig.RAW_SUBMISSIONS_PATH / output_prefix
        else:
            output_prefix = f"exam_{exam_id}"
            directory     = LayoutConfig.RAW_EXAMS_PATH / output_prefix

        directory.mkdir(parents=True, exist_ok=True)

        for i, section in enumerate(sections):
            filepath = directory / f"{output_prefix}_section_{i}.jpg"
            section.save(filepath, "JPEG")
            self.logger.info(f"Saved section {i} to {filepath}")

        self.logger.info("Finished split_and_save")
        return {
            'sections_dir':       directory,
            'number_of_sections': len(sections),
        }

    # ------------------------------------------------------------------
    # OpenAI Vision — single chunk (original y_percent prompt, unchanged)
    # Returns: List of (keyword, text, estimated_local_y)
    # ------------------------------------------------------------------

    def _openai_extract(
        self, image: Image.Image, image_height: int
    ) -> List[Tuple[str, str, int]]:

        b64    = self._pil_to_base64(image)
        prompt = self._build_prompt()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=2000,
                temperature=0,
            )

            raw = response.choices[0].message.content
            self.logger.debug(f"OpenAI raw:\n{raw[:500]}")
            return self._parse_response(raw, image_height)

        except Exception as exc:
            self.logger.error(f"OpenAI API error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Prompt  (original, unchanged — y_percent is reliable)
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        kw_list   = " | ".join(self.keywords)
        excl_list = " | ".join(self.excluded_keywords) if self.excluded_keywords else "none"
        return f"""
You are analyzing a portion of an Arabic exam page image.

STEP 1 — Read every line of text in this image carefully.

STEP 2 — Find EVERY line that contains any of these keywords
(with OR without diacritics, with OR without the prefix ال):
  {kw_list}
  Examples: سند، السند، سند:، تعليمة، التعليمة، تعليمة:

STEP 3 — Skip lines where the word is solely part of these excluded words:
  {excl_list}

STEP 4 — For each match, estimate y_percent: how far down THIS image the line
appears, as a float from 0.0 (top of this image) to 1.0 (bottom of this image).

STEP 5 — Output a JSON array sorted top to bottom. If no keywords found, return [].

Format:
[
  {{
    "keyword": "سند",
    "text": "السند 1: شارك شادي في رحلة مدرسية",
    "y_percent": 0.08,
    "vertical_order": 1
  }},
  {{
    "keyword": "تعليمة",
    "text": "تعليمة 1: أحسب مبلغ الأب",
    "y_percent": 0.45,
    "vertical_order": 2
  }}
]

Rules:
- keyword        : exactly "تعليمة" or "سند" (no diacritics, no ال)
- text           : exact line as seen in the image
- y_percent      : 0.0–1.0 relative to THIS image only
- vertical_order : 1 = topmost in this image

Your entire response must be only the JSON array, nothing else.
"""

    # ------------------------------------------------------------------
    # Parsing (original, unchanged)
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: str, image_height: int
    ) -> List[Tuple[str, str, int]]:
        """Returns list of (keyword, text, estimated_local_y)."""
        data = self._safe_json(raw)
        if data is not None and isinstance(data, list):
            return self._from_blocks(data, image_height)
        self.logger.warning("JSON parse failed – falling back to line scan")
        return self._from_lines(raw, image_height)

    @staticmethod
    def _safe_json(raw: str):
        text = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None

    def _from_blocks(
        self, blocks: list, image_height: int
    ) -> List[Tuple[str, str, int]]:
        results = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            combined     = block.get("text", "") + " " + block.get("keyword", "")
            confirmed_kw = self._detect_keyword(combined)
            if confirmed_kw is None:
                continue
            y_pct        = max(0.0, min(1.0, float(block.get("y_percent", 0.5))))
            estimated_y  = int(y_pct * image_height)
            results.append((confirmed_kw, block.get("text", ""), estimated_y))
        return results

    def _from_lines(
        self, raw: str, image_height: int
    ) -> List[Tuple[str, str, int]]:
        lines  = [l.strip() for l in raw.splitlines() if l.strip()]
        total  = max(len(lines), 1)
        results = []
        for idx, line in enumerate(lines):
            kw = self._detect_keyword(line)
            if kw is None:
                continue
            y = int((idx / total) * image_height)
            results.append((kw, line, y))
        return results

    # ------------------------------------------------------------------
    # Keyword detection (original, unchanged)
    # ------------------------------------------------------------------

    def _detect_keyword(self, text: str) -> Optional[str]:
        norm = _normalize(text)
        for excl in self._excl_roots:
            if re.search(r'(?<!\S)' + re.escape(excl) + r'(?!\S)', norm):
                return None
        for kw, root in self._kw_roots.items():
            if root in norm:
                return kw
        return None

    # ------------------------------------------------------------------
    # Deduplication — on (x_min, y_min, x_max, y_max) tuples
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_coords(
        coords: List[Tuple[int, int, int, int]], proximity: int = 60
    ) -> List[Tuple[int, int, int, int]]:
        if not coords:
            return []
        coords = sorted(coords, key=lambda c: c[1])
        out = [coords[0]]
        for c in coords[1:]:
            if abs(c[1] - out[-1][1]) > proximity:
                out.append(c)
        return out

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")