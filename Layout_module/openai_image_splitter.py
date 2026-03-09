"""
openai_image_splitter.py
------------------------
Splits exam pages into sections using OpenAI Vision API + OpenCV.

  - detect_section_lines()  →  OpenAI Vision OCR (y_percent) + OpenCV band snapping
  - highlight_keywords()    →  returns image copy with yellow bars on keyword rows
  - split_image()           →  crops into sections (keyword row → next keyword row)
  - split_and_save()        →  saves sections to disk, returns metadata dict
"""

from __future__ import annotations

import re
import base64
import json
import sys
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from logger_manager import LoggerManager
    from Layout_module.layout_config import LayoutConfig
except ImportError:
    class LoggerManager:
        @staticmethod
        def get_logger(name):
            import logging
            return logging.getLogger(name)
    class LayoutConfig:
        OPENAI_API_KEY     = ""
        OPENAI_MODEL_NAME  = "gpt-4o"
        KEY_WORDS          = ["سند", "تعليمة"]
        EXCLUDED_KEYWORDS  = []
        RAW_EXAMS_PATH     = Path("data/Sections/exams")
        RAW_SUBMISSIONS_PATH = Path("data/Sections/submissions")


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

    """Detects horizontal text lines using morphological dilation and projection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
           if len(image.shape) == 3 else image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Wide kernel bridges gaps between Arabic words
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    h_proj    = np.sum(dilated, axis=1)
    threshold = (image.shape[1] * 255) * 0.03   # 3 % of row width

    bands, in_band, band_start = [], False, 0
    for y, val in enumerate(h_proj):
        if not in_band and val > threshold:
            in_band, band_start = True, y
        elif in_band and val <= threshold:
            in_band = False
            if y - band_start >= min_height:
                bands.append((band_start, y))

    if in_band:
        bands.append((band_start, len(h_proj)))

    return bands


def anchor_to_nearest_band(
    estimated_y: int,
    bands: List[tuple],
    search_radius: int = 400,
) -> Optional[Tuple[int, int]]:
    """Return the (y0, y1) of the text band whose centre is closest to estimated_y."""
    best_dist = search_radius + 1
    best_band = None
    for y0, y1 in bands:
        dist = abs((y0 + y1) // 2 - estimated_y)
        if dist < best_dist:
            best_dist = dist
            best_band = (y0, y1)
    return best_band


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class KeywordDetection:
    keyword: str
    text:    str
    x_min:   int
    y_min:   int
    x_max:   int
    y_max:   int


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------

class OpenAIImageSplitter:

    CHUNK_HEIGHT        = 1500
    CHUNK_OVERLAP       = 250
    HIGHLIGHT_BAR_COLOR = (255, 220, 0)   # yellow RGB
    HIGHLIGHT_ALPHA     = 0.4             # semi-transparent overlay

    def __init__(
        self,
        api_key:    Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.logger     = LoggerManager.get_logger(__name__)
        self.api_key    = api_key    or LayoutConfig.OPENAI_API_KEY
        self.model_name = model_name or LayoutConfig.OPENAI_MODEL_NAME

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")

        self.client            = OpenAI(api_key=self.api_key)
        self.keywords          = LayoutConfig.KEY_WORDS
        self.excluded_keywords = LayoutConfig.EXCLUDED_KEYWORDS
        self._kw_roots         = {kw: _root(kw) for kw in self.keywords}
        self._excl_roots       = [_root(e) for e in self.excluded_keywords]

        # populated by detect_section_lines() — available for callers
        self._last_detected_keywords: List[KeywordDetection] = []

        self.logger.info(
            f"OpenAIImageSplitter ready | model={self.model_name} "
            f"| keywords={self.keywords} "
            f"| chunk={self.CHUNK_HEIGHT}px overlap={self.CHUNK_OVERLAP}px"
        )

    # ------------------------------------------------------------------
    # 1. detect_section_lines
    #    Returns: List[(x_min, y_min, x_max, y_max)] sorted top→bottom
    # ------------------------------------------------------------------

    def detect_section_lines(
        self, image: Image.Image
    ) -> List[Tuple[int, int, int, int]]:
        """
        Run OpenAI Vision OCR on overlapping chunks of *image*.
        y_percent is converted directly to absolute Y.
        No OpenCV band snapping — OpenAI detection is accurate and snapping introduces drift.
        Returns sorted [(x_min, y_min, x_max, y_max)].
        """
        self.logger.info("Running OpenAI OCR to detect section markers")

        width, total_height = image.size
        raw_detections: List[KeywordDetection] = []
        chunk_idx = 0
        y_top     = 0

        while y_top < total_height:
            y_bot   = min(y_top + self.CHUNK_HEIGHT, total_height)
            chunk   = image.crop((0, y_top, width, y_bot))
            chunk_h = y_bot - y_top

            self.logger.info(f"Chunk {chunk_idx}: y={y_top}-{y_bot} ({chunk_h}px)")

            chunk_matches = self._openai_extract(chunk, chunk_h)

            for keyword, word_text, local_y in chunk_matches:
                # convert chunk-local y directly to absolute y — no OpenCV snapping
                abs_y = y_top + local_y

                raw_detections.append(KeywordDetection(
                    keyword=keyword, text=word_text,
                    x_min=0, y_min=abs_y, x_max=width, y_max=abs_y + 40,
                ))

                self.logger.debug(
                    f"  Chunk {chunk_idx}: '{keyword}' "
                    f"local_y={local_y} -> abs_y={abs_y}"
                )
                print("-----------------Word detected:-------------------")
                print(f"  keyword='{keyword}'  text='{word_text}'  y={abs_y}")

            if y_bot >= total_height:
                break
            y_top = y_bot - self.CHUNK_OVERLAP
            chunk_idx += 1

        deduped = self._deduplicate_detections(raw_detections, proximity=self.CHUNK_OVERLAP)
        deduped.sort(key=lambda d: d.y_min)
        self._last_detected_keywords = deduped

        section_coords = [(d.x_min, d.y_min, d.x_max, d.y_max) for d in deduped]

        self.logger.info(f"Detected {len(section_coords)} section lines")
        print("-----------------OCR Response-------------------")
        for coord in section_coords:
            print(f"  Section line bbox: {coord}")

        return section_coords

    # ------------------------------------------------------------------
    # 2. highlight_keywords
    #    Returns a copy of the image with yellow bars on keyword rows
    # ------------------------------------------------------------------

    def highlight_keywords(
        self,
        image: Image.Image,
        section_coords: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Image.Image:
        """
        Draw a semi-transparent yellow highlight bar over every detected
        keyword row.  The original image is NOT modified.

        Parameters
        ----------
        image          : original PIL image
        section_coords : output of detect_section_lines(); computed automatically
                         if not provided.

        Returns
        -------
        New PIL Image (RGB) with yellow bars drawn over keyword rows.
        """
        if section_coords is None:
            section_coords = self.detect_section_lines(image)

        if not section_coords:
            self.logger.warning("highlight_keywords: nothing to highlight")
            return image.copy()

        base    = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)

        r, g, b = self.HIGHLIGHT_BAR_COLOR
        alpha   = int(255 * self.HIGHLIGHT_ALPHA)

        for (x_min, y_min, x_max, y_max) in section_coords:
            draw.rectangle([(0, y_min), (image.width, y_max)], fill=(r, g, b, alpha))

        highlighted = Image.alpha_composite(base, overlay).convert("RGB")
        self.logger.info(
            f"highlight_keywords: drew {len(section_coords)} yellow bars"
        )
        return highlighted

    # ------------------------------------------------------------------
    # 3. split_image
    # ------------------------------------------------------------------

    def split_image(self, image: Image.Image) -> List[Image.Image]:
        """
        Split *image* into sections based on detected keyword positions.
        Header content above the first keyword is always ignored.
        Sections always start at the first detected keyword row.
        """
        self.logger.info("Splitting image into sections")

        line_coords = self.detect_section_lines(image)

        if not line_coords:
            self.logger.warning("No section lines detected; returning full image")
            return [image]

        img_array     = np.array(image)
        height, width = img_array.shape[:2]
        sections      = []
        y_start       = line_coords[0][1]

        self.logger.debug(
            f"Ignoring header above first keyword: y=0-{y_start}"
        )

        for i, coords in enumerate(line_coords):
            x_min, y_min, x_max, y_max = coords

            if i == 0:
                continue

            # slice from previous keyword's y_min to this keyword's y_min
            crop = img_array[y_start:y_min, 0:width]
            if crop.size != 0:
                sections.append(Image.fromarray(crop))
                self.logger.debug(f"Section {len(sections)-1}: y={y_start}-{y_min} ({y_min-y_start}px)")
            # next section starts AT this keyword row
            y_start = y_min

        # last section: from the final keyword's y_min to the bottom
        crop = img_array[y_start:height, 0:width]
        if crop.size != 0:
            sections.append(Image.fromarray(crop))
            self.logger.debug(f"Section {len(sections)-1} (last): y={y_start}-{height} ({height-y_start}px)")

        self.logger.info(f"Total sections created: {len(sections)}")
        return sections


    # ------------------------------------------------------------------
    # 4. split_and_save
    # ------------------------------------------------------------------

    def split_and_save(
        self,
        image: Image.Image,
        exam_id: int,
        submission_id: Optional[int] = None,
    ) -> dict:
        self.logger.info("Starting split_and_save")
        sections = self.split_image(image)

        output_prefix = (
            f"exam_{exam_id}_submission_{submission_id}"
            if submission_id is not None
            else f"exam_{exam_id}"
        )
        directory = (
            LayoutConfig.RAW_SUBMISSIONS_PATH / output_prefix
            if submission_id is not None
            else LayoutConfig.RAW_EXAMS_PATH / output_prefix
        )
        directory.mkdir(parents=True, exist_ok=True)

        for i, section in enumerate(sections):
            filepath = directory / f"{output_prefix}_section_{i}.jpg"
            section.save(filepath, "JPEG", quality=95)
            self.logger.info(f"Saved section {i} to {filepath}")

        self.logger.info("Finished split_and_save")
        return {
            'sections_dir':       directory,
            'number_of_sections': len(sections),
        }

    # ------------------------------------------------------------------
    # OpenAI Vision — single chunk
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
                messages=[{
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
                }],
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
    # Prompt
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
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: str, image_height: int
    ) -> List[Tuple[str, str, int]]:
        """Returns list of (keyword, text, estimated_local_y)."""
        # strip markdown fences if model wraps output
        text = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$',        '', text,        flags=re.MULTILINE).strip()

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\[.*\]', text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if isinstance(data, list):
            return self._from_blocks(data, image_height)

        self.logger.warning("JSON parse failed – falling back to line scan")
        return self._from_lines(raw, image_height)

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
            y_pct       = max(0.0, min(1.0, float(block.get("y_percent", 0.5))))
            estimated_y = int(y_pct * image_height)
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
    # Keyword detection
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
    # Deduplication — operates on KeywordDetection objects
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_detections(
        detections: List[KeywordDetection], proximity: int = 250
    ) -> List[KeywordDetection]:
        """
        Remove duplicate detections within *proximity* pixels in Y.
        Keeps the detection with the lowest y_min (topmost).
        proximity should be >= CHUNK_OVERLAP so the same keyword seen in
        the tail of one chunk and the head of the next is merged.
        """
        if not detections:
            return []
        detections = sorted(detections, key=lambda d: d.y_min)
        out = [detections[0]]
        for d in detections[1:]:
            if abs(d.y_min - out[-1].y_min) > proximity:
                out.append(d)
        return out

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")