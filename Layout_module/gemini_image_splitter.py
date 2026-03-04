"""
gemini_image_splitter.py
-----------------------
Splits exam pages into sections using Gemini Vision API + OpenCV.

Workflow:
1. Gemini identifies which keywords exist and their rough position hint
2. OpenCV finds actual pixel Y-coordinates of text bands
3. Keyword text bands are matched to real pixel positions
4. Image is split at exact pixel boundaries
"""

from __future__ import annotations

import re
import time
import json
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai

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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KeywordMatch:
    keyword: str        # original keyword from config e.g. "تعليمة"
    text: str           # full line text containing it
    y_position: int     # pixel Y (top of that text band)


@dataclass
class ImageSection:
    section_index: int
    keyword_trigger: Optional[str]
    y_start: int
    y_end: int
    image: Image.Image


# ---------------------------------------------------------------------------
# OpenCV text-band detector
# ---------------------------------------------------------------------------

def find_text_bands(image: np.ndarray, min_height: int = 8) -> List[tuple]:
    """
    Returns list of (y_start, y_end) for every horizontal text band
    detected via morphological projection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
           if len(image.shape) == 3 else image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge characters into full text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    h_proj = np.sum(dilated, axis=1)
    threshold = image.shape[1] * 3

    bands = []
    in_band = False
    band_start = 0

    for y, val in enumerate(h_proj):
        if not in_band and val > threshold:
            in_band = True
            band_start = y
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
    search_radius: int = 200,
) -> Optional[int]:
    """
    Returns y_start of the text band whose centre is closest to estimated_y,
    or None if nothing found within search_radius.
    """
    best_dist = search_radius + 1
    best_y = None

    for y0, y1 in bands:
        centre = (y0 + y1) // 2
        dist = abs(centre - estimated_y)
        if dist < best_dist:
            best_dist = dist
            best_y = y0

    return best_y


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------

class GeminiImageSplitter:

    _ZONES = {
        "top":    (0.00, 0.15),
        "upper":  (0.15, 0.38),
        "middle": (0.38, 0.62),
        "lower":  (0.62, 0.85),
        "bottom": (0.85, 1.00),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.logger = LoggerManager.get_logger(__name__)

        self.api_key    = api_key    or LayoutConfig.GEMINI_API_KEY
        self.model_name = model_name or LayoutConfig.GEMINI_MODEL_NAME

        if not self.api_key:
            raise ValueError("GEMINI_AI_API_KEY is required")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        self.keywords          = LayoutConfig.KEY_WORDS
        self.excluded_keywords = LayoutConfig.EXCLUDED_KEYWORDS

        self._kw_roots   = {kw: _root(kw) for kw in self.keywords}
        self._excl_roots = [_root(e) for e in self.excluded_keywords]

        self.logger.info(
            f"GeminiImageSplitter ready | model={self.model_name} "
            f"| keywords={self.keywords}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_image_by_keywords(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        min_section_height: int = 100,
    ) -> List[ImageSection]:

        pil_image = self._to_pil(image)
        cv_image  = self._to_cv(pil_image)
        width, height = pil_image.size

        self.logger.info(f"Processing image {width}x{height}")

        # Step 1 — Gemini: get keywords with rough vertical positions
        raw_matches = self._gemini_extract(pil_image, height)

        if not raw_matches:
            self.logger.warning("No keywords found – returning full image")
            return [ImageSection(0, None, 0, height, pil_image)]

        # Step 2 — OpenCV: find all real text bands
        bands = find_text_bands(cv_image)
        self.logger.debug(f"OpenCV found {len(bands)} text bands")

        # Step 3 — Anchor each match to the nearest real text band
        anchored: List[KeywordMatch] = []
        for m in raw_matches:
            real_y = anchor_to_nearest_band(m.y_position, bands, search_radius=250)
            if real_y is None:
                self.logger.warning(
                    f"Could not anchor '{m.keyword}' (est y={m.y_position}) – using estimate"
                )
                real_y = m.y_position
            anchored.append(KeywordMatch(m.keyword, m.text, real_y))
            self.logger.debug(
                f"Anchored '{m.keyword}': est={m.y_position} → real={real_y}"
            )

        # Remove matches that landed on the same band
        anchored = self._deduplicate(anchored, proximity=40)
        anchored.sort(key=lambda m: m.y_position)

        self.logger.info(
            f"Final splits: {[(m.keyword, m.y_position) for m in anchored]}"
        )

        # Step 4 — Crop sections between consecutive keyword positions
        sections: List[ImageSection] = []
        for i, match in enumerate(anchored):
            y_start = match.y_position
            y_end   = anchored[i + 1].y_position if i + 1 < len(anchored) else height

            if (y_end - y_start) < min_section_height:
                self.logger.debug(f"Skipping tiny section at y={y_start}")
                continue

            sections.append(ImageSection(
                section_index   = len(sections),
                keyword_trigger = match.keyword,
                y_start         = y_start,
                y_end           = y_end,
                image           = pil_image.crop((0, y_start, width, y_end)),
            ))

        self.logger.info(f"Created {len(sections)} sections")
        return sections

    def save_sections(
        self,
        sections: List[ImageSection],
        output_dir: Union[str, Path],
        prefix: str = "section",
    ) -> List[Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for sec in sections:
            tag = sec.keyword_trigger or "header"
            fp  = out / f"{prefix}_{sec.section_index:02d}_{tag}.jpg"
            sec.image.save(fp, "JPEG", quality=95)
            saved.append(fp)

        self.logger.info(f"Saved {len(saved)} sections to {out}")
        return saved

    # ------------------------------------------------------------------
    # Gemini extraction
    # ------------------------------------------------------------------

    def _gemini_extract(
        self, image: Image.Image, image_height: int
    ) -> List[KeywordMatch]:
        prompt = self._build_prompt()
        try:
            response = self.model.generate_content([prompt, image])
            time.sleep(0.4)
            if not response or not response.text:
                self.logger.warning("Gemini returned empty response")
                return []
            return self._parse_response(response.text, image_height)
        except Exception as exc:
            self.logger.error(f"Gemini API error: {exc}")
            return []

    def _build_prompt(self) -> str:
        kw_list   = " | ".join(self.keywords)
        excl_list = " | ".join(self.excluded_keywords)
        return f"""
You are analyzing an Arabic exam page image.

TASK: Find EVERY occurrence of these section-header keywords (with or without diacritics):
  {kw_list}

Also detect them when prefixed with ال (e.g. السند, التعليمة).

DO NOT flag these excluded words:
  {excl_list}

For EACH keyword occurrence output one JSON object in an array.
Output valid JSON array ONLY — no markdown fences, no explanation.

Example output:
[
  {{
    "keyword": "سند",
    "text": "السند 1: شارك شادي في رحلة...",
    "position_hint": "top",
    "vertical_order": 1
  }},
  {{
    "keyword": "تعليمة",
    "text": "تعليمة 1: أحسب مبلغ الأب.",
    "position_hint": "upper",
    "vertical_order": 2
  }}
]

RULES:
- keyword        : "تعليمة" or "سند" only (no diacritics)
- text           : the exact line from the image containing the keyword
- position_hint  : top | upper | middle | lower | bottom
- vertical_order : 1 = topmost keyword found, 2 = next one down, etc.

List ALL keyword occurrences. A page may have 10+ keywords — include every one.
Output JSON array only.
"""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, image_height: int) -> List[KeywordMatch]:
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

    def _from_blocks(self, blocks: list, image_height: int) -> List[KeywordMatch]:
        total = max(len(blocks), 1)
        matches: List[KeywordMatch] = []

        for block in blocks:
            if not isinstance(block, dict):
                continue

            combined = block.get("text", "") + " " + block.get("keyword", "")
            confirmed_kw = self._detect_keyword(combined)
            if confirmed_kw is None:
                continue

            order = block.get("vertical_order", 1)
            hint  = block.get("position_hint", "middle")
            y     = self._estimate_y(order, total, hint, image_height)

            matches.append(KeywordMatch(confirmed_kw, block.get("text", ""), y))
            self.logger.debug(
                f"Gemini block: '{confirmed_kw}' order={order} hint={hint} est_y={y}"
            )

        return matches

    def _from_lines(self, raw: str, image_height: int) -> List[KeywordMatch]:
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        total = max(len(lines), 1)
        matches: List[KeywordMatch] = []

        for idx, line in enumerate(lines):
            kw = self._detect_keyword(line)
            if kw is None:
                continue
            y = int((idx / total) * image_height)
            matches.append(KeywordMatch(kw, line, y))

        return matches

    # ------------------------------------------------------------------
    # Keyword detection
    # ------------------------------------------------------------------

    def _detect_keyword(self, text: str) -> Optional[str]:
        norm = _normalize(text)
        for excl in self._excl_roots:
            if excl in norm:
                return None
        for kw, root in self._kw_roots.items():
            if root in norm:
                return kw
        return None

    # ------------------------------------------------------------------
    # Y estimation (coarse — refined by OpenCV anchoring)
    # ------------------------------------------------------------------

    def _estimate_y(
        self, order: int, total: int, hint: str, image_height: int
    ) -> int:
        z_min, z_max = self._ZONES.get(hint, (0.0, 1.0))
        z_centre     = (z_min + z_max) / 2
        order_frac   = (order - 1) / max(total - 1, 1)
        blended      = 0.6 * z_centre + 0.4 * order_frac
        clamped      = max(z_min, min(z_max, blended))
        return int(clamped * image_height)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(matches: List[KeywordMatch], proximity: int = 40) -> List[KeywordMatch]:
        if not matches:
            return []
        matches = sorted(matches, key=lambda m: m.y_position)
        out = [matches[0]]
        for m in matches[1:]:
            if abs(m.y_position - out[-1].y_position) > proximity:
                out.append(m)
        return out

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.exists():
                raise FileNotFoundError(f"Not found: {p}")
            return Image.open(p).convert("RGB")
        if isinstance(image, np.ndarray):
            arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) \
                  if len(image.shape) == 3 and image.shape[2] == 3 else image
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _to_cv(pil_image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)