"""
openai_image_splitter.py
------------------------
Splits exam pages into sections using OpenAI Vision API + OpenCV.

Workflow:
1. Tall image is sliced into overlapping vertical chunks
2. OpenAI Vision processes each chunk independently to find keywords + y_percent
3. Per-chunk y_percent values are converted to absolute pixel positions
4. All matches are merged, deduplicated, and snapped to OpenCV text bands
5. Image is split at exact pixel boundaries between consecutive keywords
"""

from __future__ import annotations

import re
import base64
import json
import sys
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union
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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KeywordMatch:
    keyword: str      # original keyword e.g. "تعليمة"
    text: str         # full line text containing it
    y_position: int   # absolute pixel Y in the full image


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

    # Chunk settings — tune if needed
    CHUNK_HEIGHT  = 1500   # px per chunk sent to OpenAI
    CHUNK_OVERLAP = 200    # px overlap between consecutive chunks

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

        # Step 1 — slice into chunks, run OpenAI on each
        raw_matches = self._extract_all_chunks(pil_image, height)

        if not raw_matches:
            self.logger.warning("No keywords found – returning full image")
            return [ImageSection(0, None, 0, height, pil_image)]

        # Step 2 — OpenCV bands
        bands = find_text_bands(cv_image)
        self.logger.debug(f"OpenCV found {len(bands)} text bands")

        # Step 3 — Snap to nearest band
        anchored: List[KeywordMatch] = []
        for m in raw_matches:
            real_y = anchor_to_nearest_band(m.y_position, bands, search_radius=300)
            if real_y is None:
                self.logger.warning(
                    f"Could not anchor '{m.keyword}' (est y={m.y_position}) – using estimate"
                )
                real_y = m.y_position
            anchored.append(KeywordMatch(m.keyword, m.text, real_y))
            self.logger.debug(
                f"Anchored '{m.keyword}': est={m.y_position} → real={real_y}"
            )

        # Step 4 — Deduplicate and sort
        anchored = self._deduplicate(anchored, proximity=60)
        anchored.sort(key=lambda m: m.y_position)

        self.logger.info(
            f"Final splits ({len(anchored)}): "
            f"{[(m.keyword, m.y_position) for m in anchored]}"
        )

        # Step 5 — Crop sections
        sections: List[ImageSection] = []
        for i, match in enumerate(anchored):
            y_start = match.y_position
            y_end   = anchored[i + 1].y_position if i + 1 < len(anchored) else height

            if (y_end - y_start) < min_section_height:
                self.logger.debug(
                    f"Skipping tiny section '{match.keyword}' at y={y_start} "
                    f"(height={y_end - y_start}px)"
                )
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
    # Chunked extraction
    # ------------------------------------------------------------------

    def _extract_all_chunks(
        self, pil_image: Image.Image, total_height: int
    ) -> List[KeywordMatch]:
        """Slice the image into overlapping chunks, run OpenAI on each,
        convert chunk-local y_percent → absolute pixel Y."""

        width = pil_image.width
        all_matches: List[KeywordMatch] = []
        chunk_idx = 0
        y_top = 0

        while y_top < total_height:
            y_bot = min(y_top + self.CHUNK_HEIGHT, total_height)
            chunk = pil_image.crop((0, y_top, width, y_bot))
            chunk_h = y_bot - y_top

            self.logger.info(
                f"Chunk {chunk_idx}: y={y_top}-{y_bot} ({chunk_h}px)"
            )

            matches = self._openai_extract(chunk, chunk_h)

            for m in matches:
                abs_y = y_top + m.y_position
                all_matches.append(KeywordMatch(m.keyword, m.text, abs_y))
                self.logger.debug(
                    f"  Chunk {chunk_idx}: '{m.keyword}' local_y={m.y_position} "
                    f"→ abs_y={abs_y}  text='{m.text[:60]}'"
                )

            if y_bot >= total_height:
                break
            y_top = y_bot - self.CHUNK_OVERLAP
            chunk_idx += 1

        self.logger.info(
            f"Total raw matches from {chunk_idx + 1} chunk(s): {len(all_matches)}"
        )
        return all_matches

    # ------------------------------------------------------------------
    # OpenAI Vision extraction (single chunk)
    # ------------------------------------------------------------------

    def _openai_extract(
        self, image: Image.Image, image_height: int
    ) -> List[KeywordMatch]:

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
- keyword   : exactly "تعليمة" or "سند" (no diacritics, no ال)
- text      : exact line as seen in the image
- y_percent : 0.0–1.0 relative to THIS image only
- vertical_order : 1 = topmost in this image

Your entire response must be only the JSON array, nothing else.
"""

    # ------------------------------------------------------------------
    # Parsing
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
        matches: List[KeywordMatch] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            combined = block.get("text", "") + " " + block.get("keyword", "")
            confirmed_kw = self._detect_keyword(combined)
            if confirmed_kw is None:
                continue
            y_pct = max(0.0, min(1.0, float(block.get("y_percent", 0.5))))
            estimated_y = int(y_pct * image_height)
            matches.append(KeywordMatch(confirmed_kw, block.get("text", ""), estimated_y))
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
    # Keyword detection — whole-word exclusion
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
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(matches: List[KeywordMatch], proximity: int = 60) -> List[KeywordMatch]:
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

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")