"""
utils.py — Arabic handwriting OCR utilities for Google Cloud Vision.

All functions here are pure / stateless so they are easy to test
and reuse across the pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# ─────────────────────────────────────────────────────────────────────────────
# Arabic Unicode ranges used throughout this module
# ─────────────────────────────────────────────────────────────────────────────

# Core Arabic block (U+0600–U+06FF) covers letters, diacritics, digits, punctuation
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

# Arabic Presentation Forms A & B — sometimes emitted by OCR engines
_ARABIC_PRES_RE = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")

# Tatweel / kashida — the horizontal elongation glyph (U+0640)
_TATWEEL = "\u0640"

# ─────────────────────────────────────────────────────────────────────────────
# Compiled placeholder / noise patterns
# ─────────────────────────────────────────────────────────────────────────────

# Repeated filler characters that appear in exam templates
_PLACEHOLDER_PATTERNS: list[re.Pattern] = [
    re.compile(r"^[\s.\-_،,…/\\|*~]{3,}$"),        # generic filler run
    re.compile(r"^(\.{2,}|-{2,}|_{2,}|ـ{2,})$"),  # dots / dashes / tatweel
    re.compile(r"^(\s*\.{2,}\s*)+$"),
    re.compile(r"^(\s*-{2,}\s*)+$"),
    re.compile(r"^(\s*_{2,}\s*)+$"),
    re.compile(r"^(\s*ـ{2,}\s*)+$"),               # repeated kashida
    re.compile(r"^[\s\u200b\u200c\u200d\u00a0]+$"),# invisible chars only
    re.compile(r"^[○◯□▢✓✗×+\s]+$"),               # checkbox / symbol rows
]

# Whole-block illegibility tags (produced by LLM post-processors)
_ILLEGIBLE_TAG = re.compile(
    r"^\[(illegible|unclear|unreadable|not\s+clear|cannot\s+read"
    r"|unrecognizable|unknown|empty|blank)\]$",
    re.IGNORECASE,
)

# Printed question/instruction keywords — presence suggests a block is
# printed template text rather than handwritten student content.
# Keyed on the FIRST Arabic word or phrase of common question starters.
_PRINTED_QUESTION_STARTERS = re.compile(
    r"^(اختر|اربط|أكمل|اكمل|احسب|ارسم|اكتب|علل|فسر|اذكر"
    r"|تعليمة|تعليمات|السؤال|الجواب|ملاحظة|مثال"
    r"|choose|match|fill|calculate|draw|write|note|example"
    r")",
    re.IGNORECASE,
)

# Minimum fraction of Arabic characters for a block to be considered Arabic
_MIN_ARABIC_RATIO = 0.30

# Confidence below this → block is discarded
_MIN_BLOCK_CONFIDENCE = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# Data structure returned by the extractor for each OCR block
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OcrBlock:
    """One spatial unit of text from Google Vision (word / paragraph / block)."""
    text: str
    confidence: float                          # 0.0 – 1.0
    bounding_box: Optional[list] = None        # list of (x, y) vertex dicts
    is_handwritten: bool = False               # set by classifier
    source_level: str = "word"                 # "word" | "paragraph" | "block"


# ─────────────────────────────────────────────────────────────────────────────
# 1. clean_arabic_text
# ─────────────────────────────────────────────────────────────────────────────

def clean_arabic_text(text: str) -> str:
    """Lightly normalise Arabic OCR output without losing student content.

    What this does
    ──────────────
    • Removes tatweel / kashida (ـ) — purely decorative, never meaningful
    • Collapses runs of the same repeated punctuation to a single instance
    • Strips leading / trailing whitespace
    • Normalises multiple internal spaces to one
    • Converts Arabic-Indic digits (٠–٩) to Western digits (0–9)
    • Removes zero-width characters and non-breaking spaces

    What this does NOT do
    ──────────────────────
    • Does NOT fix spelling errors
    • Does NOT normalise letter shapes (e.g. ا vs أ vs إ) beyond tatweel
    • Does NOT re-order RTL text — callers must handle display direction
    • Does NOT remove diacritics (تشكيل) — they carry meaning for children
    """
    if not text:
        return ""

    # Remove tatweel
    text = text.replace(_TATWEEL, "")

    # Remove zero-width and non-breaking space characters
    text = re.sub(r"[\u200b\u200c\u200d\u00a0\uFEFF]", "", text)

    # Convert Arabic-Indic digits → Western digits
    _AR_DIGIT = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    text = text.translate(_AR_DIGIT)

    # Collapse runs of the same punctuation character (> 2 in a row → 1)
    text = re.sub(r"([^\w\u0600-\u06FF])\1{2,}", r"\1", text)

    # Normalise internal whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. remove_placeholders
# ─────────────────────────────────────────────────────────────────────────────

def remove_placeholders(text: str) -> str:
    """Return *text* with placeholder tokens removed, or ``""`` if entirely filler.

    A placeholder is a repeated filler character used in printed exam
    templates to indicate where the student should write:
        ............    ------    ______    ـــــ

    The function strips these from the extracted text so they are not
    mistaken for student answers.
    """
    if not text:
        return ""

    # Split on whitespace, filter out placeholder-only tokens, rejoin
    tokens = text.split()
    clean_tokens = [t for t in tokens if not _is_filler_token(t)]
    result = " ".join(clean_tokens)
    return result.strip()


def _is_filler_token(token: str) -> bool:
    """True when *token* is a pure placeholder with no real content."""
    if not token:
        return True
    if _ILLEGIBLE_TAG.match(token):
        return True
    for pat in _PLACEHOLDER_PATTERNS:
        if pat.fullmatch(token):
            return True
    # Token has no Arabic character, digit, or Latin letter → filler
    if not re.search(r"[A-Za-z0-9\u0600-\u06FF]", token):
        return True
    return False


def is_placeholder_only(text: str) -> bool:
    """True when *text* as a whole is nothing but filler content."""
    if not text or not text.strip():
        return True
    cleaned = remove_placeholders(text)
    return not bool(cleaned)


# ─────────────────────────────────────────────────────────────────────────────
# 3. filter_by_confidence
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_confidence(
    blocks: Sequence[OcrBlock],
    min_confidence: float = _MIN_BLOCK_CONFIDENCE,
) -> List[OcrBlock]:
    """Return only blocks whose confidence meets *min_confidence*.

    Google Vision does not always emit per-word confidence for
    DOCUMENT_TEXT_DETECTION; when confidence is missing (0.0) the block
    passes through so as not to silently discard valid text.
    """
    result = []
    for block in blocks:
        # 0.0 means "not reported" in Vision API — pass through
        if block.confidence == 0.0 or block.confidence >= min_confidence:
            result.append(block)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. is_handwritten_candidate
# ─────────────────────────────────────────────────────────────────────────────

def is_handwritten_candidate(text: str, confidence: float = 1.0) -> bool:
    """Heuristic classifier: is *text* likely to be handwritten student content?

    Returns False (→ skip block) when ANY of these hold:
    ─────────────────────────────────────────────────────
    1. Text is a placeholder / filler only.
    2. Text starts with a printed question keyword (instruction starter).
    3. Text has < 30% Arabic characters and no digits — likely printed
       metadata (page numbers, headers, section labels).
    4. Confidence is reported and is below the minimum threshold.
       (0.0 = not reported → treat as passing)
    5. Text is all-uppercase Latin — likely a printed label.
    6. Text is a single repeated character run (printed separator).

    Returns True (→ keep block) otherwise.

    Note: this is a heuristic, not a guaranteed classifier.
    The extractor combines this with spatial / confidence signals.
    """
    if not text or not text.strip():
        return False

    t = text.strip()

    # 1. Pure filler
    if is_placeholder_only(t):
        return False

    # 2. Starts with a printed question keyword
    if _PRINTED_QUESTION_STARTERS.match(t):
        return False

    # 3. Very low Arabic ratio with no digits
    arabic_chars  = len(_ARABIC_RE.findall(t))
    total_chars   = len(re.sub(r"\s", "", t))
    digit_count   = len(re.findall(r"\d", t))
    arabic_ratio  = arabic_chars / total_chars if total_chars > 0 else 0.0

    if arabic_ratio < _MIN_ARABIC_RATIO and digit_count == 0:
        return False

    # 4. Below minimum confidence (skip check when confidence not reported)
    if confidence > 0.0 and confidence < _MIN_BLOCK_CONFIDENCE:
        return False

    # 5. All-uppercase Latin (printed label / header)
    latin_only = re.sub(r"[^A-Za-z]", "", t)
    if latin_only and latin_only == latin_only.upper() and len(latin_only) > 3:
        return False

    # 6. Single character repeated ≥ 4 times (separator / filler)
    if re.fullmatch(r"(.)\1{3,}", t):
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bounding-box helpers
# ─────────────────────────────────────────────────────────────────────────────

def bbox_top(vertices: list) -> int:
    """Return the top Y coordinate of a bounding-box vertex list."""
    ys = [v.get("y", 0) for v in vertices if "y" in v]
    return min(ys) if ys else 0


def bbox_bottom(vertices: list) -> int:
    """Return the bottom Y coordinate of a bounding-box vertex list."""
    ys = [v.get("y", 0) for v in vertices if "y" in v]
    return max(ys) if ys else 0


def bbox_left(vertices: list) -> int:
    """Return the left X coordinate of a bounding-box vertex list."""
    xs = [v.get("x", 0) for v in vertices if "x" in v]
    return min(xs) if xs else 0


def bbox_right(vertices: list) -> int:
    """Return the right X coordinate of a bounding-box vertex list."""
    xs = [v.get("x", 0) for v in vertices if "x" in v]
    return max(xs) if xs else 0


def bbox_height(vertices: list) -> int:
    return bbox_bottom(vertices) - bbox_top(vertices)


def bbox_width(vertices: list) -> int:
    return bbox_right(vertices) - bbox_left(vertices)


def blocks_in_zone(
    blocks: Sequence[OcrBlock],
    zone_top: int,
    zone_bottom: int,
    image_height: int = 0,
) -> List[OcrBlock]:
    """Return blocks whose bounding box falls (at least partially) within
    the vertical zone [zone_top, zone_bottom].

    If zone coordinates are given as fractions (0.0–1.0), they are
    scaled by *image_height* automatically.
    """
    if image_height > 0 and zone_top <= 1.0 and zone_bottom <= 1.0:
        zone_top    = int(zone_top    * image_height)
        zone_bottom = int(zone_bottom * image_height)

    result = []
    for block in blocks:
        if block.bounding_box is None:
            result.append(block)   # no spatial info — include by default
            continue
        top    = bbox_top(block.bounding_box)
        bottom = bbox_bottom(block.bounding_box)
        # Overlap: block overlaps the zone if top < zone_bottom AND bottom > zone_top
        if top < zone_bottom and bottom > zone_top:
            result.append(block)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. RTL reassembly
# ─────────────────────────────────────────────────────────────────────────────

def sort_blocks_rtl(blocks: Sequence[OcrBlock]) -> List[OcrBlock]:
    """Sort blocks in reading order for Arabic (RTL, top-to-bottom).

    Grouping strategy:
    1. Sort blocks by vertical position (top Y ascending).
    2. Within each horizontal band (same row), sort right-to-left by X.

    A "row" is defined as blocks whose top-Y values are within
    LINE_TOLERANCE pixels of each other.
    """
    LINE_TOLERANCE = 15   # pixels

    if not blocks:
        return []

    # Separate blocks with and without spatial info
    spatial  = [b for b in blocks if b.bounding_box]
    no_space = [b for b in blocks if not b.bounding_box]

    # Sort by top-Y first
    spatial.sort(key=lambda b: bbox_top(b.bounding_box))

    # Group into rows
    rows: list[list[OcrBlock]] = []
    current_row: list[OcrBlock] = []
    current_top: int = -9999

    for block in spatial:
        top = bbox_top(block.bounding_box)
        if abs(top - current_top) <= LINE_TOLERANCE:
            current_row.append(block)
        else:
            if current_row:
                rows.append(current_row)
            current_row = [block]
            current_top = top

    if current_row:
        rows.append(current_row)

    # Within each row: sort right-to-left (descending X)
    for row in rows:
        row.sort(key=lambda b: bbox_right(b.bounding_box), reverse=True)

    ordered = [block for row in rows for block in row]
    return ordered + no_space


def assemble_text_rtl(blocks: Sequence[OcrBlock]) -> str:
    """Join block texts in RTL reading order into a single string.

    Lines are separated by newlines; words within a line by spaces.
    """
    sorted_blocks = sort_blocks_rtl(blocks)
    if not sorted_blocks:
        return ""

    LINE_TOLERANCE = 15
    lines: list[list[str]] = []
    current_line: list[str] = []
    current_top: int = -9999

    for block in sorted_blocks:
        top = bbox_top(block.bounding_box) if block.bounding_box else current_top
        if abs(top - current_top) <= LINE_TOLERANCE or current_top == -9999:
            current_line.append(block.text)
            current_top = top
        else:
            if current_line:
                lines.append(current_line)
            current_line = [block.text]
            current_top = top

    if current_line:
        lines.append(current_line)

    return "\n".join(" ".join(line) for line in lines)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Confidence helpers (shared with pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def confidence_label(score: float) -> str:
    if score >= 0.90: return "HIGH"
    if score >= 0.70: return "MEDIUM"
    if score >= 0.50: return "LOW"
    return "VERY_LOW"


def confidence_emoji(score: float) -> str:
    if score >= 0.90: return "🟢"
    if score >= 0.70: return "🟡"
    if score >= 0.50: return "🟠"
    return "🔴"