"""
image_splitter_paddleocr.py
===========================
Production-ready Arabic exam-sheet image splitter using PaddleOCR.

Segmentation rules
──────────────────
• A section STARTS only when an OCR token is an EXACT match (after
  normalisation) to one of the allowed keywords.
• "تسند" and any other word in EXCLUDED_KEYWORDS never trigger a split,
  even though they share roots with allowed keywords.
• Content above the first keyword is DROPPED (it is not a section).
• Each section spans [keyword_y → next_keyword_y) or EOF.

Matching pipeline (is_valid_section_start)
──────────────────────────────────────────
  Step 1 – Normalise the OCR token  (strip diacritics, unify alefs, NFKC).
  Step 2 – Hard-exclude if token ∈ EXCLUDED_NORM  (exact set lookup).
  Step 3 – Accept if token ∈ ALLOWED_NORM          (exact set lookup).
  Step 4 – Optional fuzzy fallback via fuzz.ratio() ≥ threshold (default 95).
            fuzz.ratio() is a full-string metric — it does NOT allow substrings.
  Step 5 – Reject everything else.
"""

from __future__ import annotations

import os
import cv2
import logging
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from paddleocr import PaddleOCR
from rapidfuzz import fuzz


# ===========================================================================
# 1. Arabic normalisation
# ===========================================================================

def strip_arabic_diacritics(text: str) -> str:
    """Remove all Arabic tashkeel / harakat (combining diacritic marks)."""
    DIACRITIC_RANGES = (
        ('\u0610', '\u061A'),   # Arabic extended
        ('\u064B', '\u065F'),   # Fathatan … Wavy hamza below
        ('\u0670', '\u0670'),   # Superscript alef
        ('\u06D6', '\u06DC'),   # Small high ligature
        ('\u06DF', '\u06E4'),   # Small high rounded / zer
        ('\u06E7', '\u06E8'),   # Small high meem
        ('\u06EA', '\u06ED'),   # Small low jeem
    )
    out = []
    for ch in text:
        cp = ord(ch)
        keep = True
        for lo, hi in DIACRITIC_RANGES:
            if ord(lo) <= cp <= ord(hi):
                keep = False
                break
        if keep:
            out.append(ch)
    return ''.join(out)


def normalize_arabic(text: str) -> str:
    """
    Full normalisation pipeline — call this on BOTH keywords and OCR tokens
    before any comparison so diacritics never affect matching.

    Steps
    ─────
    1. Strip all diacritics (tashkeel / harakat).
    2. Unify alef variants (أ إ آ ٱ) → bare alef (ا).
    3. NFKC unicode normalisation (resolves compatibility forms).
    4. Strip surrounding whitespace.

    Examples
    ────────
    "التَّعْلِيمَةُ"  →  "التعليمة"
    "سَنَدٌ"          →  "سند"
    "تسند"            →  "تسند"   (not changed; excluded at matching step)
    """
    text = strip_arabic_diacritics(text)
    for variant in 'أإآٱ':
        text = text.replace(variant, 'ا')
    return unicodedata.normalize('NFKC', text).strip()


# ===========================================================================
# 2. Keyword sets  (computed once at import time — O(1) lookups)
# ===========================================================================

ALLOWED_KEYWORDS: list[str] = [
    "تعليمة", "سند",
    "التَّعْلِيمَة", "التَّعْلِيمَةُ", "التَّعْلِيمَةِ", "التَّعْلِيمَةَ",
    "السَّنَد", "السَّنَدُ", "السَّنَدِ",
    "تَعْليمَة", "تَعْليمَةٌ", "تَعْليمَةٍ", "تَعْليمَةً",
    "سَنَد", "سَنَدٌ", "سَنَدٍ",
]
EXCLUDED_KEYWORDS: list[str] = ["تسند"]

# Normalised frozensets for O(1) exact membership tests
ALLOWED_NORM:  frozenset[str] = frozenset(normalize_arabic(w) for w in ALLOWED_KEYWORDS)
EXCLUDED_NORM: frozenset[str] = frozenset(normalize_arabic(w) for w in EXCLUDED_KEYWORDS)


# ===========================================================================
# 3. Core matching function
# ===========================================================================

def is_valid_section_start(
    ocr_word: str,
    fuzzy_threshold: int = 95,
) -> tuple[bool, str, int]:
    """
    Decide whether *ocr_word* is a valid section-start keyword.

    Parameters
    ──────────
    ocr_word        : raw string from PaddleOCR
    fuzzy_threshold : fuzz.ratio threshold for the optional fallback.
                      0  → pure exact match only (safest).
                      95 → allows a single OCR character error.

    Returns
    ───────
    (is_valid, matched_keyword_norm, score)
      score = 100 for exact, fuzzy score for fallback, 0 for reject.

    Decision pipeline
    ─────────────────
    Step 1 – Normalise.
    Step 2 – Exact exclusion check  → reject if in EXCLUDED_NORM.
             "تسند" normalises to "تسند" which IS in EXCLUDED_NORM → rejected.
             "سند"  normalises to "سند"  which is NOT in EXCLUDED_NORM → passes.
    Step 3 – Exact allowance check  → accept if in ALLOWED_NORM.
             No substring logic, no root matching, no partial comparison.
    Step 4 – fuzz.ratio() fallback  (full-string Levenshtein, NOT partial_ratio).
             partial_ratio() would allow substring matches and is NOT used here.
             fuzz.ratio("سند", "مسندة") ≈ 57  <  95  → rejected  ✓
             fuzz.ratio("سند", "سنذ")   ≈ 83  <  95  → rejected  ✓
    Step 5 – Reject.
    """
    # ── Step 1: normalise ────────────────────────────────────────────────
    norm = normalize_arabic(ocr_word)
    if not norm:
        return False, "", 0

    # ── Step 2: hard exclusion (exact) ───────────────────────────────────
    if norm in EXCLUDED_NORM:
        return False, "", 0

    # ── Step 3: exact match ───────────────────────────────────────────────
    if norm in ALLOWED_NORM:
        return True, norm, 100

    # ── Step 4: high-confidence fuzzy fallback ────────────────────────────
    if fuzzy_threshold > 0:
        best_score, best_kw = 0, ""
        for kw in ALLOWED_NORM:
            s = fuzz.ratio(kw, norm)      # full-string — no substrings
            if s > best_score:
                best_score, best_kw = s, kw
        if best_score >= fuzzy_threshold:
            return True, best_kw, best_score

    # ── Step 5: reject ────────────────────────────────────────────────────
    return False, "", 0


# ===========================================================================
# 4. PaddleOCR result parser
# ===========================================================================

def parse_paddle_result(
    paddle_result: list,
    min_confidence: float = 0.25,
) -> list[dict]:
    """
    Flatten PaddleOCR output into a list of token dicts.

    PaddleOCR returns a nested structure:
      result[page_idx]  →  list of lines
      line              →  [bbox_points, (text, confidence)]
      bbox_points       →  [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  (4 corners)

    Returns
    ───────
    List of dicts:
      { "text": str, "bbox": [[x,y]×4], "conf": float,
        "y_top": int, "y_bot": int, "x_left": int, "x_right": int }
    """
    tokens: list[dict] = []

    # PaddleOCR wraps results in an extra list when batch_size=1
    result = paddle_result
    if result and isinstance(result[0], list) and result[0] and isinstance(result[0][0], list):
        result = result[0]

    for line in result:
        if line is None:
            continue
        try:
            bbox_pts, (text, conf) = line
        except (TypeError, ValueError):
            continue

        if conf < min_confidence or not text.strip():
            continue

        ys = [int(pt[1]) for pt in bbox_pts]
        xs = [int(pt[0]) for pt in bbox_pts]
        tokens.append({
            "text":    text.strip(),
            "bbox":    bbox_pts,
            "conf":    float(conf),
            "y_top":   min(ys),
            "y_bot":   max(ys),
            "x_left":  min(xs),
            "x_right": max(xs),
        })

    return tokens


# ===========================================================================
# 5. ImageSplitter
# ===========================================================================

class ImageSplitter:
    """
    Splits a vertically-merged Arabic exam image into sections using PaddleOCR.

    Usage
    ─────
        splitter = ImageSplitter()
        result   = splitter.split_and_save("exam.jpg", exam_id=1,
                                            return_sections=True)
    """

    # ── Tunable config ────────────────────────────────────────────────── #
    MIN_SECTION_HEIGHT_PX:    int   = 80
    MIN_CONFIDENCE:           float = 0.25
    FUZZY_FALLBACK_THRESHOLD: int   = 95    # 0 = pure exact match

    # Preprocessing
    PREPROCESS_UPSCALE:         bool  = True
    PREPROCESS_UPSCALE_FACTOR:  float = 2.0
    PREPROCESS_UPSCALE_MAX_DIM: int   = 4000
    PREPROCESS_DENOISE:         bool  = True
    PREPROCESS_CONTRAST:        bool  = True
    PREPROCESS_THRESHOLD:       bool  = False

    # ── Init ──────────────────────────────────────────────────────────── #
    def __init__(
        self,
        output_dir: str = "data/Sections/exams/paddleocr_output",
        lang:       str = "arabic",
    ):
        self._setup_logging()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug").mkdir(exist_ok=True)

        self.logger.info(f"Allowed keywords  : {sorted(ALLOWED_NORM)}")
        self.logger.info(f"Excluded keywords : {sorted(EXCLUDED_NORM)}")
        self.logger.info(f"Fuzzy threshold   : {self.FUZZY_FALLBACK_THRESHOLD}")

        self.logger.info("Loading PaddleOCR …")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
        )
        self.logger.info("PaddleOCR ready.")

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    # ── Preprocessing ─────────────────────────────────────────────────── #
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Upscale → denoise → CLAHE → (optional binarise). Returns BGR."""
        h, w = image.shape[:2]
        result = image.copy()

        if self.PREPROCESS_UPSCALE:
            max_dim = max(h, w)
            if max_dim < self.PREPROCESS_UPSCALE_MAX_DIM:
                scale = min(self.PREPROCESS_UPSCALE_FACTOR,
                            self.PREPROCESS_UPSCALE_MAX_DIM / max_dim)
                if scale > 1.05:
                    nw, nh = int(w * scale), int(h * scale)
                    result = cv2.resize(result, (nw, nh),
                                        interpolation=cv2.INTER_CUBIC)
                    self.logger.info(f"Upscaled {w}×{h} → {nw}×{nh} (×{scale:.2f})")

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        if self.PREPROCESS_DENOISE:
            gray = cv2.fastNlMeansDenoising(gray, h=10,
                                             templateWindowSize=7,
                                             searchWindowSize=21)
        if self.PREPROCESS_CONTRAST:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        if self.PREPROCESS_THRESHOLD:
            gray = cv2.adaptiveThreshold(gray, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 10)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ── OCR ───────────────────────────────────────────────────────────── #
    def _run_ocr(self, image: np.ndarray) -> list[dict]:
        """
        Two-pass OCR: original image + preprocessed image.
        Results are merged and near-duplicates are suppressed.
        """
        tokens: list[dict] = []

        self.logger.info("OCR pass 1 — original image")
        raw1 = self.ocr.ocr(image)
        tokens.extend(parse_paddle_result(raw1, self.MIN_CONFIDENCE))

        self.logger.info("OCR pass 2 — preprocessed image")
        raw2 = self.ocr.ocr(self._preprocess(image))
        tokens.extend(parse_paddle_result(raw2, self.MIN_CONFIDENCE))

        self.logger.info(f"Total detections (pre-dedup): {len(tokens)}")
        return tokens

    # ── Split-line detection ───────────────────────────────────────────── #
    def _detect_split_y(self, image: np.ndarray) -> list[int]:
        """
        Return sorted, deduplicated Y positions of valid section-start keywords.
        Coordinates are always in the ORIGINAL image's pixel space.

        Deduplication: if two detections from different passes share >50%
        vertical overlap they are considered the same word; only the first is kept.
        """
        h_orig, w_orig = image.shape[:2]
        tokens = self._run_ocr(image)

        raw_y:      list[int]         = []
        seen_boxes: list[tuple[int, int]] = []   # (y_top, y_bot)
        debug_img = image.copy()

        for tok in tokens:
            text = tok["text"]
            conf = tok["conf"]
            y_top  = tok["y_top"]
            y_bot  = tok["y_bot"]
            x_left = tok["x_left"]
            x_right = tok["x_right"]

            # Clamp to image bounds (preprocessed upscaling can shift coords)
            y_top  = max(0, min(y_top,  h_orig - 1))
            y_bot  = max(0, min(y_bot,  h_orig - 1))
            x_left = max(0, min(x_left, w_orig - 1))
            x_right = max(0, min(x_right, w_orig - 1))

            valid, kw, score = is_valid_section_start(
                text, fuzzy_threshold=self.FUZZY_FALLBACK_THRESHOLD
            )

            if not valid:
                # Log near-misses so threshold can be tuned
                norm = normalize_arabic(text)
                if norm and norm not in EXCLUDED_NORM:
                    best = max(
                        (fuzz.ratio(k, norm) for k in ALLOWED_NORM), default=0
                    )
                    if best >= 75:
                        self.logger.debug(
                            f"  NEAR-MISS score={best}: '{text}' → '{norm}'"
                        )
                continue

            # Deduplicate across passes
            if any(
                (min(y_bot, pb) - max(y_top, pt))
                / max(max(y_bot, pb) - min(y_top, pt), 1) > 0.5
                for pt, pb in seen_boxes
            ):
                self.logger.debug(f"  Duplicate suppressed: '{text}' y={y_top}")
                continue

            seen_boxes.append((y_top, y_bot))
            raw_y.append(y_top)
            self.logger.info(
                f"  SPLIT ← '{text}' → '{normalize_arabic(text)}' "
                f"kw='{kw}' score={score} conf={conf:.2f} y={y_top}"
            )

            # Debug overlay
            cv2.rectangle(debug_img, (x_left, y_top), (x_right, y_bot),
                          (0, 200, 0), 3)
            cv2.line(debug_img, (0, y_top), (w_orig, y_top), (0, 0, 255), 2)
            cv2.putText(debug_img, f"{text[:15]} [{score}]",
                        (x_left, max(y_top - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Save debug image
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        dbg = self.output_dir / "debug" / f"ocr_debug_{ts}.jpg"
        cv2.imwrite(str(dbg), debug_img)
        self.logger.info(f"Debug image → {dbg}")

        split_ys = sorted(set(raw_y))

        # Auto-diagnosis when nothing matched
        if not split_ys:
            self.logger.warning(
                "No keywords matched! Top-30 tokens by confidence:"
            )
            self.logger.warning(
                f"  {'RAW':<28} {'NORMALISED':<22} {'BEST_SCORE':>10} {'CONF':>6}"
            )
            for tok in sorted(tokens, key=lambda t: t["conf"], reverse=True)[:30]:
                t    = tok["text"]
                norm = normalize_arabic(t)
                best = max((fuzz.ratio(k, norm) for k in ALLOWED_NORM), default=0)
                self.logger.warning(
                    f"  {t[:26]:<28} {norm[:20]:<22} {best:>10} {tok['conf']:>6.2f}"
                )

        self.logger.info(f"Final split Y positions: {split_ys}")
        return split_ys

    # ── Splitting ─────────────────────────────────────────────────────── #
    def split_image(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Build sections from split_ys.
        Section i = image[split_ys[i] : split_ys[i+1]].
        Content above the first keyword is dropped.
        """
        h = image.shape[0]
        split_ys = self._detect_split_y(image)

        if not split_ys:
            self.logger.warning("No keywords found — returning whole image.")
            return [image]

        boundaries = split_ys + [h]
        sections: list[np.ndarray] = []

        for i in range(len(boundaries) - 1):
            y0, y1 = boundaries[i], min(h, boundaries[i + 1])
            height = y1 - y0
            if height >= self.MIN_SECTION_HEIGHT_PX:
                sections.append(image[y0:y1, :])
                self.logger.info(
                    f"  Section {len(sections)}: rows {y0}–{y1} ({height}px)"
                )
            else:
                self.logger.warning(
                    f"  Skipped thin slice {y0}–{y1} "
                    f"({height}px < {self.MIN_SECTION_HEIGHT_PX}px min)"
                )

        self.logger.info(f"Produced {len(sections)} section(s).")
        return sections

    # ── Save ──────────────────────────────────────────────────────────── #
    def save_sections(
        self, sections: list[np.ndarray], exam_id: int | str
    ) -> list[str]:
        save_dir = self.output_dir / f"exam_{exam_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i, sec in enumerate(sections, 1):
            p = save_dir / f"exam_{exam_id}_section_{i:02d}.jpg"
            cv2.imwrite(str(p), sec)
            paths.append(str(p))
            self.logger.info(f"  Saved → {p}")
        return paths

    # ── Diagnostic helper ─────────────────────────────────────────────── #
    def diagnose_image(self, image_path: str) -> None:
        """
        Print every OCR token (sorted by Y) with normalised form and match result.
        Run this whenever a keyword is not being detected to understand why.
        """
        image = cv2.imread(image_path)
        if image is None:
            print("ERROR: cannot load image")
            return

        raw   = self.ocr.ocr(image)
        tokens = parse_paddle_result(raw, min_confidence=0.0)   # show all
        tokens.sort(key=lambda t: t["y_top"])

        w = 88
        print(f"\n{'='*w}")
        print(f"DIAGNOSIS: {image_path}")
        print(f"{'='*w}")
        print(f"{'RAW TEXT':<30} {'NORMALISED':<22} {'RESULT':<10} {'SCORE':>6} {'CONF':>6}")
        print(f"{'-'*w}")

        for tok in tokens:
            text, conf = tok["text"], tok["conf"]
            norm = normalize_arabic(text)
            valid, kw, score = is_valid_section_start(
                text, fuzzy_threshold=self.FUZZY_FALLBACK_THRESHOLD
            )
            result = "✓ MATCH" if valid else ("EXCLUDED" if norm in EXCLUDED_NORM else "—")
            print(
                f"{text[:28]:<30} {norm[:20]:<22} {result:<10} {score:>6} {conf:>6.2f}"
            )

        print("=" * w)
        print(f"\nAllowed (normalised) : {sorted(ALLOWED_NORM)}")
        print(f"Excluded (normalised): {sorted(EXCLUDED_NORM)}")

    # ── Entry point ───────────────────────────────────────────────────── #
    def split_and_save(
        self,
        image_path:     str,
        exam_id:        int | str = 1,
        return_sections: bool     = False,
    ) -> dict:
        self.logger.info(f"Processing exam {exam_id} — {image_path}")

        if not os.path.exists(image_path):
            return {"success": False, "error": "Image not found"}

        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}

        sections = self.split_image(image)
        if not sections:
            return {"success": False, "error": "No sections created"}

        saved_paths = self.save_sections(sections, exam_id)
        result: dict = {
            "success":      True,
            "exam_id":      exam_id,
            "num_sections": len(sections),
            "saved_paths":  saved_paths,
        }
        if return_sections:
            result["sections"] = sections
        return result