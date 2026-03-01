import os
import cv2
import logging
import unicodedata
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import easyocr
from rapidfuzz import fuzz, process as rfprocess


# ---------------------------------------------------------------------------
# Arabic normalisation helpers
# ---------------------------------------------------------------------------

def strip_arabic_diacritics(text: str) -> str:
    """Remove all Arabic diacritics (harakat / tashkeel) from a string."""
    # Unicode ranges for Arabic combining marks
    diacritic_ranges = (
        ('\u0610', '\u061A'),  # Arabic extended
        ('\u064B', '\u065F'),  # Fathatan … Wavy hamza below
        ('\u0670', '\u0670'),  # Superscript alef
        ('\u06D6', '\u06DC'),  # Small high ligature
        ('\u06DF', '\u06E4'),  # Small high rounded/zer
        ('\u06E7', '\u06E8'),  # Small high meem
        ('\u06EA', '\u06ED'),  # Small low jeem
    )
    result = []
    for ch in text:
        cp = ord(ch)
        keep = True
        for lo, hi in diacritic_ranges:
            if ord(lo) <= cp <= ord(hi):
                keep = False
                break
        if keep:
            result.append(ch)
    return ''.join(result)


def normalize_arabic(text: str) -> str:
    """
    Full normalisation pipeline:
      1. Strip diacritics
      2. Normalise hamzas / alefs to bare alef (ا)
      3. Normalise teh marbuta (ة → ه)  — optional, disabled here
      4. NFKC normalisation
      5. Strip whitespace
    """
    text = strip_arabic_diacritics(text)
    # Unify alef variants → bare alef
    alef_variants = 'أإآٱ'
    for ch in alef_variants:
        text = text.replace(ch, 'ا')
    text = unicodedata.normalize('NFKC', text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# ImageSplitter v2
# ---------------------------------------------------------------------------

class ImageSplitter:
    """
    Splits a vertically-merged exam image into sections by detecting
    Arabic section-header keywords (e.g. "تعليمة", "سند") via EasyOCR
    and fuzzy matching, with extensive preprocessing and debug support.
    """

    # ------------------------------------------------------------------ #
    # Config
    # ------------------------------------------------------------------ #

    # Raw keywords — diacritized forms are kept so we can normalise them
    KEY_WORDS_RAW: list[str] = [
        "تعليمة", "سند",
        "التَّعْلِيمَة", "التَّعْلِيمَةُ", "التَّعْلِيمَةِ", "التَّعْلِيمَةَ",
        "السَّنَد", "السَّنَدُ", "السَّنَدِ",
        "تَعْليمَة", "سَنَد", "تَعْليمَةٌ", "تَعْليمَةٍ", "تَعْليمَةً",
        "سَنَدٌ", "سَنَدٍ",
    ]

    EXCLUDED_KEYWORDS_RAW: list[str] = ["تسند"]

    # After normalisation these are the base forms we actually match against
    # (computed in __init__)
    KEY_WORDS_NORM: list[str] = []
    EXCLUDED_KEYWORDS_NORM: list[str] = []

    # Matching
    SIMILARITY_THRESHOLD: int = 65          # lowered: OCR drops diacritics
    EXCLUSION_THRESHOLD: int = 85           # stricter: don't over-exclude
    MIN_SECTION_HEIGHT_PX: int = 80
    MIN_CONFIDENCE: float = 0.25            # lowered from 0.3

    # Preprocessing toggles
    PREPROCESS_UPSCALE: bool = True         # upscale if image is small
    PREPROCESS_UPSCALE_FACTOR: float = 2.0
    PREPROCESS_UPSCALE_MAX_DIM: int = 4000  # don't upscale huge images
    PREPROCESS_DENOISE: bool = True
    PREPROCESS_CONTRAST: bool = True        # CLAHE contrast enhancement
    PREPROCESS_THRESHOLD: bool = False      # adaptive threshold (use carefully)

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #
    def __init__(self, output_dir: str = "data/Sections/exams", gpu: bool = False):
        self._setup_logging()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug").mkdir(exist_ok=True)

        # Pre-compute normalised keyword lists
        self.KEY_WORDS_NORM = list({normalize_arabic(kw) for kw in self.KEY_WORDS_RAW})
        self.EXCLUDED_KEYWORDS_NORM = list({normalize_arabic(kw) for kw in self.EXCLUDED_KEYWORDS_RAW})

        self.logger.info(f"Normalised keywords: {self.KEY_WORDS_NORM}")
        self.logger.info(f"Normalised exclusions: {self.EXCLUDED_KEYWORDS_NORM}")

        self.logger.info("Loading EasyOCR (Arabic + English) …")
        self.reader = easyocr.Reader(["ar", "en"], gpu=gpu)
        self.logger.info("EasyOCR ready.")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a chain of preprocessing steps to maximise OCR accuracy.
        Returns a BGR image (EasyOCR accepts BGR or grayscale).
        """
        h, w = image.shape[:2]
        result = image.copy()

        # 1. Upscale small images
        if self.PREPROCESS_UPSCALE:
            max_dim = max(h, w)
            if max_dim < self.PREPROCESS_UPSCALE_MAX_DIM:
                scale = min(self.PREPROCESS_UPSCALE_FACTOR,
                            self.PREPROCESS_UPSCALE_MAX_DIM / max_dim)
                if scale > 1.05:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    result = cv2.resize(result, (new_w, new_h),
                                        interpolation=cv2.INTER_CUBIC)
                    self.logger.info(f"Upscaled {w}×{h} → {new_w}×{new_h} (×{scale:.2f})")

        # 2. Convert to grayscale for enhancement
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # 3. Denoising
        if self.PREPROCESS_DENOISE:
            gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                             searchWindowSize=21)

        # 4. CLAHE contrast enhancement
        if self.PREPROCESS_CONTRAST:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # 5. Optional adaptive threshold (binarisation)
        if self.PREPROCESS_THRESHOLD:
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=31, C=10,
            )

        # Return as BGR so EasyOCR is happy
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return result

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #
    def _run_ocr(self, image: np.ndarray) -> list[dict]:
        """
        Run EasyOCR on original image AND a preprocessed version,
        merge results, and return deduplicated detections.
        """
        all_results: list[dict] = []

        def _parse(raw, paragraph_mode: bool = False):
            """
            Parse EasyOCR output safely.
            - detail=1, paragraph=False → (bbox, text, conf)   3-tuple
            - detail=1, paragraph=True  → (bbox, text)         2-tuple  ← EasyOCR quirk
            - detail=0                  → text only            (we never use this)
            """
            parsed = []
            for item in raw:
                if paragraph_mode or len(item) == 2:
                    bbox, text = item
                    conf = 0.5          # paragraph mode gives no per-word confidence
                elif len(item) == 3:
                    bbox, text, conf = item
                else:
                    self.logger.warning(f"Unexpected OCR item length {len(item)}: {item}")
                    continue
                parsed.append({
                    "text": text.strip(),
                    "bbox": bbox,
                    "conf": float(conf),
                })
            return parsed

        # Pass 1 — original image
        self.logger.info("OCR pass 1: original image")
        raw1 = self.reader.readtext(image, detail=1, paragraph=False)
        all_results.extend(_parse(raw1))

        # Pass 2 — preprocessed image (different signal)
        preprocessed = self._preprocess(image)
        self.logger.info("OCR pass 2: preprocessed image")
        raw2 = self.reader.readtext(preprocessed, detail=1, paragraph=False)
        all_results.extend(_parse(raw2))

        # Pass 3 — paragraph mode (catches words missed by word-level detection)
        self.logger.info("OCR pass 3: paragraph mode")
        raw3 = self.reader.readtext(image, detail=1, paragraph=True)
        all_results.extend(_parse(raw3, paragraph_mode=True))

        self.logger.info(f"Total OCR detections (before dedup): {len(all_results)}")
        return all_results

    # ------------------------------------------------------------------ #
    # Keyword matching
    # ------------------------------------------------------------------ #
    def _is_excluded(self, text_norm: str) -> bool:
        """True if the text matches any exclusion keyword."""
        for excl in self.EXCLUDED_KEYWORDS_NORM:
            # Use token_set_ratio to handle reordering / extra words
            if fuzz.partial_ratio(excl, text_norm) >= self.EXCLUSION_THRESHOLD:
                self.logger.debug(f"  Excluded '{text_norm}' (matched exclusion '{excl}')")
                return True
        return False

    def _match_keyword(self, text_norm: str) -> tuple[bool, str, int]:
        """
        Returns (matched, best_keyword, score).
        Uses multiple fuzzy strategies and returns the best score.
        """
        best_score = 0
        best_kw = ""
        for kw in self.KEY_WORDS_NORM:
            # Strategy 1: partial_ratio — keyword as substring of OCR token
            s1 = fuzz.partial_ratio(kw, text_norm)
            # Strategy 2: ratio — overall similarity
            s2 = fuzz.ratio(kw, text_norm)
            # Strategy 3: token_set_ratio — robust to extra tokens
            s3 = fuzz.token_set_ratio(kw, text_norm)
            score = max(s1, s2, s3)
            if score > best_score:
                best_score = score
                best_kw = kw
        matched = best_score >= self.SIMILARITY_THRESHOLD
        return matched, best_kw, best_score

    def _is_section_keyword(self, text: str) -> tuple[bool, str, int]:
        """
        Full pipeline: normalise → exclude check → fuzzy match.
        Returns (is_keyword, matched_kw, score).
        """
        text_norm = normalize_arabic(text)

        if not text_norm:
            return False, "", 0

        if self._is_excluded(text_norm):
            return False, "", 0

        matched, kw, score = self._match_keyword(text_norm)
        return matched, kw, score

    # ------------------------------------------------------------------ #
    # Section-line detection
    # ------------------------------------------------------------------ #
    def _detect_split_y(self, image: np.ndarray) -> list[int]:
        """
        Return sorted, de-duplicated list of Y pixel positions where the
        image should be split.

        Key improvement: we work in the *original* image's coordinate space
        even when OCR was run on a preprocessed/upscaled copy, by tracking
        scale factors.
        """
        h_orig, w_orig = image.shape[:2]
        ocr_data = self._run_ocr(image)

        # When pass 2 runs on a resized image we need to map coords back.
        # Since _run_ocr merges all passes on the same input size (original),
        # no remapping is needed here. If you split passes and resize, adjust.

        raw_y: list[int] = []
        debug_img = image.copy()
        seen_boxes: list[tuple[int, int]] = []   # (y_top, y_bot) for near-duplicate removal

        for item in ocr_data:
            text = item["text"]
            bbox = item["bbox"]
            conf = item["conf"]

            if conf < self.MIN_CONFIDENCE:
                continue

            matched, kw, score = self._is_section_keyword(text)

            if matched:
                ys = [int(pt[1]) for pt in bbox]
                xs = [int(pt[0]) for pt in bbox]
                y_top = max(0, min(ys))
                y_bot = min(h_orig, max(ys))
                x_left = max(0, min(xs))
                x_right = min(w_orig, max(xs))

                # Near-duplicate suppression (same word detected by multiple passes)
                is_dup = False
                for prev_top, prev_bot in seen_boxes:
                    overlap_y = min(y_bot, prev_bot) - max(y_top, prev_top)
                    union_y = max(y_bot, prev_bot) - min(y_top, prev_top)
                    if union_y > 0 and overlap_y / union_y > 0.5:
                        is_dup = True
                        break

                if is_dup:
                    self.logger.debug(f"  Skipping duplicate bbox for '{text}' at y={y_top}")
                    continue

                seen_boxes.append((y_top, y_bot))
                raw_y.append(y_top)
                self.logger.info(
                    f"  Keyword '{text}' → norm='{normalize_arabic(text)}' "
                    f"matched='{kw}' score={score} conf={conf:.2f} → split at y={y_top}"
                )

                # Draw on debug image
                cv2.rectangle(debug_img, (x_left, y_top), (x_right, y_bot),
                              (0, 200, 0), 3)
                cv2.line(debug_img, (0, y_top), (w_orig, y_top), (0, 0, 255), 2)
                cv2.putText(
                    debug_img, f"{text[:15]} [{score}]",
                    (x_left, max(y_top - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                )
            else:
                # Log near-miss for debugging
                _, kw_miss, score_miss = self._match_keyword(normalize_arabic(text))
                if score_miss >= self.SIMILARITY_THRESHOLD - 15:
                    self.logger.debug(
                        f"  NEAR-MISS: '{text}' → score={score_miss} "
                        f"(threshold={self.SIMILARITY_THRESHOLD})"
                    )

        # Save debug image
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = self.output_dir / "debug" / f"ocr_debug_{ts}.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        self.logger.info(f"Debug image saved → {debug_path}")

        filtered = sorted(set(raw_y))
        self.logger.info(f"Split lines after dedup: {filtered}")
        return filtered

    # ------------------------------------------------------------------ #
    # Splitting
    # ------------------------------------------------------------------ #
    def split_image(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Cut image at detected keyword Y-positions.
        Content above the first keyword becomes section 0 (intro).
        """
        h = image.shape[0]
        split_ys = self._detect_split_y(image)

        if not split_ys:
            self.logger.warning("No section keywords found — returning whole image.")
            return [image]

        boundaries = split_ys + [h]
        sections: list[np.ndarray] = []

        # Intro block (before first keyword)
        if split_ys[0] > self.MIN_SECTION_HEIGHT_PX:
            intro = image[0: split_ys[0], :]
            sections.append(intro)
            self.logger.info(f"  Section {len(sections)} (intro): rows 0–{split_ys[0]}")

        # One section per keyword
        for i in range(len(boundaries) - 1):
            y0 = boundaries[i]
            y1 = min(h, boundaries[i + 1])
            if (y1 - y0) >= self.MIN_SECTION_HEIGHT_PX:
                sections.append(image[y0:y1, :])
                self.logger.info(f"  Section {len(sections)}: rows {y0}–{y1}")

        return sections

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    def save_sections(self, sections: list[np.ndarray], exam_id) -> list[str]:
        save_dir = self.output_dir / f"exam_{exam_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i, sec in enumerate(sections, 1):
            p = save_dir / f"exam_{exam_id}_section_{i:02d}.jpg"
            cv2.imwrite(str(p), sec)
            paths.append(str(p))
            self.logger.info(f"  Saved → {p}")
        return paths

    # ------------------------------------------------------------------ #
    # Diagnostic helper
    # ------------------------------------------------------------------ #
    def diagnose_image(self, image_path: str) -> None:
        """
        Run OCR and print ALL detected tokens with their normalised forms
        and fuzzy scores against every keyword. Useful for debugging missed splits.
        """
        image = cv2.imread(image_path)
        if image is None:
            print("ERROR: cannot load image")
            return

        ocr_data = self._run_ocr(image)
        print(f"\n{'='*70}")
        print(f"DIAGNOSIS: {image_path}")
        print(f"{'='*70}")
        print(f"{'RAW TEXT':<30} {'NORM':<20} {'BEST KW':<15} {'SCORE':>6} {'CONF':>6}")
        print("-" * 70)

        for item in ocr_data:
            text = item["text"]
            conf = item["conf"]
            text_norm = normalize_arabic(text)
            _, kw, score = self._match_keyword(text_norm)
            flag = " ✓ MATCH" if score >= self.SIMILARITY_THRESHOLD else ""
            excl = " [EXCLUDED]" if self._is_excluded(text_norm) else ""
            print(f"{text[:28]:<30} {text_norm[:18]:<20} {kw[:13]:<15} {score:>6} {conf:>6.2f}{flag}{excl}")

        print("=" * 70)

    # ------------------------------------------------------------------ #
    # Main entry-point
    # ------------------------------------------------------------------ #
    def split_and_save(
        self,
        image_path: str,
        exam_id=1,
        return_sections: bool = False,
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

        result = {
            "success": True,
            "exam_id": exam_id,
            "num_sections": len(sections),
            "saved_paths": saved_paths,
        }
        if return_sections:
            result["sections"] = sections
        return result