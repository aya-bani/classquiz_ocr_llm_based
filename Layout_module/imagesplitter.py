import os
import cv2
import logging
import re
from pathlib import Path
from datetime import datetime

import easyocr
from rapidfuzz import fuzz


class ImageSplitter:
    """
    Splits a vertically-merged exam image into sections by detecting
    Arabic section-header keywords (e.g. "تعليمة", "سند") via EasyOCR
    and fuzzy matching.
    """

    # ------------------------------------------------------------------ #
    # Config
    # ------------------------------------------------------------------ #
    KEY_WORDS: list[str] = ["تعليمة", "سند"]
    EXCLUDED_KEYWORDS: list[str] = ["تسند"]
    SIMILARITY_THRESHOLD: int = 70          # rapidfuzz score 0-100
    MIN_SECTION_HEIGHT_PX: int = 80         # ignore tiny slices
    DEDUP_RATIO: float = 0.03               # min gap between split lines (3% of height)
    SECTION_PADDING: int = 10               # pixels of overlap on each slice

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #
    def __init__(self, output_dir: str = "data/Sections/exams"):
        self._setup_logging()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug").mkdir(exist_ok=True)

        self.logger.info("Loading EasyOCR (Arabic + English) …")
        # gpu=False is safe everywhere; set True if CUDA is available
        self.reader = easyocr.Reader(["ar", "en"], gpu=False)
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
    # OCR
    # ------------------------------------------------------------------ #
    def _run_ocr(self, image) -> list[dict]:
        """
        Run EasyOCR and return a flat list of
        {"text": str, "bbox": [[x,y],[x,y],[x,y],[x,y]], "conf": float}
        """
        results = self.reader.readtext(image, detail=1, paragraph=False)
        parsed = []
        for bbox, text, conf in results:
            parsed.append({"text": text.strip(), "bbox": bbox, "conf": conf})
        return parsed

    # ------------------------------------------------------------------ #
    # Keyword matching
    # ------------------------------------------------------------------ #
    def _is_section_keyword(self, text: str) -> bool:
        """Return True if `text` fuzzy-matches a KEY_WORD but NOT an EXCLUDED_KEYWORD."""
        # Hard exclusion first
        for excl in self.EXCLUDED_KEYWORDS:
            if fuzz.partial_ratio(excl, text) >= self.SIMILARITY_THRESHOLD:
                return False
        # Positive match
        for kw in self.KEY_WORDS:
            if fuzz.partial_ratio(kw, text) >= self.SIMILARITY_THRESHOLD:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Section-line detection
    # ------------------------------------------------------------------ #
    def _detect_split_y(self, image) -> list[int]:
        """
        Return sorted, de-duplicated list of Y pixel positions where the
        image should be split (top edge of each detected keyword bbox).
        """
        ocr_data = self._run_ocr(image)
        img_h, img_w = image.shape[:2]

        raw_y: list[int] = []
        debug_img = image.copy()

        for item in ocr_data:
            text = item["text"]
            bbox = item["bbox"]   # 4 corners [[x,y],…]
            conf = item["conf"]

            if conf < 0.3:
                continue

            if self._is_section_keyword(text):
                # Top-left y of the bounding box
                ys = [int(pt[1]) for pt in bbox]
                xs = [int(pt[0]) for pt in bbox]
                y_top = min(ys)
                y_bot = max(ys)
                x_left = min(xs)
                x_right = max(xs)
                y_center = (y_top + y_bot) // 2

                raw_y.append(y_top)
                self.logger.info(
                    f"  Keyword '{text}' (conf={conf:.2f}) → split at y={y_top}"
                )

                # Draw on debug image
                cv2.rectangle(debug_img, (x_left, y_top), (x_right, y_bot), (0, 200, 0), 2)
                cv2.line(debug_img, (0, y_top), (img_w, y_top), (0, 0, 255), 2)
                cv2.putText(
                    debug_img, text[:20], (x_left, max(y_top - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
                )

        # Save debug image
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = self.output_dir / "debug" / f"ocr_debug_{ts}.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        self.logger.info(f"Debug image saved → {debug_path}")

        # De-duplicate: keep only splits that are > 3% of image height apart
        raw_y = sorted(set(raw_y))
        min_gap = int(img_h * self.DEDUP_RATIO)
        filtered: list[int] = []
        for y in raw_y:
            if not filtered or (y - filtered[-1]) >= min_gap:
                filtered.append(y)

        self.logger.info(f"Split lines after dedup: {filtered}")
        return filtered

    # ------------------------------------------------------------------ #
    # Splitting
    # ------------------------------------------------------------------ #
    def split_image(self, image) -> list:
        """Cut image at detected keyword Y-positions and return section crops."""
        h = image.shape[0]
        split_ys = self._detect_split_y(image)

        if not split_ys:
            self.logger.warning("No section keywords found — returning whole image.")
            return [image]

        boundaries = [0] + split_ys + [h]
        sections = []
        for i in range(len(boundaries) - 1):
            y0 = max(0, boundaries[i] - self.SECTION_PADDING)
            y1 = min(h, boundaries[i + 1] + self.SECTION_PADDING)
            if (y1 - y0) >= self.MIN_SECTION_HEIGHT_PX:
                sections.append(image[y0:y1, :])
                self.logger.info(f"  Section {len(sections)}: rows {y0}–{y1}")

        return sections

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    def save_sections(self, sections: list, exam_id) -> list[str]:
        save_dir = self.output_dir / f"exam_{exam_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, sec in enumerate(sections, 1):
            p = save_dir / f"exam_{exam_id}_section_{i:02d}.jpg"
            cv2.imwrite(str(p), sec)
            paths.append(str(p))
            self.logger.info(f"  Saved → {p}")
        return paths

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