# image_splitter.py

import re
import io
import numpy as np
from typing import List, Tuple
from PIL import Image
from google.cloud import vision
from rapidfuzz import fuzz
from logger_manager import LoggerManager
from .layout_config import LayoutConfig


class ImageSplitter:

    def __init__(self):
        self.logger = LoggerManager.get_logger(__name__)
        self.client = vision.ImageAnnotatorClient(
            client_options={"api_key": LayoutConfig.GOOGLE_API_KEY}
        )
        self.logger.info("Initialized ImageSplitter")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def is_keyword_match(self, word_text: str) -> bool:
        """Fuzzy-match a single token against configured keywords."""
        for keyword in LayoutConfig.KEY_WORDS:
            if (fuzz.ratio(word_text, keyword) >= LayoutConfig.SIMILARITY_THRESHOLD
                    and word_text not in LayoutConfig.EXCLUDED_KEYWORDS):
                self.logger.debug(f"Keyword match: '{word_text}' ~ '{keyword}'")
                return True
        return False

    def is_section_number(self, word_text: str) -> bool:
        """
        Return True if the token matches a section-starter number pattern.
        Handles: -1, 1-, 1., 1/
        """
        for pattern in LayoutConfig.SECTION_NUMBER_PATTERNS:
            if re.match(pattern, word_text.strip()):
                return True
        return False

    def _group_words_into_lines(self, all_words: list) -> list:
        """
        Group word-dicts into lines based on y-proximity.
        Words whose y_min values are within LINE_Y_TOLERANCE px go on the same line.
        """
        all_words.sort(key=lambda w: w["y_min"])

        lines = []
        for word in all_words:
            placed = False
            for line in lines:
                rep_y = line[0]["y_min"]
                if abs(word["y_min"] - rep_y) <= LayoutConfig.LINE_Y_TOLERANCE:
                    line.append(word)
                    placed = True
                    break
            if not placed:
                lines.append([word])
        return lines

    def _line_is_numbered_header(self, line: list) -> bool:
        """
        Check ONLY for numbered section headers (1-, -1, 1., etc.)
        The number/dash must appear at the START of the line (leftmost position),
        not in the middle (which would indicate a math expression).
        """
        if not line:
            return False

        texts = [w["text"] for w in line]

        # Sort words by x position to find leftmost and rightmost
        sorted_by_x = sorted(line, key=lambda w: w["x_min"])
        leftmost_word = sorted_by_x[0]   # leftmost token
        rightmost_word = sorted_by_x[-1] # rightmost token (Arabic starts right)

        # The section number must be the FIRST token (leftmost or rightmost)
        # For Arabic exams: numbers like 1- appear on the LEFT margin
        # Check leftmost token
        first_token = leftmost_word["text"].strip()
        last_token = rightmost_word["text"].strip()

        # Rule B — section number must be at the edge of the line, not middle
        has_edge_number = (
            self.is_section_number(first_token) or
            self.is_section_number(last_token)
        )

        if has_edge_number:
            # Must have accompanying text (not a lone number)
            non_number_tokens = [t for t in texts if not self.is_section_number(t)]
            if non_number_tokens:
                self.logger.debug(f"Rule B match (edge number): {texts}")
                return True

        # Rule C — OCR split: check if digit+dash or dash+digit is at the EDGE
        # Only check first two or last two tokens
        edge_pairs = []
        if len(sorted_by_x) >= 2:
            # leftmost pair
            edge_pairs.append((sorted_by_x[0]["text"].strip(), sorted_by_x[1]["text"].strip()))
            # rightmost pair  
            edge_pairs.append((sorted_by_x[-2]["text"].strip(), sorted_by_x[-1]["text"].strip()))

        for current, next_tok in edge_pairs:
            # "1" followed by "-"
            if re.match(r"^\d+$", current) and next_tok == "-":
                remaining = [t for t in texts
                            if t.strip() != current and t.strip() != next_tok]
                if remaining:
                    self.logger.debug(f"Rule C match (digit+dash at edge): {texts}")
                    return True

            # "-" followed by "1"
            if current == "-" and re.match(r"^\d+$", next_tok):
                remaining = [t for t in texts
                            if t.strip() != current and t.strip() != next_tok]
                if remaining:
                    self.logger.debug(f"Rule C match (dash+digit at edge): {texts}")
                    return True

        return False
    # ------------------------------------------------------------------ #
    #  OCR + detection                                                     #
    # ------------------------------------------------------------------ #

    def detect_section_lines(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Run OCR and return bounding boxes (x_min, y_min, x_max, y_max)
        for every detected section-header line, sorted top-to-bottom.

        Pass 1 — word-level: original logic for keywords (تعليمة / السند / سند)
        Pass 2 — line-level: numbered headers (1- / -1 / 1.)
        """
        self.logger.info("Running OCR to detect section markers")

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        content = img_byte_arr.getvalue()

        vision_image = vision.Image(content=content)
        response = self.client.document_text_detection(image=vision_image)
        doc = response.full_text_annotation

        # Collect every word with its bounding box
        all_words = []
        for page in doc.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = ''.join(s.text for s in word.symbols)
                        box = word.bounding_box.vertices
                        all_words.append({
                            "text":  word_text,
                            "x_min": min(v.x for v in box),
                            "y_min": min(v.y for v in box),
                            "x_max": max(v.x for v in box),
                            "y_max": max(v.y for v in box),
                        })

        self.logger.debug(f"Total words detected by OCR: {len(all_words)}")

        section_coords = []
        matched_y_positions = []  # track already-matched y to avoid duplicates

        # ── Pass 1: word-level keyword matching (original logic) ──────────
        for word in all_words:
            if self.is_keyword_match(word["text"]):
                y_min = word["y_min"]
                if not any(
                    abs(y_min - y) <= LayoutConfig.LINE_Y_TOLERANCE
                    for y in matched_y_positions
                ):
                    section_coords.append((
                        word["x_min"], word["y_min"],
                        word["x_max"], word["y_max"]
                    ))
                    matched_y_positions.append(y_min)
                    self.logger.info(
                        f"Pass1 keyword: '{word['text']}' at y={y_min}"
                    )

        # ── Pass 2: line-level numbered header matching ───────────────────
        lines = self._group_words_into_lines(all_words)
        for line in lines:
            line_y_min = min(w["y_min"] for w in line)

            # Skip lines already matched by a keyword in Pass 1
            if any(
                abs(line_y_min - y) <= LayoutConfig.LINE_Y_TOLERANCE
                for y in matched_y_positions
            ):
                continue

            if self._line_is_numbered_header(line):
                x_min = min(w["x_min"] for w in line)
                x_max = max(w["x_max"] for w in line)
                y_max = max(w["y_max"] for w in line)
                section_coords.append((x_min, line_y_min, x_max, y_max))
                matched_y_positions.append(line_y_min)
                self.logger.info(
                    f"Pass2 numbered: {[w['text'] for w in line]} at y={line_y_min}"
                )

        section_coords.sort(key=lambda c: c[1])
        self.logger.info(f"Total section headers found: {len(section_coords)}")
        return section_coords
    
    # ------------------------------------------------------------------ #
    #  Splitting                                                           #
    # ------------------------------------------------------------------ #

    def split_image(self, image: Image.Image) -> List[Image.Image]:
        """Split image into sections based on detected header lines."""
        self.logger.info("Splitting image into sections")
        line_coords = self.detect_section_lines(image)

        if not line_coords:
            self.logger.warning("No section lines detected; returning full image")
            return [image]

        img_array = np.array(image)
        height, width = img_array.shape[:2]
        sections = []
        y_start = 0

        for i, (x_min, y_min, x_max, y_max) in enumerate(line_coords):
            if i == 0:
                crop = img_array[y_start:y_min, 0:width]
                if crop.size:
                    sections.append(Image.fromarray(crop))
                y_start = y_min
            else:
                crop = img_array[y_start:y_min, 0:width]
                if crop.size:
                    sections.append(Image.fromarray(crop))
                y_start = y_min

        # Last section: from final header to bottom
        crop = img_array[y_start:height, 0:width]
        if crop.size:
            sections.append(Image.fromarray(crop))

        self.logger.info(f"Total sections created: {len(sections)}")
        return sections

    def split_and_save(self, image: Image.Image,
                       exam_id: int, submission_id: int = None) -> dict:
        """Split image and save each section to disk."""
        self.logger.info("Starting split_and_save")
        sections = self.split_image(image)

        if submission_id is not None:
            output_prefix = f"exam_{exam_id}_submission_{submission_id}"
            directory = LayoutConfig.RAW_SUBMISSIONS_PATH / output_prefix
        else:
            output_prefix = f"exam_{exam_id}"
            directory = LayoutConfig.RAW_EXAMS_PATH / output_prefix

        directory.mkdir(parents=True, exist_ok=True)

        for i, section in enumerate(sections):
            filepath = directory / f"{output_prefix}_section_{i}.jpg"
            section.save(filepath, "JPEG")
            self.logger.info(f"Saved section {i} to {filepath}")

        self.logger.info("Finished split_and_save")
        return {"sections_dir": directory, "number_of_sections": len(sections)}