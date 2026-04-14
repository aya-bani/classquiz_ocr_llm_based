import os
import io
import numpy as np
from PIL import Image
from typing import List, Tuple
from rapidfuzz import fuzz
from logger_manager import LoggerManager
from .layout_config import LayoutConfig

from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()


class ImageSplitter:
    """
    Splits exam images into exercises using Gemini OCR
    """

    def __init__(self):
        self.logger = LoggerManager.get_logger(__name__)

        GEMINI_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")

        if not GEMINI_API_KEY:
            raise ValueError("❌ GOOGLE_CLOUD_API_KEY not found in .env")

        # ✅ SAME CONFIG
        self.client = genai.Client(
            vertexai=True,
            api_key=GEMINI_API_KEY
        )

        self.logger.info("✅ Gemini OCR client initialized")

    # ---------------- KEYWORD MATCH ---------------- #
    def is_keyword_match(self, word_text: str) -> bool:
        for keyword in LayoutConfig.KEY_WORDS:
            similarity = fuzz.ratio(word_text.lower(), keyword.lower())

            if (
                similarity >= LayoutConfig.SIMILARITY_THRESHOLD
                and word_text not in LayoutConfig.EXCLUDED_KEYWORDS
            ):
                self.logger.debug(
                    f"Keyword match: '{word_text}' ~ '{keyword}' ({similarity}%)"
                )
                return True

        return False

    # ---------------- OCR + DETECTION ---------------- #
    def detect_section_lines(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Gemini OCR → approximate bounding boxes (same format as before)
        """
        self.logger.info("🔍 Running Gemini OCR to detect section markers")

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        content = img_byte_arr.getvalue()

        image_part = types.Part.from_bytes(
            data=content,
            mime_type="image/jpeg",
        )

        OCR_PROMPT = """
Extract ALL visible text from this exam image.
- Preserve line breaks exactly
- Do not explain anything
- Do not summarize
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=[OCR_PROMPT, image_part]
            )

            text = response.text if response else ""

        except Exception as e:
            self.logger.error(f"❌ Gemini OCR failed: {e}")
            return []

        print("\n--------- GEMINI OCR OUTPUT ---------")
        print(text)
        print("------------------------------------\n")

        lines = text.split("\n")
        section_coords = []

        height = image.height
        width = image.width

        if len(lines) == 0:
            return []

        line_height = height / len(lines)

        # SAME STRUCTURE AS VISION OUTPUT (coords)
        for idx, line in enumerate(lines):
            words = line.split()

            for word in words:
                if self.is_keyword_match(word):

                    y_min = int(idx * line_height)
                    y_max = int((idx + 1) * line_height)

                    section_coords.append((0, y_min, width, y_max))

                    self.logger.info(
                        f"✅ Detected section keyword '{word}' at {y_min},{y_max}"
                    )

        # SAME SORT
        section_coords.sort(key=lambda coord: coord[1])

        self.logger.info(f"📊 Detected {len(section_coords)} section lines")
        return section_coords

    # ---------------- SPLITTING (UNCHANGED) ---------------- #
    def split_image(self, image: Image.Image) -> List[Image.Image]:
        self.logger.info("✂️ Splitting image into sections")

        line_coords = self.detect_section_lines(image)

        if not line_coords:
            self.logger.warning("⚠️ No section lines detected; returning full image")
            return [image]

        img_array = np.array(image)
        height, width = img_array.shape[:2]

        sections = []
        y_start = 0

        for i, coords in enumerate(line_coords):
            x_min, y_min, x_max, y_max = coords

            # EXACT SAME LOGIC
            if i == 0:
                crop = img_array[y_start:y_min, 0:width]
                if crop.size != 0:
                    sections.append(Image.fromarray(crop))
                y_start = y_min
            else:
                crop = img_array[y_start:y_min, 0:width]
                if crop.size != 0:
                    sections.append(Image.fromarray(crop))
                y_start = y_min

        # LAST SECTION (UNCHANGED)
        crop = img_array[y_start:height, 0:width]
        if crop.size != 0:
            sections.append(Image.fromarray(crop))
            self.logger.debug(f"Created section from y={y_start} to end")

        self.logger.info(f"📦 Total sections created: {len(sections)}")
        return sections

    # ---------------- SAVE ---------------- #
    def split_and_save(
        self,
        image: Image.Image,
        exam_id: int,
        submission_id: int = None
    ):
        self.logger.info("🚀 Starting split_and_save")

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
            self.logger.info(f"💾 Saved section {i} to {filepath}")

        self.logger.info("✅ Finished split_and_save")

        return {
            'sections_dir': directory,
            'number_of_sections': len(sections)
        }