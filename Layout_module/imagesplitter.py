import cv2
import numpy as np
import os
import logging
from datetime import datetime
from paddleocr import PaddleOCR
import json
import re


class ImageSplitter:
    def __init__(self, output_dir='Exams/splited images into sections'):
        """
        Initialize ImageSplitter with Arabic PaddleOCR
        """
        self.setup_logging()

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)

        self.logger.info("Initializing PaddleOCR (Arabic)...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ar')
        self.logger.info("PaddleOCR initialized successfully")

        # Arabic section keywords
        self.section_keywords = [
            "تعليمة",
            "سند",
            "التمرين",
            "تمرين",
            "السؤال",
            "سؤال"
        ]

        # Regex patterns for Arabic sections
        self.section_patterns = [
            r"تعليمة\s*\d*",
            r"التمرين\s*\d+",
            r"تمرين\s*\d+",
            r"السؤال\s*\d+",
            r"سؤال\s*\d+"
        ]

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    # ---------------------------------------------------
    # Image Preprocessing
    # ---------------------------------------------------
    def preprocess_image(self, image):
        """
        Preprocess image to improve OCR accuracy
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Increase contrast
        gray = cv2.equalizeHist(gray)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return binary

    # ---------------------------------------------------
    # Section Detection
    # ---------------------------------------------------
    def detect_section_lines(self, image):
        """
        Detect Arabic section markers
        Returns list of Y positions
        """
        self.logger.info("Running OCR to detect Arabic section markers...")

        try:
            # Use original image for OCR
            result = self.ocr.ocr(image)
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            self.logger.warning("Falling back to no section detection")
            return []

        section_positions = []
        debug_image = image.copy()

        if result and result[0]:
            for line in result:
                for word_info in line:
                    bbox = np.array(word_info[0]).astype(int)
                    text = word_info[1][0]
                    confidence = word_info[1][1]

                    text_clean = text.strip()

                    # Draw OCR bounding box
                    cv2.polylines(debug_image, [bbox], True, (0, 255, 0), 2)

                    if confidence < 0.6:
                        continue

                    # Check simple keyword match
                    keyword_match = any(
                        re.search(rf"\b{kw}\b", text_clean)
                        for kw in self.section_keywords
                    )

                    # Check regex patterns
                    pattern_match = any(
                        re.search(pattern, text_clean)
                        for pattern in self.section_patterns
                    )

                    if keyword_match or pattern_match:
                        y_coords = [point[1] for point in bbox]
                        y_center = int(sum(y_coords) / len(y_coords))

                        section_positions.append(y_center)

                        self.logger.info(
                            f"Detected section marker '{text_clean}' at y={y_center}"
                        )

                        # Draw horizontal split line
                        cv2.line(
                            debug_image,
                            (0, y_center),
                            (image.shape[1], y_center),
                            (0, 0, 255),
                            2
                        )

        # Remove duplicates and very close lines
        section_positions = sorted(section_positions)
        filtered_positions = []

        for y in section_positions:
            if not filtered_positions:
                filtered_positions.append(y)
            elif abs(y - filtered_positions[-1]) > 50:
                filtered_positions.append(y)

        # Save debug image
        debug_path = os.path.join(
            self.output_dir,
            'debug',
            f'ocr_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        )
        cv2.imwrite(debug_path, debug_image)
        self.logger.info(f"Saved debug image to {debug_path}")

        self.logger.info(f"Final detected section markers: {len(filtered_positions)}")
        return filtered_positions

    # ---------------------------------------------------
    # Split Image
    # ---------------------------------------------------
    def split_image(self, image):
        """
        Split image into sections based on detected Y positions
        """
        self.logger.info("Splitting image into sections")

        height = image.shape[0]
        section_lines = self.detect_section_lines(image)

        if not section_lines:
            self.logger.warning("No section markers found")
            return [image]

        split_points = [0] + section_lines + [height]
        sections = []

        for i in range(len(split_points) - 1):
            start_y = split_points[i]
            end_y = split_points[i + 1]

            # Add padding
            padding = 10
            start_y = max(0, start_y - padding)
            end_y = min(height, end_y + padding)

            if end_y - start_y > 80:  # minimum height
                section = image[start_y:end_y, :]
                sections.append(section)
                self.logger.info(
                    f"Created section {i+1}: y={start_y}-{end_y}"
                )

        return sections

    # ---------------------------------------------------
    # Save Sections
    # ---------------------------------------------------
    def save_sections(self, sections, exam_id):
        saved_paths = []
        save_dir = os.path.join(self.output_dir, f'exam_{exam_id}')
        os.makedirs(save_dir, exist_ok=True)

        for i, section in enumerate(sections, 1):
            filename = f"exam_{exam_id}_section_{i:02d}.jpg"
            path = os.path.join(save_dir, filename)
            cv2.imwrite(path, section)
            saved_paths.append(path)
            self.logger.info(f"Saved section {i} -> {path}")

        return saved_paths

    # ---------------------------------------------------
    # Main Function
    # ---------------------------------------------------
    def split_and_save(self, image_path, exam_id=1, return_sections=False):
        self.logger.info(f"Processing exam {exam_id}")

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
            "saved_paths": saved_paths
        }

        if return_sections:
            result["sections"] = sections

        return result