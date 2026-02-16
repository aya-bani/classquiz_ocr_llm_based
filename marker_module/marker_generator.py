import cv2
import numpy as np
from PIL import Image
from .marker_config import MarkerConfig
from logger_manager import LoggerManager


class MarkerGenerator:
    """Generates and places ArUco markers on exam pages."""


    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(MarkerConfig.DICT_TYPE)
        self.logger = LoggerManager.get_logger(__name__)
        self.logger.info("MarkerGenerator initialized")

    def calculate_markers(self, exam_id: int, page_number: int) -> list:
        if not (0 <= page_number < MarkerConfig.PAGES_PER_EXAM):
            self.logger.error(f"Invalid page_number {page_number}")
            raise ValueError(f"page_number must be between 0 and {MarkerConfig.PAGES_PER_EXAM - 1}")

        if not (0 <= exam_id < MarkerConfig.MAX_EXAMS):
            self.logger.error(f"Invalid exam_id {exam_id}")
            raise ValueError(
                f"exam_id must be between 0 and {MarkerConfig.MAX_EXAMS - 1}. "
                f"Current dictionary supports only {MarkerConfig.MAX_EXAMS} exams."
            )

        base = exam_id * MarkerConfig.BLOCK_SIZE + page_number * MarkerConfig.CORNERS_PER_PAGE
        marker_ids = [base + i for i in range(MarkerConfig.CORNERS_PER_PAGE)]

        if marker_ids[-1] > MarkerConfig.MAX_MARKER_ID:
            self.logger.error(f"Marker ID {marker_ids[-1]} exceeds dictionary capacity")
            raise ValueError(
                f"Generated marker ID {marker_ids[-1]} exceeds dictionary capacity {MarkerConfig.MAX_MARKER_ID}"
            )

        self.logger.debug(f"Marker IDs for exam {exam_id}, page {page_number}: {marker_ids}")
        return marker_ids

    def generate_marker(self, marker_id: int) -> np.ndarray:
        if not (0 <= marker_id <= MarkerConfig.MAX_MARKER_ID):
            self.logger.error(f"Invalid marker_id {marker_id}")
            raise ValueError(f"Marker ID must be between 0 and {MarkerConfig.MAX_MARKER_ID}")

        self.logger.debug(f"Generating marker {marker_id}")
        return cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, MarkerConfig.MARKER_SIZE)

    def add_markers_to_page(self, exam_id: int, page_image: Image.Image, page_number: int) -> np.ndarray:
        self.logger.info(f"Adding markers to exam {exam_id}, page {page_number}")

        # Convert PIL Image to numpy array if needed
        img_array = np.array(page_image) if isinstance(page_image, Image.Image) else page_image.copy()

        # Ensure BGR format
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        height, width = img_array.shape[:2]

        marker_ids = self.calculate_markers(exam_id, page_number)
        markers = [self.generate_marker(mid) for mid in marker_ids]
        markers_bgr = [cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in markers]

        positions = [
            (MarkerConfig.MARGIN, MarkerConfig.MARGIN),  # Top-left
            (width - MarkerConfig.MARKER_SIZE - MarkerConfig.MARGIN, MarkerConfig.MARGIN),  # Top-right
            (MarkerConfig.MARGIN, height - MarkerConfig.MARKER_SIZE - MarkerConfig.MARGIN),  # Bottom-left
            (width - MarkerConfig.MARKER_SIZE - MarkerConfig.MARGIN, height - MarkerConfig.MARKER_SIZE - MarkerConfig.MARGIN)  # Bottom-right
        ]

        for marker, (x, y) in zip(markers_bgr, positions):
            img_array[y:y+MarkerConfig.MARKER_SIZE, x:x+MarkerConfig.MARKER_SIZE] = marker

        self.logger.info(f"Markers added to exam {exam_id}, page {page_number}")
        return img_array

    def get_exam_marker_range(self, exam_id: int) -> tuple:
        if not (0 <= exam_id < MarkerConfig.MAX_EXAMS):
            self.logger.error(f"Invalid exam_id {exam_id} for marker range")
            raise ValueError(f"exam_id must be between 0 and {MarkerConfig.MAX_EXAMS - 1}")

        first = exam_id * MarkerConfig.BLOCK_SIZE
        last = first + MarkerConfig.BLOCK_SIZE - 1
        self.logger.debug(f"Marker range for exam {exam_id}: {first}-{last}")
        return (first, last)
    

    def generate_marked_exam(self, exam_id: int,  pages: list[Image.Image]) -> list[Image.Image]:
        """Generate markers for all pages of a given exam"""
        self.logger.info(f"Generating markers for exam_id={exam_id}")
        marked_pages = []
        for page_number, page in enumerate(pages):
            page_with_markers = self.add_markers_to_page(
                exam_id,
                page, 
                page_number
            )
        
            page_with_markers = cv2.cvtColor(page_with_markers, cv2.COLOR_BGR2RGB)
            page_with_markers = Image.fromarray(page_with_markers)
            marked_pages.append(page_with_markers)

        return marked_pages
