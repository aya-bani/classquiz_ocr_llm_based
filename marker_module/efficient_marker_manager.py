from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

from logger_manager import LoggerManager
from .marker_config import MarkerConfig
from .marker_generator import MarkerGenerator
from .marker_scanner import ExamScanner
from .coordinate_mapper import CoordinateMapper

ImageInput = Union[Image.Image, str, Path]


class EfficientMarkerManager:
    """
    Two-pipeline manager for marker-based exam processing.

    Pipeline 1 (marking):
      Input: blank exam path, exam id
      Process: generate markers -> generate marked pages -> save PDF
      Output: path of marked exam PDF

    Pipeline 2 (submission scanning):
      Input: submitted images
      Process: scanner detect -> scanner organize -> mapper correct -> save PDF
      Output: path(s) of corrected PDF(s)
    """

    def __init__(self) -> None:
        self.logger = LoggerManager.get_logger(__name__)
        MarkerConfig.create_directories()
        self.marker_generator = MarkerGenerator()
        self.logger.info("EfficientMarkerManager initialized")

    def mark_exam(self, exam_id: int, exam_path: Union[str, Path]) -> Dict:
        """
        Pipeline 1: mark a blank exam PDF and save marked PDF.
        """
        input_pdf = Path(exam_path)
        if not input_pdf.exists():
            raise FileNotFoundError(f"Exam PDF not found: {input_pdf}")

        pages = self._load_pdf(input_pdf)
        self.logger.info(f"Marking exam {exam_id} with {len(pages)} page(s)")

        marked_pages = self.marker_generator.generate_marked_exam(exam_id, pages)
        output_path = self._save_pages_to_pdf(marked_pages, exam_id=exam_id)

        return {
            "exam_id": exam_id,
            "num_pages": len(marked_pages),
            "output_path": output_path,
        }

    def scan_submission(
        self,
        submission_id: int,
        submitted_images: Sequence[ImageInput],
    ) -> List[Dict]:
        """
        Pipeline 2: scan submitted images, correct pages, and save corrected PDF(s).
        """
        pages = self._normalize_pages(submitted_images)
        self.logger.info(
            f"Scanning submission {submission_id} with {len(pages)} input image(s)"
        )

        markers_coords = ExamScanner.scan_multiple_pages(pages)
        organized_pages = ExamScanner.organize_by_page(markers_coords)

        if not organized_pages:
            self.logger.warning("No valid exam/page groups found after scanning")
            return []

        results: List[Dict] = []

        for exam_id, exam_pages in organized_pages.items():
            corrected_pages: List[Image.Image] = []
            failed_pages: List[int] = []

            for page_num in sorted(exam_pages.keys()):
                page_result = exam_pages[page_num]
                image_index = page_result.get("image_index")
                if image_index is None or not (0 <= image_index < len(pages)):
                    self.logger.warning(
                        f"Exam {exam_id} page {page_num}: invalid image_index={image_index}"
                    )
                    failed_pages.append(page_num)
                    continue

                corrected = self._correct_page_with_mapper(pages[image_index], page_result)
                if corrected is None:
                    self.logger.warning(
                        f"Exam {exam_id} page {page_num}: correction failed"
                    )
                    failed_pages.append(page_num)
                    continue

                corrected_pages.append(corrected)

            if not corrected_pages:
                self.logger.warning(
                    f"Exam {exam_id}: no corrected pages to save (failed_pages={failed_pages})"
                )
                continue

            output_path = self._save_pages_to_pdf(
                corrected_pages,
                exam_id=exam_id,
                submission_id=submission_id,
            )

            results.append(
                {
                    "exam_id": exam_id,
                    "num_pages": len(corrected_pages),
                    "output_path": output_path,
                    "failed_pages": failed_pages,
                }
            )

        return results

    def _correct_page_with_mapper(
        self,
        image_pil: Image.Image,
        page_result: Dict,
    ) -> Optional[Image.Image]:
        """
        Correct one page with CoordinateMapper.

        Uses page_result if it already contains marker corners; otherwise,
        re-detects marker corners on the same page so mapper gets the required
        scan_result fields (`detected_markers` + `corners`).
        """
        if not page_result.get("success", False):
            return None

        # Fast path for scanner versions that already include `corners`
        if "corners" in page_result and page_result.get("detected_markers"):
            mapper_scan_result = {
                "success": True,
                "markers_found": page_result.get(
                    "markers_found", len(page_result["detected_markers"])
                ),
                "detected_markers": page_result["detected_markers"],
                "corners": page_result["corners"],
            }
            return CoordinateMapper.extract_full_document(image_pil, mapper_scan_result)

        # Robust path: recompute marker corners for mapper
        image_bgr = self._pil_to_opencv(image_pil)
        preprocessed = ExamScanner._preprocess_image(image_bgr)
        corners, ids = ExamScanner._detect_markers_with_fallback(image_bgr, preprocessed)

        if ids is None or len(ids) == 0:
            return None

        detected_markers, corners_list, _, _ = ExamScanner._process_markers_with_corners(
            ids, corners
        )

        mapper_scan_result = {
            "success": True,
            "markers_found": len(detected_markers),
            "detected_markers": detected_markers,
            "corners": corners_list,
        }

        return CoordinateMapper.extract_full_document(image_pil, mapper_scan_result)

    def _save_pages_to_pdf(
        self,
        processed_pages: List[Image.Image],
        exam_id: int,
        submission_id: Optional[int] = None,
    ) -> Path:
        """Save processed pages as a single PDF."""
        if not processed_pages:
            raise ValueError("No pages to save")

        if submission_id is None:
            output_prefix = f"exam_{exam_id}"
            directory = MarkerConfig.MARKED_EXAMS_PATH
        else:
            output_prefix = f"exam_{exam_id}_submission_{submission_id}"
            directory = MarkerConfig.SCANNED_SUBMISSIONS_PATH

        output_path = directory / f"{output_prefix}.pdf"
        processed_pages[0].save(
            output_path,
            "PDF",
            save_all=True,
            append_images=processed_pages[1:] if len(processed_pages) > 1 else [],
        )
        return output_path

    @staticmethod
    def _load_pdf(pdf_path: Path) -> List[Image.Image]:
        """Load PDF and return PIL pages."""
        return convert_from_path(pdf_path, dpi=300)

    @staticmethod
    def _normalize_pages(submitted_images: Sequence[ImageInput]) -> List[Image.Image]:
        """Normalize image inputs (PIL/path/str) into RGB PIL list."""
        pages: List[Image.Image] = []
        for item in submitted_images:
            if isinstance(item, Image.Image):
                pages.append(item.convert("RGB"))
                continue

            path = Path(item)
            if not path.exists():
                raise FileNotFoundError(f"Submitted image not found: {path}")
            pages.append(Image.open(path).convert("RGB"))

        if not pages:
            raise ValueError("submitted_images is empty")

        return pages

    @staticmethod
    def _pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL RGB image to OpenCV BGR array."""
        image_array = np.array(pil_image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        return image_array
