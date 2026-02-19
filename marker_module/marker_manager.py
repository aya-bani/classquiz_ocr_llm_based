from pathlib import Path
from .coordinate_mapper import CoordinateMapper
from .marker_generator import MarkerGenerator
from .marker_scanner import ExamScanner
from .marker_config import MarkerConfig
from pdf2image import convert_from_path
from logger_manager import LoggerManager
from PIL import Image
import numpy as np
import cv2

class MarkerManager:
    """
    Manages the marking and scanning of exam pages with ArUco markers.
    Coordinates between marker generation, scanning, and coordinate mapping.
    """

    def __init__(self):
        self.logger = LoggerManager.get_logger(__name__)
        MarkerConfig.create_directories()
        self.logger.info("MarkerManager initialized with provided configuration")
        self.marker_generator = MarkerGenerator()

    def mark_exam(self, exam_id: int, exam_path:Path) -> dict:
        """
        Add ArUco markers to exam pages and save to PDF.
        
        Args:
            exam_id: Unique identifier for the exam
            exam_path: Path to the exam PDF file
            
        Returns:
            Dictionary containing exam_id, marked_pages, num_pages, output_path
        """
        pages = self._load_pdf(exam_path)
        self.logger.info(f"Marking exam {exam_id} with {len(pages)} pages")
        
        marked_pages = self.marker_generator.generate_marked_exam(exam_id, pages)
        output_path = self._save_pages_to_pdf(marked_pages, exam_id)
        
        return {
            'exam_id': exam_id,
            'num_pages': len(marked_pages),
            'output_path': output_path
        }
    
    def scan_submission(self, submission_id: int, pages: list) -> list:
        """
        Scan submitted exam pages, detect markers, and correct perspective.
        
        Returns:
            List of dictionaries, one per exam found
        """
        self.logger.info(
            f"Scanning submission {submission_id} for exam with {len(pages)} pages"
        )
        
        scan_results = ExamScanner.scan_multiple_pages(pages)
        organized_pages = ExamScanner.organize_by_page(scan_results)
        
        results = []
        
        # Process each exam found
        for exam_id, exam_pages in organized_pages.items():
            scanned_pages = []
            for page_num in sorted(exam_pages.keys()):
                page_result = exam_pages[page_num]
                image_index = page_result['image_index']
                original_image = pages[image_index]

                corrected_image = CoordinateMapper.extract_full_document(
                    original_image, 
                    page_result
                )
                print("--------------------------------")
                print(corrected_image)
                scanned_pages.append(corrected_image)
            
            output_path = self._save_pages_to_pdf(scanned_pages, exam_id, submission_id)
            
            results.append({
                'exam_id': exam_id,
                'num_pages': len(scanned_pages),
                'output_path': output_path
            })
        
        return results
    
    def _save_pages_to_pdf(self, processed_pages: list, exam_id: int, submission_id: int = None):
        """Save all processed pages to a single PDF file"""
        if not processed_pages:
            raise ValueError("No pages to save")
        
        if submission_id is not None:
            output_prefix = f"exam_{exam_id}_submission_{submission_id}"
            directory = MarkerConfig.SCANNED_SUBMISSIONS_PATH
        else:
            output_prefix = f"exam_{exam_id}"
            directory = MarkerConfig.MARKED_EXAMS_PATH
        
        # Create output path
        output_path = directory / f"{output_prefix}.pdf"
        
        # Save first page and append the rest
        processed_pages[0].save(
            output_path,
            "PDF",
            save_all=True,
            append_images=processed_pages[1:] if len(processed_pages) > 1 else []
        )
        return output_path
    
    def _load_pdf(self, pdf_path:Path) -> tuple:
        """Load PDF file and return list of pages"""
        pages = convert_from_path(pdf_path, dpi=300)
        return pages
    
    @classmethod
    def _pil_to_opencv(cls, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)."""
        img_array = np.array(pil_image)
        
        # Convert RGB to BGR if needed (PIL uses RGB, OpenCV uses BGR)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array