import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from marker_module.marker_scanner import ExamScanner
from logger_manager import LoggerManager


def test_scan_exam_image():
    """Test ExamScanner on a real exam image"""
    logger = LoggerManager.get_logger(__name__)
    logger.info("Testing ExamScanner on real exam image")
    
    # Load image
    img_path = str(project_root / "Exams" / "new_real_exams" / "hand_writting_corrected_exam" / "math_corrige1.jpeg")
    image = cv2.imread(img_path)
    
    if image is None:
        logger.error(f"Failed to load image: {img_path}")
        print(f"ERROR: Image not found at {img_path}")
        return
    
    logger.info(f"Loaded image: {img_path} ({image.shape})")
    print(f"\n{'='*60}")
    print(f"SCANNING: {img_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"{'='*60}\n")
    
    # Scan the page
    result = ExamScanner.scan_page(image)
    
    # Display results
    print(f"SUCCESS: {result['success']}")
    print(f"EXAM ID: {result.get('exam_id', 'N/A')}")
    print(f"PAGE NUMBER: {result.get('page_number', 'N/A')}")
    print(f"Markers Found: {result.get('markers_found', 0)}/{result.get('expected_markers', 4)}")
    print(f"Dynamic Markers: {result.get('dynamic_markers_found', 0)}")
    print(f"Fixed Markers: {result.get('fixed_markers_found', 0)}")
    print(f"All Corners Detected: {result.get('all_corners_detected', False)}")
    print(f"Inferred Corner: {result.get('inferred_corner', 'None')}")
    
    if not result['success']:
        print(f"\nERROR: {result.get('error', 'Unknown error')}")
    
    if result.get('detected_markers'):
        print(f"\nDetected Markers:")
        for marker in result['detected_markers']:
            inferred = " (inferred)" if 'inferred' in marker else ""
            print(f"  - ID: {marker.get('marker_id', 'N/A')}, "
                  f"Exam: {marker.get('exam_id', 'N/A')}, "
                  f"Page: {marker.get('page_number', 'N/A')}, "
                  f"Corner: {marker.get('corner', 'N/A')}{inferred}")
    
    if result.get('paper_corners') is not None:
        print(f"\nPaper Corners Detected:")
        for i, corner in enumerate(['TL', 'TR', 'BL', 'BR']):
            print(f"  - {corner}: {result['paper_corners'][i]}")
    
    print(f"\n{'='*60}\n")
    
    return result


if __name__ == "__main__":
    result = test_scan_exam_image()
