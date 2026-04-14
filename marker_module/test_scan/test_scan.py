import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from marker_module.marker_scanner import ExamScanner
from marker_module.marker_config import MarkerConfig
from logger_manager import LoggerManager


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ARRANGED_OUTPUT_DIR = project_root / "Exams" / "exams_arraged"


def _read_image_unicode_safe(path: Path):
    """Read image from paths that may include non-ASCII characters on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _resolve_input_path(input_path: str = None) -> Path:
    if input_path is None and len(sys.argv) > 1:
        input_path = sys.argv[1]

    if input_path:
        p = Path(input_path)
        return p if p.is_absolute() else (project_root / p)

    return project_root / "Exams" / "output_mapper" / "ex1_3_dewarped.jpg"


def _print_scan_result(result: dict):
    print(f"SUCCESS: {result['success']}")
    print(f"EXAM ID: {result.get('exam_id', 'N/A')}")
    raw_page_number = result.get('page_number', None)
    display_page_number = (
        raw_page_number + 1 if isinstance(raw_page_number, int) and raw_page_number >= 0 else 'N/A'
    )
    print(f"PAGE NUMBER: {display_page_number}")
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
            marker_page = marker.get('page_number', 'N/A')
            marker_id = marker.get('marker_id', -1)
            marker_corner = marker.get('corner', 'N/A')

            # For inferred fixed corners, display the conventional fixed IDs.
            if 'inferred' in marker and marker_id == -1:
                inferred_fixed_ids = {
                    'top_left': 0,
                    'top_right': 1,
                    'bottom_left': 2,
                }
                marker_id = inferred_fixed_ids.get(marker_corner, marker_id)

            if isinstance(marker_page, int) and marker_page >= 0 and marker_id >= (max(MarkerConfig.FIXED_MARKER_IDS) + 1):
                marker_page = marker_page + 1

            print(f"  - ID: {marker_id}, "
                  f"Exam: {marker.get('exam_id', 'N/A')}, "
                  f"Page: {marker_page}, "
                  f"Corner: {marker_corner}{inferred}")

    if result.get('paper_corners') is not None:
        print(f"\nPaper Corners Detected:")
        for i, corner in enumerate(['TL', 'TR', 'BL', 'BR']):
            print(f"  - {corner}: {result['paper_corners'][i]}")


def test_scan_exam_image(image_path: str = None):
    """Test ExamScanner on a real exam image"""
    logger = LoggerManager.get_logger(__name__)
    logger.info("Testing ExamScanner on real exam image")
    input_path = _resolve_input_path(image_path)
    img_path = str(input_path)
    image = _read_image_unicode_safe(input_path)
    
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
    _print_scan_result(result)
    
    print(f"\n{'='*60}\n")
    
    return result


def test_scan_exam_folder(folder_path: str = None):
    """Scan all images in a folder, print each result, then export ordered PDF(s)."""
    logger = LoggerManager.get_logger(__name__)
    input_folder = _resolve_input_path(folder_path)

    if not input_folder.exists() or not input_folder.is_dir():
        print(f"ERROR: Folder not found: {input_folder}")
        return {
            'success': False,
            'error': f'Folder not found: {input_folder}',
            'results': []
        }

    image_paths = sorted(
        [p for p in input_folder.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS],
        key=lambda p: str(p.relative_to(input_folder)).lower()
    )

    if not image_paths:
        print(f"ERROR: No images found in folder: {input_folder}")
        return {
            'success': False,
            'error': f'No images found in folder: {input_folder}',
            'results': []
        }

    all_results = []
    print(f"\nScanning folder: {input_folder}")
    print(f"Images found: {len(image_paths)}")

    for image_path in image_paths:
        image = _read_image_unicode_safe(image_path)
        if image is None:
            result = {
                'success': False,
                'error': f'Failed to load image: {image_path}',
                'exam_id': None,
                'page_number': None,
                'markers_found': 0
            }
        else:
            result = ExamScanner.scan_page(image)

        result['source_image'] = str(image_path)
        result['source_name'] = str(image_path.relative_to(input_folder))
        all_results.append(result)

        print(f"\n{'='*60}")
        print(f"SCANNING: {image_path}")
        if image is not None:
            print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
        print(f"{'='*60}\n")
        _print_scan_result(result)

    successful = [
        r for r in all_results
        if r.get('success') and r.get('exam_id') is not None and r.get('page_number') is not None
    ]

    pdf_outputs = {}
    if successful:
        ARRANGED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        by_exam = {}
        for result in successful:
            by_exam.setdefault(result['exam_id'], []).append(result)

        for exam_id, results in by_exam.items():
            ordered = sorted(results, key=lambda r: (r['page_number'], r['source_name']))

            pil_images = []
            for item in ordered:
                pil_images.append(Image.open(item['source_image']).convert('RGB'))

            output_pdf = ARRANGED_OUTPUT_DIR / f"exam_{exam_id}_ordered.pdf"
            pil_images[0].save(
                output_pdf,
                "PDF",
                save_all=True,
                append_images=pil_images[1:]
            )

            for im in pil_images:
                im.close()

            pdf_outputs[exam_id] = str(output_pdf)

            display_order = [r['page_number'] + 1 for r in ordered]
            ordered_names = [r['source_name'] for r in ordered]
            logger.info(f"Exam {exam_id} ordered pages: {display_order}")
            print(f"\nOrdered exam {exam_id} pages: {display_order}")
            print(f"Ordered files: {ordered_names}")
            print(f"Output PDF: {output_pdf}")

    return {
        'success': len(successful) > 0,
        'results': all_results,
        'pdf_outputs': pdf_outputs
    }


if __name__ == "__main__":
    input_arg = sys.argv[1] if len(sys.argv) > 1 else None
    resolved = _resolve_input_path(input_arg)
    if resolved.exists() and resolved.is_dir():
        result = test_scan_exam_folder(input_arg)
    else:
        result = test_scan_exam_image(input_arg)
