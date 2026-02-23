"""Complete pipeline: scan images → organize → dewarp → save PDF.

Usage:
    python marker_module/exam_processing_pipeline.py --input-dir Exams/real_exams --output-dir output_exams

Process:
  1. Scan images for markers (detect exam_id, page_number)
  2. Organize pages by exam and page number
  3. Dewarp each page using detected/inferred markers
  4. Save corrected pages as PDF
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

# ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from marker_module.marker_scanner import ExamScanner
from marker_module.coordinate_mapper import CoordinateMapper
from marker_module.marker_config import MarkerConfig
from logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}


def find_images(folder: Path, recursive: bool = True) -> List[Path]:
    """Find all image files in a folder."""
    files: List[Path] = []
    if recursive:
        for p in folder.rglob('*'):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                files.append(p)
    else:
        for p in folder.iterdir():
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                files.append(p)
    return sorted(files)


def scan_and_organize(images: List[Path]) -> Dict[int, Dict[int, Dict]]:
    """
    Scan images and organize by exam_id -> page_number -> data.
    
    Returns:
        {exam_id: {page_number: {'image_path': Path, 'scan_result': Dict, ...}}}
    """
    organized = defaultdict(lambda: defaultdict(dict))
    
    for img_path in images:
        try:
            pil_img = Image.open(img_path).convert('RGB')
            img_cv = ExamScanner._pil_to_opencv(pil_img)
            
            result = ExamScanner.scan_page(img_cv)
            
            if result.get('success'):
                exam_id = result['exam_id']
                page_num = result['page_number']
                organized[exam_id][page_num] = {
                    'image_path': img_path,
                    'pil_image': pil_img,
                    'scan_result': result
                }
                logger.info(f"Scanned {img_path.name} -> exam {exam_id}, page {page_num}")
            else:
                logger.warning(f"Scan failed for {img_path.name}: {result.get('error')}")
        except Exception as e:
            logger.error(f"Error scanning {img_path}: {e}")
    
    return dict(organized)


def try_infer_and_enrich(organized: Dict[int, Dict[int, Dict]]) -> Dict[int, Dict[int, Dict]]:
    """
    For each page, try to infer missing 4th corner if only 3 detected.
    
    Returns updated organized dictionary with inferred corners where applicable.
    """
    for exam_id, pages in organized.items():
        for page_num, page_data in pages.items():
            result = page_data['scan_result']
            detected_markers = result.get('detected_markers', [])
            corners_data = result.get('corners', [])
            
            if len(detected_markers) == 3 and len(corners_data) == 3:
                logger.debug(f"Attempting to infer 4th corner for exam {exam_id}, page {page_num}")
                inferred = ExamScanner.infer_missing_corner(detected_markers, corners_data)
                
                if inferred:
                    corner_name, inferred_coords = inferred
                    # Add inferred corner to result
                    page_data['inferred_corner'] = corner_name
                    page_data['inferred_coords'] = inferred_coords
                    logger.info(f"Inferred {corner_name} for exam {exam_id}, page {page_num}")
    
    return organized


def dewarp_page(pil_image: Image.Image, scan_result: Dict) -> Optional[Image.Image]:
    """
    Dewarp a single page using homography from scan result.
    
    Returns dewarped PIL Image or None if dewarping fails.
    """
    try:
        dewarped = CoordinateMapper.extract_full_document(pil_image, scan_result)
        return dewarped
    except Exception as e:
        logger.warning(f"Dewarping failed: {e}")
        return None


def save_exam_as_pdf(exam_id: int, pages_dict: Dict[int, Dict], output_dir: Path) -> Optional[Path]:
    """
    Collate all pages of an exam and save as PDF.
    
    Pages are dewarped if possible, or original images used.
    
    Returns path to saved PDF or None.
    """
    if not pages_dict:
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"exam_{exam_id}_corrected.pdf"
    
    try:
        # Sort pages by page number
        sorted_pages = sorted(pages_dict.items(), key=lambda x: x[0])
        
        # Collect images in order
        images_for_pdf = []
        for page_num, page_data in sorted_pages:
            pil_img = page_data['pil_image']
            scan_result = page_data['scan_result']
            
            # Try to dewarp
            dewarped = dewarp_page(pil_img, scan_result)
            if dewarped:
                images_for_pdf.append(dewarped)
                logger.debug(f"  page {page_num}: dewarped")
            else:
                # Fallback to original
                images_for_pdf.append(pil_img)
                logger.debug(f"  page {page_num}: original (no dewarp)")
        
        if images_for_pdf:
            # Convert to RGB and save as PDF
            rgb_images = [img.convert('RGB') for img in images_for_pdf]
            rgb_images[0].save(pdf_path, save_all=True, append_images=rgb_images[1:])
            logger.info(f"Saved exam {exam_id} PDF: {pdf_path}")
            return pdf_path
        else:
            logger.warning(f"No images for exam {exam_id}")
            return None
    except Exception as e:
        logger.error(f"Failed to save PDF for exam {exam_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Scan, organize, dewarp, and save exams as PDFs")
    parser.add_argument('--input-dir', '-i', default='Exams/real_exams', help='Input image directory')
    parser.add_argument('--output-dir', '-o', default='output_exams', help='Output PDF directory')
    parser.add_argument('--infer', action='store_true', help='Attempt to infer missing corners')
    parser.add_argument('--recursive', '-r', action='store_true', default=True, help='Search recursively')
    
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    logger.info(f"Starting pipeline: {input_dir} -> {output_dir}")
    
    # Step 1: Find images
    images = find_images(input_dir, args.recursive)
    logger.info(f"Found {len(images)} images")
    
    if not images:
        logger.warning("No images found")
        return
    
    # Step 2: Scan and organize
    organized = scan_and_organize(images)
    logger.info(f"Organized into {len(organized)} exams")
    
    # Step 3: Optionally infer missing corners
    if args.infer:
        organized = try_infer_and_enrich(organized)
    
    # Step 4: Dewarp and save to PDF
    saved_pdfs = []
    for exam_id in sorted(organized.keys()):
        pages = organized[exam_id]
        pdf_path = save_exam_as_pdf(exam_id, pages, output_dir)
        if pdf_path:
            saved_pdfs.append(pdf_path)
    
    logger.info(f"Pipeline complete! Saved {len(saved_pdfs)} PDFs")
    for pdf_path in saved_pdfs:
        print(f"  {pdf_path}")


if __name__ == '__main__':
    main()
