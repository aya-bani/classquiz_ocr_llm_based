"""Enhanced exam processing pipeline with proper document extraction and dewarping.

Workflow:
1. Scan image → find all 4 markers (or infer missing corner)
2. Extract document region bounded by marker corners
3. Dewarp using coordinate mapper
4. Save corrected document only (no background)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
import numpy as np
import cv2

from marker_module.marker_scanner import ExamScanner
from marker_module.coordinate_mapper import CoordinateMapper
from marker_module.marker_config import MarkerConfig
from logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}


def find_images(folder: Path, recursive: bool = True) -> List[Path]:
    """Find all image files."""
    files = []
    search_fn = folder.rglob if recursive else folder.iterdir
    for p in search_fn('*' if recursive else '.'):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            files.append(p)
    return sorted(files)


def extract_document_region(
    pil_image: Image.Image, 
    scan_result: Dict
) -> Optional[Image.Image]:
    """
    Extract only the document region bounded by the 4 marker corners.
    
    Uses marker corner positions to define document boundaries,
    then crops the image to that region.
    
    Returns: Cropped PIL Image or None if extraction fails
    """
    if not scan_result.get('success'):
        return None
    
    corners_data = scan_result.get('corners')
    detected_markers = scan_result.get('detected_markers', [])
    
    if not corners_data or not detected_markers:
        return None
    
    # Try to get all 4 corners
    corner_map = {}  # corner_name -> coordinates
    for idx, marker in enumerate(detected_markers):
        if idx >= len(corners_data):
            break
        corner_name = marker['corner']
        corners = corners_data[idx][0]  # shape (4, 2)
        corner_map[corner_name] = corners
    
    logger.debug(f"Found corners: {list(corner_map.keys())}")
    
    # If missing corners, try to infer
    if len(corner_map) < 4:
        inferred = ExamScanner.infer_missing_corner(detected_markers, corners_data)
        if inferred:
            corner_name, inferred_coords = inferred
            corner_map[corner_name] = inferred_coords[0]
            logger.info(f"Inferred missing corner: {corner_name}")
    
    if len(corner_map) < 4:
        logger.warning(f"Cannot extract: only {len(corner_map)}/4 corners available")
        return None
    
    # Get bounding box of all 4 corner markers
    all_points = np.vstack([corners for corners in corner_map.values()])  # shape (N, 2)
    x_min = int(np.min(all_points[:, 0]))
    y_min = int(np.min(all_points[:, 1]))
    x_max = int(np.max(all_points[:, 0]))
    y_max = int(np.max(all_points[:, 1]))
    
    # Add small margin
    margin = 10
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(pil_image.width, x_max + margin)
    y_max = min(pil_image.height, y_max + margin)
    
    logger.debug(f"Document bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    
    # Crop to document region
    cropped = pil_image.crop((x_min, y_min, x_max, y_max))
    logger.info(f"Extracted document region: {cropped.size}")
    
    return cropped


def scan_and_organize(images: List[Path]) -> Dict[int, Dict[int, Dict]]:
    """Scan all images and organize by exam/page."""
    organized = defaultdict(lambda: defaultdict(dict))
    
    for img_path in images:
        try:
            pil_img = Image.open(img_path).convert('RGB')
            img_cv = ExamScanner._pil_to_opencv(pil_img)
            
            result = ExamScanner.scan_page(img_cv)
            
            if result.get('success'):
                exam_id = result['exam_id']
                page_num = result['page_number']
                
                # Extract document region based on marker corners
                extracted = extract_document_region(pil_img, result)
                
                organized[exam_id][page_num] = {
                    'image_path': img_path,
                    'pil_image': pil_img,
                    'extracted_image': extracted or pil_img,  # fallback to full image
                    'scan_result': result
                }
                
                corners_info = f"{result.get('markers_found')} markers"
                logger.info(f"Scanned {img_path.name} → exam {exam_id}, page {page_num} ({corners_info})")
            else:
                logger.warning(f"Scan failed for {img_path.name}: {result.get('error')}")
        except Exception as e:
            logger.error(f"Error scanning {img_path}: {e}")
    
    return dict(organized)


def dewarp_page(
    extracted_img: Image.Image,
    full_img: Image.Image, 
    scan_result: Dict
) -> Optional[Image.Image]:
    """
    Dewarp the extracted document region using coordinate mapper.
    
    Args:
        extracted_img: The cropped document region
        full_img: Original full image (for homography calculation)
        scan_result: Scanner result with marker positions
    
    Returns: Dewarped PIL Image or None
    """
    try:
        # Use full image for homography calculation, but warp to extracted region size
        homography = CoordinateMapper.compute_homography_from_scan(scan_result)
        
        if homography is None:
            logger.warning("Could not compute homography")
            return None
        
        # Convert extracted to OpenCV
        extracted_cv = ExamScanner._pil_to_opencv(extracted_img)
        
        # Warp to straighten
        dewarped = cv2.warpPerspective(
            extracted_cv,
            np.linalg.inv(homography),
            (MarkerConfig.DOC_WIDTH, MarkerConfig.DOC_HEIGHT)
        )
        
        # Convert back to PIL
        dewarped_rgb = cv2.cvtColor(dewarped, cv2.COLOR_BGR2RGB)
        dewarped_pil = Image.fromarray(dewarped_rgb)
        
        logger.info(f"Dewarped to {dewarped_pil.size}")
        return dewarped_pil
    except Exception as e:
        logger.warning(f"Dewarping failed: {e}")
        return None


def save_exam_as_pdf(exam_id: int, pages_dict: Dict[int, Dict], output_dir: Path) -> Optional[Path]:
    """Save exam pages as corrected PDF."""
    if not pages_dict:
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"exam_{exam_id}_corrected.pdf"
    
    try:
        sorted_pages = sorted(pages_dict.items(), key=lambda x: x[0])
        images_for_pdf = []
        
        for page_num, page_data in sorted_pages:
            extracted_img = page_data['extracted_image']
            full_img = page_data['pil_image']
            scan_result = page_data['scan_result']
            
            # Try to dewarp the extracted region
            dewarped = dewarp_page(extracted_img, full_img, scan_result)
            
            if dewarped:
                images_for_pdf.append(dewarped)
                logger.debug(f"  page {page_num}: dewarped")
            else:
                # Fallback to extracted region
                images_for_pdf.append(extracted_img)
                logger.debug(f"  page {page_num}: extracted (no dewarp)")
        
        if images_for_pdf:
            # Ensure all RGB and save as PDF
            rgb_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images_for_pdf]
            rgb_images[0].save(pdf_path, save_all=True, append_images=rgb_images[1:])
            logger.info(f"Saved exam {exam_id} PDF: {pdf_path} ({len(rgb_images)} pages)")
            return pdf_path
        else:
            logger.warning(f"No images for exam {exam_id}")
            return None
    except Exception as e:
        logger.error(f"Failed to save PDF for exam {exam_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Scan markers → extract document → dewarp → save PDF"
    )
    parser.add_argument('--input-dir', '-i', default='Exams/real_exams')
    parser.add_argument('--output-dir', '-o', default='output_exams_corrected')
    parser.add_argument('--recursive', '-r', action='store_true', default=True)
    
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    logger.info(f"Pipeline: {input_dir} → extract document → dewarp → {output_dir}")
    
    # Step 1: Find images
    images = find_images(input_dir, args.recursive)
    logger.info(f"Found {len(images)} images")
    
    if not images:
        logger.warning("No images found")
        return
    
    # Step 2: Scan and organize
    organized = scan_and_organize(images)
    logger.info(f"Organized into {len(organized)} exams")
    
    # Step 3: Save corrected PDFs
    saved_pdfs = []
    for exam_id in sorted(organized.keys()):
        pages = organized[exam_id]
        pdf_path = save_exam_as_pdf(exam_id, pages, output_dir)
        if pdf_path:
            saved_pdfs.append(pdf_path)
    
    logger.info(f"Pipeline complete! Saved {len(saved_pdfs)} corrected PDFs")
    for pdf_path in saved_pdfs:
        print(f"  ✓ {pdf_path}")


if __name__ == '__main__':
    main()
