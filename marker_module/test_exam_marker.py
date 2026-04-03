"""Test script to add markers to exams in Exams/3ème année"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import fitz  # PyMuPDF
from marker_module.marker_generator import MarkerGenerator
from marker_module.marker_config import MarkerConfig
from logger_manager import LoggerManager


def test_add_markers_to_exams():
    """Add markers to exams in Exams/3ème année, excluding 'corr' files"""
    logger = LoggerManager.get_logger(__name__)
    
    # Create output directory
    output_dir = Path("ExamsMarked")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created: {output_dir}")
    
    # Initialize marker generator
    marker_gen = MarkerGenerator()
    
    # Base exam directory
    base_exam_dir = Path("Exams") / "2eme" 
    
    if not base_exam_dir.exists():
        logger.error(f"Exam directory not found: {base_exam_dir}")
        return
    
    exam_id = 0
    
    # Iterate through subject subdirectories
    for subject_dir in sorted(base_exam_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        subject_name = subject_dir.name
        subject_output_dir = output_dir / subject_name
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing subject: {subject_name}")
        
        # Process each PDF in the subject directory
        for pdf_file in sorted(subject_dir.glob("*.pdf")):
            # Skip correction files
            if pdf_file.name.startswith("corr"):
                logger.info(f"Skipping correction file: {pdf_file.name}")
                continue
            
            logger.info(f"Processing exam: {pdf_file.name}")
            
            try:
                # Convert PDF to images using PyMuPDF
                pdf_document = fitz.open(str(pdf_file))
                pages = []
                
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    # Render page to image with good resolution
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    # Convert to PIL Image
                    img = Image.frombytes(
                        "RGB", 
                        (pix.width, pix.height), 
                        pix.samples
                    )
                    pages.append(img)
                
                pdf_document.close()
                logger.info(f"  Converted {len(pages)} pages from PDF")
                
                # Add markers to all pages
                marked_pages = marker_gen.generate_marked_exam(
                    exam_id, pages
                )
                
                # Save marked pages as PDF
                output_pdf = (
                    subject_output_dir / 
                    f"marked_{pdf_file.stem}.pdf"
                )
                marked_pages[0].save(
                    output_pdf,
                    save_all=True,
                    append_images=marked_pages[1:]
                )
                
                logger.info(
                    f"  Saved marked exam to: {output_pdf}"
                )
                exam_id += 1
                
            except Exception as e:
                logger.error(
                    f"Error processing {pdf_file.name}: {str(e)}"
                )
                continue
    
    logger.info(f"Marker test completed. Output directory: {output_dir}")


if __name__ == "__main__":
    test_add_markers_to_exams()
