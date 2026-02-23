import sys
import argparse
from pathlib import Path
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import tempfile

from marker_module.marker_generator import MarkerGenerator
from marker_module.marker_config import MarkerConfig
from logger_manager import LoggerManager


def add_markers_to_pdf(input_pdf_path: str, exam_id: int, output_folder: str = None) -> bool:
    """Generate marked PDF by adding ArUco markers to a blank exam PDF."""
    
    logger = LoggerManager.get_logger(__name__)
    
    try:
        input_path = Path(input_pdf_path)
        
        # Validate input file exists
        if not input_path.exists():
            logger.error(f"Input PDF not found: {input_pdf_path}")
            raise FileNotFoundError(f"Input PDF not found: {input_pdf_path}")
        
        # Default output folder
        if output_folder is None:
            output_folder = "output_exams"
        
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output PDF path
        output_pdf_path = output_dir / f"{input_path.stem}_marked.pdf"
        
        logger.info(f"Opening PDF: {input_pdf_path}")
        
        # Open PDF and extract pages as images
        pdf_document = fitz.open(input_pdf_path)
        pdf_pages = []
        
        logger.info(f"Converting {len(pdf_document)} PDF pages to images")
        
        for page_num in range(len(pdf_document)):
            # Render page as image
            page = pdf_document[page_num]
            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert pixmap to PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(fitz.io.BytesIO(img_data))
            pdf_pages.append(img)
            logger.debug(f"Converted page {page_num + 1}: {img.size}")
        
        pdf_document.close()
        logger.info(f"Extracted {len(pdf_pages)} pages from PDF")
        
        # Generate markers for each page
        logger.info(f"Generating markers for exam_id={exam_id}")
        marker_gen = MarkerGenerator()
        marked_pages = marker_gen.generate_marked_exam(exam_id, pdf_pages)
        logger.info(f"Generated markers for {len(marked_pages)} pages")
        
        # Create output PDF with marked pages
        logger.info(f"Creating marked PDF: {output_pdf_path}")
        output_pdf = fitz.open()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for page_idx, marked_page in enumerate(marked_pages):
                # Save marked page as temporary PNG
                temp_png = f"{tmpdir}/page_{page_idx}.png"
                marked_page.save(temp_png, format='PNG')
                
                # Get image dimensions
                width, height = marked_page.size
                
                # Create new PDF page with same dimensions
                page = output_pdf.new_page(width=width, height=height)
                
                # Insert image into page
                page.insert_image(
                    page.rect, 
                    filename=temp_png, 
                    keep_proportion=False
                )
                
                logger.debug(f"Added marked page {page_idx + 1} to PDF")
        
        # Save output PDF
        output_pdf.save(str(output_pdf_path))
        output_pdf.close()
        
        logger.info(f"Marked PDF saved: {output_pdf_path}")
        print(f"\n{'='*60}")
        print(f"SUCCESS: Marked PDF generated")
        print(f"Input:  {input_pdf_path}")
        print(f"Output: {output_pdf_path}")
        print(f"Exam ID: {exam_id}")
        print(f"Pages: {len(marked_pages)}")
        print(f"Marker Size: {MarkerConfig.MARKER_SIZE}px")
        print(f"Margin: {MarkerConfig.MARGIN}px")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating marked PDF: {str(e)}", exc_info=True)
        print(f"\nERROR: {str(e)}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add ArUco markers to blank exam PDF"
    )
    parser.add_argument("input_pdf", help="Path to input blank PDF")
    parser.add_argument("exam_id", type=int, help="Exam ID for marker generation")
    parser.add_argument(
        "-o", "--output",
        help="Output folder (default: output_exams)",
        default="output_exams"
    )
    
    args = parser.parse_args()
    
    success = add_markers_to_pdf(args.input_pdf, args.exam_id, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
