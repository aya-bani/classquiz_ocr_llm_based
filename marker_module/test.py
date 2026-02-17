from .marker_manager import MarkerManager
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image

'''def _load_pdf(pdf_path:Path) -> tuple:
    """Load PDF file and return list of pages"""
    pages = convert_from_path(pdf_path, dpi=300)
    return pages


pages = _load_pdf("blank.pdf")

manager = MarkerManager()

result = manager.mark_exam(
    exam_id=0,
    pages=pages
)
print(result)'''
script_dir = Path(__file__).parent.absolute()

img1 = Image.open(script_dir / "0.jpg")
# img2 = Image.open("2.jpg")
pages = [img1]
manager = MarkerManager()
result = manager.scan_submission(
    pages=pages,
    submission_id=0
)
print(result)
