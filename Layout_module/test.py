from layout_manager import LayoutManager
from pdf2image import convert_from_path
pdf_path = "corrected.pdf"
pages = convert_from_path(pdf_path, dpi=300)

manager = LayoutManager()
section_paths = manager.process_correction(pages, exam_id=123)
print("Sections saved at:", section_paths)