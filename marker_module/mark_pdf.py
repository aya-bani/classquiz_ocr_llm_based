import sys
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from marker_module.marker_generator import MarkerGenerator

# ensure package imports work when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_pdf_to_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """Load PDF pages using PyMuPDF and return a list of PIL Images."""
    doc = fitz.open(pdf_path)
    images: list[Image.Image] = []
    zoom = dpi / 72  # 72 dpi is the default PDF resolution
    for page in doc:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


def mark_pdf(pdf_path: str, exam_id: int):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Processing {pdf_file} (exam_id={exam_id})")
    pages = convert_pdf_to_images(pdf_path)
    print(f"  converted {len(pages)} page(s)")

    gen = MarkerGenerator()
    marked_pages = gen.generate_marked_exam(exam_id, pages)

    output_dir = Path(__file__).parent / "marked_output"
    output_dir.mkdir(exist_ok=True)
    output_pdf = output_dir / f"MARKED_{pdf_file.stem}_exam{exam_id}.pdf"

    marked_pages[0].save(
        output_pdf,
        save_all=True,
        append_images=marked_pages[1:],
        format="PDF",
        resolution=100.0,
    )
    print(f"Saved marked exam to {output_pdf}")
    return output_pdf


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mark_pdf.py <pdf_path> <exam_id>")
    else:
        path = sys.argv[1]
        try:
            eid = int(sys.argv[2])
        except ValueError:
            print("Exam ID must be an integer")
            sys.exit(1)
        mark_pdf(path, eid)
