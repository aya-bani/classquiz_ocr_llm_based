from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Layout_module.image_splitter import ImageSplitter
import shutil


def main():
    # input image from the Exams folder
    project_root = Path(__file__).resolve().parents[2]
    img_file = project_root / "Exams" / "croped_corrections" / "exam_pdf_cropped.jpg"
    if not img_file.exists():
        raise FileNotFoundError(f"Expected test image not found: {img_file}")

    splitter = ImageSplitter()
    image = Image.open(img_file)

    # run the splitter (use a dummy exam_id)
    result = splitter.split_and_save(image, exam_id=1)

    # move the generated directory to the requested output location
    dest_root = project_root / "Exams" / "google_vision" / "math" / "splited images into sections"
    dest_root.mkdir(parents=True, exist_ok=True)

    src_dir = Path(result["sections_dir"])
    dest_dir = dest_root / src_dir.name

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.move(str(src_dir), str(dest_dir))

    print(f"Sections written to {dest_dir}")


if __name__ == "__main__":
    main()
