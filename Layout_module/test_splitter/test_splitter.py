from pathlib import Path
from PIL import Image
import sys
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Layout_module.image_splitter import ImageSplitter


def main():
    project_root = Path(__file__).resolve().parents[2]

    # ✅ input from terminal OR default
    img_file = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else project_root / "Exams" / "croped_corrections" / "exam_pdf_cropped.jpg"
    )

    # make absolute if relative
    img_file = img_file if img_file.is_absolute() else project_root / img_file

    if not img_file.exists():
        raise FileNotFoundError(f"Image not found: {img_file}")

    image = Image.open(img_file)

    splitter = ImageSplitter()
    result = splitter.split_and_save(image, exam_id=1)

    output_dir = img_file.parent / "sections_output"
    output_dir.mkdir(exist_ok=True)

    src = Path(result["sections_dir"])
    dst = output_dir / src.name

    if dst.exists():
        shutil.rmtree(dst)

    shutil.move(str(src), str(dst))

    print(f"Sections saved to: {dst}")


if __name__ == "__main__":
    main()