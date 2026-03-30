import sys
import os
from pathlib import Path

# ensure project root is on sys.path so imports work when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from logger_manager import LoggerManager
import importlib.util
import types

# Create a lightweight package module to host Layout_module submodules
pkg_name = 'Layout_module'
pkg = types.ModuleType(pkg_name)
pkg.__path__ = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))]
sys.modules[pkg_name] = pkg

# Load layout_config first (required by image_cropping)
layout_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layout_config.py'))
spec_cfg = importlib.util.spec_from_file_location(f"{pkg_name}.layout_config", layout_config_path)
layout_config = importlib.util.module_from_spec(spec_cfg)
spec_cfg.loader.exec_module(layout_config)
sys.modules[f"{pkg_name}.layout_config"] = layout_config

# Now load image_cropping as part of the package
img_crop_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'image_cropping.py'))
spec = importlib.util.spec_from_file_location(f"{pkg_name}.image_cropping", img_crop_path)
image_cropping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_cropping)
sys.modules[f"{pkg_name}.image_cropping"] = image_cropping
ImageCropping = image_cropping.ImageCropping


def run_test():
    logger = LoggerManager.get_logger(__name__)

    pdf_path = Path("Exams/new_real_exams/exam_pdf.pdf") 
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        print(f"ERROR: PDF not found: {pdf_path}")
        return

    out_dir = Path("Exams/croped_corrections")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (pdf_path.stem + "_cropped.jpg")

    logger.info(f"Cropping PDF {pdf_path} -> {out_path}")
    cropper = ImageCropping()
    try:
        img = cropper.process_pdf_to_single_image(pdf_path, is_correction=True, save_output=True, output_path=str(out_path))
        logger.info(f"Saved cropped image to {out_path}")
        print(f"Saved cropped image to {out_path}")
    except Exception as e:
        logger.exception("Failed to process PDF")
        print(f"ERROR: {e}")


if __name__ == "__main__":
    run_test()
