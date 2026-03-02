import os
import sys
from pathlib import Path
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Layout_module.imagesplitter import ImageSplitter


def main():
    splitter = ImageSplitter()

    result = splitter.split_and_save(
        str(Path(__file__).parent / "corr3matht1d2_cropped.jpg"),
        exam_id=1,
        return_sections=True
    )

    if not result["success"]:
        print("Error:", result["error"])
        return

    print(f"\nCreated {result['num_sections']} sections")

    # DISPLAY EACH SECTION
    for i, section in enumerate(result["sections"], 1):
        window_name = f"Section {i}"
        cv2.imshow(window_name, section)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
