"""
test_openai_splitter.py
-----------------------
Usage:
    python Layout_module/test_openai_splitter.py --image "path/to/exam.jpg"
    python Layout_module/test_openai_splitter.py --image "path/to/exam.jpg" --output "output_sections"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Layout_module.openai_image_splitter import OpenAIImageSplitter
from Layout_module.layout_config import LayoutConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test OpenAI Image Splitter")
    parser.add_argument("--image",      required=True)
    parser.add_argument("--output",     default="data/Sections/test_output_openai")
    parser.add_argument("--min-height", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    if not image_path.exists():
        print(f"[FAIL] Image not found: {image_path}")
        return 1

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    print("=" * 72)
    print("OPENAI IMAGE SPLITTER TEST")
    print("=" * 72)
    print(f"Input image    : {image_path}")
    print(f"Output dir     : {output_dir}")
    print(f"Target keywords: {LayoutConfig.KEY_WORDS}")
    print(f"Excluded words : {LayoutConfig.EXCLUDED_KEYWORDS}")
    print(f"OpenAI model   : {LayoutConfig.OPENAI_MODEL_NAME}")
    print()

    try:
        LayoutConfig.validate()
    except Exception as exc:
        print(f"[FAIL] Config validation failed: {exc}")
        return 1

    try:
        splitter = OpenAIImageSplitter()
    except Exception as exc:
        print(f"[FAIL] Init failed: {exc}")
        return 1

    image = PILImage.open(image_path).convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: detect ───────────────────────────────────────────────
    print("-" * 72)
    print("STEP 1: detect_section_lines()")
    print("-" * 72)
    try:
        section_coords = splitter.detect_section_lines(image)
        print(f"\nDetected {len(section_coords)} keyword(s):\n")
        for i, (x_min, y_min, x_max, y_max) in enumerate(section_coords, 1):
            kw = splitter._last_detected_keywords[i - 1].keyword
            tx = splitter._last_detected_keywords[i - 1].text[:60]
            print(f"  [{i:02d}] keyword='{kw}'  y={y_min}  text='{tx}'")
    except Exception as exc:
        print(f"[FAIL] detect_section_lines: {exc}")
        import traceback; traceback.print_exc()
        return 1

    # ── Step 2: highlight ────────────────────────────────────────────
    print()
    print("-" * 72)
    print("STEP 2: highlight_keywords()  ->  highlighted.jpg")
    print("-" * 72)
    try:
        highlighted    = splitter.highlight_keywords(image, section_coords)
        highlight_path = output_dir / "highlighted.jpg"
        highlighted.save(highlight_path, "JPEG", quality=95)
        print(f"[PASS] Saved: {highlight_path}")
    except Exception as exc:
        print(f"[WARN] Highlight failed: {exc}")

    # ── Step 3: split ────────────────────────────────────────────────
    print()
    print("-" * 72)
    print("STEP 3: split_image()")
    print("-" * 72)
    try:
        sections = splitter.split_image(image)
    except Exception as exc:
        print(f"[FAIL] split_image: {exc}")
        import traceback; traceback.print_exc()
        return 1

    print(f"\n[PASS] {len(sections)} section(s):\n")
    for i, s in enumerate(sections):
        print(f"  Section {i:02d}:  size={s.size}")

    # ── Step 4: save ─────────────────────────────────────────────────
    print(f"\nSaving to {output_dir} ...")
    for i, section in enumerate(sections):
        fp = output_dir / f"exam_section_{i:02d}.jpg"
        section.save(fp, "JPEG", quality=95)
        print(f"  {fp}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Keywords detected : {len(section_coords)}")
    print(f"  Sections created  : {len(sections)}")
    print(f"  Highlighted image : {output_dir / 'highlighted.jpg'}")
    print(f"  Output directory  : {output_dir}")
    print("=" * 72)
    print("FINAL RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())