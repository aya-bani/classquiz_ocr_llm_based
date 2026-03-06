"""
test_openai_splitter.py
-----------------------
Test script for OpenAIImageSplitter.

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
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input exam image",
    )
    parser.add_argument(
        "--output",
        default="data/Sections/test_output_openai",
        help="Output directory for sections",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=1,
        help="Minimum section height in pixels",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve paths
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
    print(f"Min section h  : {args.min_height}px")
    print(f"Target keywords: {LayoutConfig.KEY_WORDS}")
    print(f"Excluded words : {LayoutConfig.EXCLUDED_KEYWORDS}")
    print(f"OpenAI model   : {LayoutConfig.OPENAI_MODEL_NAME}")
    print()

    # Validate config
    try:
        LayoutConfig.validate()
    except Exception as exc:
        print(f"[FAIL] Config validation failed: {exc}")
        return 1

    # Initialize splitter
    try:
        splitter = OpenAIImageSplitter()
    except Exception as exc:
        print(f"[FAIL] Failed to initialize splitter: {exc}")
        return 1

    image = PILImage.open(image_path).convert("RGB")

    # ----------------------------------------------------------------
    # DEBUG: detect_section_lines — shows what OpenAI found per chunk
    # ----------------------------------------------------------------
    print("-" * 72)
    print("DEBUG: detect_section_lines()")
    print("-" * 72)
    section_coords = []
    try:
        section_coords = splitter.detect_section_lines(image)
        print(f"\nDetected {len(section_coords)} section line(s):\n")
        for i, (x_min, y_min, x_max, y_max) in enumerate(section_coords, 1):
            print(f"  [{i:02d}]  y_min={y_min:5d}  y_max={y_max:5d}  x=({x_min}–{x_max})")
    except Exception as exc:
        print(f"[WARN] detect_section_lines failed: {exc}")

    print()

    # ----------------------------------------------------------------
    # Split image
    # ----------------------------------------------------------------
    print("-" * 72)
    print("Splitting image ...")
    print("-" * 72)
    try:
        sections = splitter.split_image(image)
    except Exception as exc:
        print(f"[FAIL] Splitting failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    if not sections:
        print("[FAIL] No sections extracted")
        return 1

    print(f"\n[PASS] Extracted {len(sections)} section(s):\n")
    for i, section in enumerate(sections):
        print(f"  Section {i:02d}:  size={section.size}")

    # ----------------------------------------------------------------
    # Save sections
    # ----------------------------------------------------------------
    print(f"\nSaving sections to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    try:
        for i, section in enumerate(sections):
            fp = output_dir / f"exam_section_{i:02d}.jpg"
            section.save(fp, "JPEG", quality=95)
            saved_paths.append(fp)
    except Exception as exc:
        print(f"[FAIL] Save failed: {exc}")
        return 1

    print(f"\n[PASS] Saved {len(saved_paths)} file(s):\n")
    for path in saved_paths:
        print(f"  {path}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Section lines detected : {len(section_coords)}")
    print(f"  Sections created       : {len(sections)}")
    print(f"  Output directory       : {output_dir}")
    print("=" * 72)
    print("FINAL RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())