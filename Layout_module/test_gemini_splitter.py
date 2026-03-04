"""
test_gemini_splitter.py
-----------------------
Test script for GeminiImageSplitter.

Usage:
    python Layout_module/test_gemini_splitter.py --image "path/to/exam.jpg"
    python Layout_module/test_gemini_splitter.py --image "path/to/exam.jpg" --output "output_sections"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Layout_module.gemini_image_splitter import GeminiImageSplitter
from Layout_module.layout_config import LayoutConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Gemini Image Splitter")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input exam image",
    )
    parser.add_argument(
        "--output",
        default="data/Sections/test_output",
        help="Output directory for sections",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=100,
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
    print("GEMINI IMAGE SPLITTER TEST")
    print("=" * 72)
    print(f"Input image    : {image_path}")
    print(f"Output dir     : {output_dir}")
    print(f"Min section h  : {args.min_height}px")
    print(f"Target keywords: {LayoutConfig.KEY_WORDS}")
    print(f"Excluded words : {LayoutConfig.EXCLUDED_KEYWORDS}")
    print()

    # Validate config
    try:
        LayoutConfig.validate()
    except Exception as exc:
        print(f"[FAIL] Config validation failed: {exc}")
        return 1

    # Initialize splitter
    try:
        splitter = GeminiImageSplitter()
    except Exception as exc:
        print(f"[FAIL] Failed to initialize splitter: {exc}")
        return 1

    # ----------------------------------------------------------------
    # DEBUG: show raw Gemini keyword detections before splitting
    # ----------------------------------------------------------------
    print("-" * 72)
    print("DEBUG: Raw Gemini keyword detection")
    print("-" * 72)
    try:
        debug_image = PILImage.open(image_path).convert("RGB")
        raw_matches = splitter._gemini_extract(debug_image, debug_image.height)

        print(f"Gemini detected {len(raw_matches)} keyword(s):\n")
        for i, m in enumerate(raw_matches, 1):
            print(
                f"  [{i:02d}] keyword='{m.keyword}'"
                f"  est_y={m.y_position}"
                f"  text='{m.text[:70]}'"
            )
    except Exception as exc:
        print(f"[WARN] Debug extraction failed: {exc}")

    print()

    # ----------------------------------------------------------------
    # Split image
    # ----------------------------------------------------------------
    print("-" * 72)
    print("Splitting image by keywords...")
    print("-" * 72)
    try:
        sections = splitter.split_image_by_keywords(
            image=image_path,
            min_section_height=args.min_height,
        )
    except Exception as exc:
        print(f"[FAIL] Splitting failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    if not sections:
        print("[FAIL] No sections extracted")
        return 1

    print(f"\n[PASS] Extracted {len(sections)} section(s):\n")
    for section in sections:
        keyword_text = section.keyword_trigger or "NO_KEYWORD"
        height = section.y_end - section.y_start
        print(
            f"  Section {section.section_index:02d}: "
            f"keyword='{keyword_text}'"
            f"  y={section.y_start}-{section.y_end}"
            f"  height={height}px"
            f"  size={section.image.size}"
        )

    # Save sections
    print(f"\nSaving sections to {output_dir}...")
    try:
        saved_paths = splitter.save_sections(
            sections=sections,
            output_dir=output_dir,
            prefix="exam_section",
        )
    except Exception as exc:
        print(f"[FAIL] Save failed: {exc}")
        return 1

    print(f"\n[PASS] Saved {len(saved_paths)} files:\n")
    for path in saved_paths:
        print(f"  {path}")

    print("\n" + "=" * 72)
    print(f"SUMMARY")
    print("=" * 72)
    print(f"  Keywords detected by Gemini : {len(raw_matches)}")
    print(f"  Sections created            : {len(sections)}")
    print(f"  Output directory            : {output_dir}")
    print("=" * 72)
    print("FINAL RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())