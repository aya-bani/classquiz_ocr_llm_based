"""
testsplitterpaddleocr.py
========================
Test / demo script for image_splitter_paddleocr.ImageSplitter.

• Does NOT use cv2.imshow (crashes on headless / opencv-headless installs).
• Saves each section as a JPEG and writes a summary PNG grid via matplotlib.
• Automatically runs diagnose_image() when no keywords are found.
"""

import os
# Disable oneDNN before any Paddle imports
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")           # headless-safe; change to "TkAgg" if you have a display
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────── #
# Adjust if your project layout differs
sys.path.append(str(Path(__file__).parent.parent.parent))
from Layout_module.image_splitter_paddleocr import ImageSplitter

# ── Config ────────────────────────────────────────────────────────────── #
IMAGE_PATH = str(Path(__file__).parent / "corr3matht1d2_cropped.jpg")
EXAM_ID    = 2
OUTPUT_DIR = str(Path(__file__).parent / "debug_sections_paddle")


# ===========================================================================
# Helpers
# ===========================================================================

def save_and_display(sections: list, output_dir: str) -> None:
    """Save each section JPEG + write a summary PNG grid."""
    os.makedirs(output_dir, exist_ok=True)
    saved: list[str] = []

    for i, sec in enumerate(sections, 1):
        p = os.path.join(output_dir, f"section_{i:02d}.jpg")
        cv2.imwrite(p, sec)
        saved.append(p)
        print(f"  Section {i:02d}  →  {p}  ({sec.shape[1]}×{sec.shape[0]}px)")

    if not saved:
        print("  (no sections to display)")
        return

    # Build summary grid
    n    = len(saved)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 8 * rows),
                             squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for ax, p in zip(flat_axes, saved):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(Path(p).name, fontsize=9)
        ax.axis("off")
    for ax in flat_axes[n:]:
        ax.axis("off")

    summary = os.path.join(output_dir, "summary.png")
    plt.tight_layout()
    plt.savefig(summary, dpi=100)
    plt.close()
    print(f"\n  Summary grid  →  {summary}")


def print_diagnosis_hint() -> None:
    print("\n" + "=" * 65)
    print("HOW TO READ THE DIAGNOSIS TABLE")
    print("=" * 65)
    print("  ✓ MATCH  → keyword accepted — section starts at this Y")
    print("  EXCLUDED → word is in EXCLUDED_KEYWORDS (e.g. تسند)")
    print("  —        → word is not a keyword")
    print()
    print("If your keyword shows '—':")
    print("  1. Check the NORMALISED column — is it exactly 'تعليمة' or 'سند'?")
    print("  2. If it looks right but still '—', it may have invisible unicode.")
    print("     Fix: add the raw OCR form to ALLOWED_KEYWORDS in")
    print("     image_splitter_paddleocr.py and re-run.")
    print("  3. If it's garbled (e.g. 'تعلىمة' vs 'تعليمة'), lower")
    print("     FUZZY_FALLBACK_THRESHOLD from 95 → 85 in ImageSplitter.")
    print("=" * 65)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    splitter = ImageSplitter(
        output_dir="data/Sections/exams/paddleocr_output",
        lang="ar",
    )

    # ── Run split ─────────────────────────────────────────────────────── #
    result = splitter.split_and_save(
        IMAGE_PATH,
        exam_id=EXAM_ID,
        return_sections=True,
    )

    if not result["success"]:
        print(f"\nERROR: {result['error']}")
        return

    sections = result["sections"]
    print(f"\nProduced {result['num_sections']} section(s)")
    print(f"Saved to : {result['saved_paths']}")

    # ── Auto-diagnose when only the fallback whole-image was returned ──── #
    if result["num_sections"] == 1:
        print("\n⚠  Only 1 section — no keywords were detected.")
        print("   Running diagnose_image() …\n")
        splitter.diagnose_image(IMAGE_PATH)
        print_diagnosis_hint()

    # ── Save + display ────────────────────────────────────────────────── #
    save_and_display(sections, OUTPUT_DIR)


if __name__ == "__main__":
    main()