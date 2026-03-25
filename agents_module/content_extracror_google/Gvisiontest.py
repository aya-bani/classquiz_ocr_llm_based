"""
run_extraction.py
─────────────────────────────────────────────────────────────────────────────
Google Cloud Vision — Arabic handwritten answer extractor.

HOW TO USE
──────────
1. Edit the CONFIG block below (folder path, section type, credentials, etc.)
2. Run:
       python agents_module/content_extracror_google/run_extraction.py

Output is saved to:
       agents_module/content_extractor/results/<folder_name>.json
(or to OUTPUT_PATH if you set it explicitly)
─────────────────────────────────────────────────────────────────────────────
"""


from __future__ import annotations

import sys
from pathlib import Path

# Ensure the script's directory is in sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Make sibling modules importable when run as a script ─────────────────────

# Always use direct imports since all files are in the same directory and sys.path is set
try:
    from extractor import ArabicHandwrittenExtractor
    from utils import confidence_emoji, confidence_label
except ImportError as e:
    print(f"[ERROR] Could not import extractor modules: {e}")
    print("  Make sure extractor.py, utils.py, and prompt.py are in the same folder.")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# ✏️  CONFIG — edit this block, then run the file
# ═════════════════════════════════════════════════════════════════════════════

# Folder containing the exam section images to process
INPUT_FOLDER = "Exams/google_vision/math/splited images into sections/exam_1"

# Where to save the output JSON.
# Set to None to use the default:
#   agents_module/content_extractor/results/<folder_name>.json
OUTPUT_PATH = None

# Google Cloud credentials are configured in extractor.py (GOOGLE_CREDENTIALS_PATH).
# Edit that variable there — no credential config needed here.

# Question type applied to all images in INPUT_FOLDER.
# Common values: FILL_BLANK, CALCULATION, SHORT_ANSWER, TRUE_FALSE,
#                MULTIPLE_CHOICE, RELATING, WRITING, unknown
SECTION_TYPE = "unknown"

# The printed question text (optional).
# Used by the LLM post-processing step to separate printed from handwritten.
# Leave as "" if not needed or if LLM post-processing is disabled.
QUESTION_TEXT = ""

# Minimum OCR block confidence (0.0–1.0).
# Blocks below this score are discarded before reassembly.
MIN_CONFIDENCE = 0.40

# Print detailed per-image output to the console.
VERBOSE = True

# ═════════════════════════════════════════════════════════════════════════════
# (no edits needed below this line)
# ═════════════════════════════════════════════════════════════════════════════


def _resolve_output(folder: Path) -> Path:
    """Return the output JSON path (explicit config or auto-derived)."""
    if OUTPUT_PATH:
        return Path(OUTPUT_PATH)

    # Walk up from this file to find the project root (dir containing agents_module/)
    here = Path(__file__).resolve().parent
    project_root = Path.cwd()
    for parent in [here, *here.parents]:
        if (parent / "agents_module").exists():
            project_root = parent
            break

    return (
        project_root
        / "agents_module"
        / "content_extractor"
        / "results"
        / f"{folder.name}.json"
    )


def _print_summary(results: list, output_path: Path) -> None:
    """Print a per-image summary table to the console."""
    total    = len(results)
    answered = sum(1 for r in results if r.student_answer)
    empty    = total - answered

    W = 60
    print(f"\n  {'═' * W}")
    print(f"  {'EXTRACTION SUMMARY':^{W}}")
    print(f"  {'═' * W}")
    print(f"  Total images   : {total}")
    print(f"  With answers   : {answered}")
    print(f"  Empty / blank  : {empty}")
    print(f"  Output file    : {output_path}")
    print(f"  {'─' * W}")
    print(f"  {'CONF':<8}  {'IMAGE':<28}  ANSWER")
    print(f"  {'─' * W}")

    for r in results:
        emoji   = confidence_emoji(r.confidence)
        label   = confidence_label(r.confidence)
        name    = Path(r.image_path).name[:27]
        answer  = r.student_answer or "(no answer)"
        display = answer[:40] + "…" if len(answer) > 40 else answer
        print(f"  {emoji} {label:<7}  {name:<28}  {display!r}")

    print(f"  {'═' * W}\n")


def main() -> None:
    folder = Path(INPUT_FOLDER)

    if not folder.exists():
        print(f"[ERROR] Input folder not found: {folder}")
        print(f"  Edit INPUT_FOLDER at the top of this file and try again.")
        sys.exit(1)

    output_path = _resolve_output(folder)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input  : {folder}")
    print(f"[INFO] Output : {output_path}")
    print(f"[INFO] Type   : {SECTION_TYPE}")

    extractor = ArabicHandwrittenExtractor(
        min_block_confidence = MIN_CONFIDENCE,
        verbose              = VERBOSE,
    )

    results = extractor.extract_folder(
        folder_path   = folder,
        section_type  = SECTION_TYPE,
        question_text = QUESTION_TEXT,
        output_path   = output_path,
    )

    _print_summary(results, output_path)


if __name__ == "__main__":
    main()