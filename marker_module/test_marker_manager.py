"""
test_marker_manager.py
----------------------
CLI smoke/integration tester for MarkerManager.

Usage examples
--------------
1) Test marking only:
   python marker_module/test_marker_manager.py --mode mark --exam-id 1 --exam-pdf "Exams/3ème année/math/exam.pdf"

2) Test scanning only from images:
   python marker_module/test_marker_manager.py --mode scan --submission-id 1 --scan-images "Exams/new_real_exams/ex5.jpg"

3) Test scan from multiple images:
   python marker_module/test_marker_manager.py --mode scan --submission-id 1 --scan-images "img1.jpg" "img2.jpg"

4) Full flow (mark then scan):
   python marker_module/test_marker_manager.py --mode both --exam-id 1 --exam-pdf "Exams/3ème année/math/exam.pdf" --submission-id 1 --scan-images "Exams/new_real_exams/ex5.jpg"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from marker_module.marker_manager import MarkerManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MarkerManager methods")
    parser.add_argument(
        "--mode",
        choices=["mark", "scan", "both"],
        default="scan",
        help="Which MarkerManager method(s) to test",
    )
    parser.add_argument("--exam-id", type=int, default=0, help="Exam ID for mark_exam")
    parser.add_argument(
        "--exam-pdf",
        type=str,
        default="",
        help="Path to input PDF for mark_exam",
    )
    parser.add_argument(
        "--submission-id",
        type=int,
        default=0,
        help="Submission ID for scan_submission",
    )
    parser.add_argument(
        "--scan-images",
        nargs="*",
        default=[],
        help="One or more scanned image paths for scan_submission",
    )
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_scan_images(image_paths: List[str]) -> List[Image.Image]:
    pages: List[Image.Image] = []
    for item in image_paths:
        p = _resolve_path(item)
        if not p.exists():
            raise FileNotFoundError(f"Scan image not found: {p}")
        pages.append(Image.open(p).convert("RGB"))
    if not pages:
        raise ValueError("No scan images provided. Use --scan-images <img1> [img2 ...]")
    return pages


def _print_header(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def _validate_mark_result(result: dict) -> bool:
    required_keys = {"exam_id", "num_pages", "output_path"}
    ok = required_keys.issubset(result.keys())
    if not ok:
        print(f"[FAIL] mark_exam result missing keys: {required_keys - set(result.keys())}")
        return False

    output_path = Path(result["output_path"])
    if not output_path.exists():
        print(f"[FAIL] mark_exam output PDF does not exist: {output_path}")
        return False

    if int(result["num_pages"]) <= 0:
        print("[FAIL] mark_exam num_pages <= 0")
        return False

    print(f"[PASS] mark_exam created: {output_path}")
    return True


def _validate_scan_results(results: list) -> bool:
    if not isinstance(results, list) or len(results) == 0:
        print("[FAIL] scan_submission returned empty/non-list results")
        return False

    all_ok = True
    for i, item in enumerate(results, start=1):
        required_keys = {"exam_id", "num_pages", "output_path"}
        if not required_keys.issubset(item.keys()):
            print(f"[FAIL] result #{i} missing keys: {required_keys - set(item.keys())}")
            all_ok = False
            continue

        output_path = Path(item["output_path"])
        if not output_path.exists():
            print(f"[FAIL] result #{i} output PDF does not exist: {output_path}")
            all_ok = False
            continue

        if int(item["num_pages"]) <= 0:
            print(f"[FAIL] result #{i} num_pages <= 0")
            all_ok = False
            continue

        print(
            f"[PASS] scan_submission result #{i}: exam_id={item['exam_id']} "
            f"pages={item['num_pages']} output={output_path}"
        )

    return all_ok


def _run_mark(manager: MarkerManager, exam_id: int, exam_pdf: str) -> bool:
    if not exam_pdf:
        print("[FAIL] --exam-pdf is required for mode=mark or mode=both")
        return False

    exam_pdf_path = _resolve_path(exam_pdf)
    if not exam_pdf_path.exists():
        print(f"[FAIL] Exam PDF not found: {exam_pdf_path}")
        return False

    _print_header("MARKER MANAGER TEST - mark_exam")
    print(f"Exam ID : {exam_id}")
    print(f"Input   : {exam_pdf_path}")

    try:
        result = manager.mark_exam(exam_id=exam_id, exam_path=exam_pdf_path)
    except Exception as exc:
        print(f"[FAIL] mark_exam raised exception: {exc}")
        return False

    print(f"Raw result: {result}")
    return _validate_mark_result(result)


def _run_scan(manager: MarkerManager, submission_id: int, scan_images: List[str]) -> bool:
    _print_header("MARKER MANAGER TEST - scan_submission")
    print(f"Submission ID : {submission_id}")

    try:
        pages = _load_scan_images(scan_images)
    except Exception as exc:
        print(f"[FAIL] Unable to load scan images: {exc}")
        return False

    print(f"Pages loaded  : {len(pages)}")

    try:
        results = manager.scan_submission(submission_id=submission_id, pages=pages)
    except Exception as exc:
        print(f"[FAIL] scan_submission raised exception: {exc}")
        return False

    print(f"Raw results: {results}")
    return _validate_scan_results(results)


def main() -> int:
    args = _parse_args()
    manager = MarkerManager()

    overall_ok = True

    if args.mode in {"mark", "both"}:
        ok = _run_mark(manager, args.exam_id, args.exam_pdf)
        overall_ok = overall_ok and ok

    if args.mode in {"scan", "both"}:
        ok = _run_scan(manager, args.submission_id, args.scan_images)
        overall_ok = overall_ok and ok

    print("\n" + "-" * 72)
    if overall_ok:
        print("FINAL RESULT: PASS")
        return 0

    print("FINAL RESULT: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
