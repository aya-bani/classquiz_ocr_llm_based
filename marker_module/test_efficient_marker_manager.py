"""
test_efficient_marker_manager.py
--------------------------------
Test runner for EfficientMarkerManager (new 2-pipeline manager).

Usage examples
--------------
Scan pipeline only:
  python marker_module/test_efficient_marker_manager.py --mode scan --submission-id 1 --scan-images "Exams/new_real_exams/ex5.jpg"

Mark pipeline only:
  python marker_module/test_efficient_marker_manager.py --mode mark --exam-id 1 --exam-pdf "path/to/blank_exam.pdf"

Both pipelines:
  python marker_module/test_efficient_marker_manager.py --mode both --exam-id 1 --exam-pdf "path/to/blank_exam.pdf" --submission-id 1 --scan-images "Exams/new_real_exams/ex5.jpg"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from marker_module.efficient_marker_manager import EfficientMarkerManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test EfficientMarkerManager")
    parser.add_argument("--mode", choices=["mark", "scan", "both"], default="scan")
    parser.add_argument("--exam-id", type=int, default=0)
    parser.add_argument("--exam-pdf", type=str, default="")
    parser.add_argument("--submission-id", type=int, default=0)
    parser.add_argument("--scan-images", nargs="*", default=[])
    return parser.parse_args()


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _check_mark_result(result: dict) -> bool:
    required = {"exam_id", "num_pages", "output_path"}
    if not required.issubset(result.keys()):
        print(f"[FAIL] mark result missing keys: {required - set(result.keys())}")
        return False
    output_path = Path(result["output_path"])
    if not output_path.exists():
        print(f"[FAIL] mark output missing: {output_path}")
        return False
    if int(result["num_pages"]) <= 0:
        print("[FAIL] mark num_pages <= 0")
        return False
    print(f"[PASS] mark output: {output_path}")
    return True


def _check_scan_results(results: list) -> bool:
    if not isinstance(results, list) or len(results) == 0:
        print("[FAIL] scan returned empty results")
        return False

    ok = True
    for idx, item in enumerate(results, start=1):
        required = {"exam_id", "num_pages", "output_path", "failed_pages"}
        if not required.issubset(item.keys()):
            print(f"[FAIL] scan result#{idx} missing keys: {required - set(item.keys())}")
            ok = False
            continue

        output_path = Path(item["output_path"])
        if not output_path.exists():
            print(f"[FAIL] scan result#{idx} output missing: {output_path}")
            ok = False
            continue

        if int(item["num_pages"]) <= 0:
            print(f"[FAIL] scan result#{idx} num_pages <= 0")
            ok = False
            continue

        print(
            f"[PASS] scan result#{idx}: exam_id={item['exam_id']} "
            f"pages={item['num_pages']} failed_pages={item['failed_pages']} "
            f"output={output_path}"
        )

    return ok


def run_mark(manager: EfficientMarkerManager, exam_id: int, exam_pdf: str) -> bool:
    if not exam_pdf:
        print("[FAIL] --exam-pdf required for mode=mark/both")
        return False

    exam_pdf_path = _resolve(exam_pdf)
    if not exam_pdf_path.exists():
        print(f"[FAIL] exam pdf not found: {exam_pdf_path}")
        return False

    print("=" * 72)
    print("PIPELINE 1 TEST: mark_exam")
    print("=" * 72)
    print(f"Exam ID : {exam_id}")
    print(f"Input   : {exam_pdf_path}")

    try:
        result = manager.mark_exam(exam_id=exam_id, exam_path=exam_pdf_path)
    except Exception as exc:
        print(f"[FAIL] mark_exam exception: {exc}")
        return False

    print(f"Raw result: {result}")
    return _check_mark_result(result)


def run_scan(manager: EfficientMarkerManager, submission_id: int, scan_images: list[str]) -> bool:
    if not scan_images:
        print("[FAIL] --scan-images required for mode=scan/both")
        return False

    scan_paths = [str(_resolve(item)) for item in scan_images]

    print("=" * 72)
    print("PIPELINE 2 TEST: scan_submission")
    print("=" * 72)
    print(f"Submission ID : {submission_id}")
    print(f"Input images  : {scan_paths}")

    try:
        results = manager.scan_submission(
            submission_id=submission_id,
            submitted_images=scan_paths,
        )
    except Exception as exc:
        print(f"[FAIL] scan_submission exception: {exc}")
        return False

    print(f"Raw results: {results}")
    return _check_scan_results(results)


def main() -> int:
    args = parse_args()
    manager = EfficientMarkerManager()

    overall_ok = True

    if args.mode in {"mark", "both"}:
        overall_ok = run_mark(manager, args.exam_id, args.exam_pdf) and overall_ok

    if args.mode in {"scan", "both"}:
        overall_ok = run_scan(manager, args.submission_id, args.scan_images) and overall_ok

    print("\n" + "-" * 72)
    print("FINAL RESULT: PASS" if overall_ok else "FINAL RESULT: FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
