from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow direct execution: python agents_module\test_question_extractor_real.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents_module.question_extractor import QuestionExtractor
from agents_module.agents_config import AgentsConfig


def _assert_result_schema(result: Dict[str, Any], context: str) -> None:
    if not isinstance(result, dict):
        raise AssertionError(f"{context}: result must be a dict")

    required = {"question_type", "confidence", "meta_data"}
    missing = required - set(result.keys())
    if missing:
        raise AssertionError(f"{context}: missing keys: {sorted(missing)}")

    if not isinstance(result["question_type"], str) or not result["question_type"].strip():
        raise AssertionError(f"{context}: invalid question_type")

    confidence = result["confidence"]
    if not isinstance(confidence, (int, float)):
        raise AssertionError(f"{context}: confidence must be numeric")
    if not (0.0 <= float(confidence) <= 1.0):
        raise AssertionError(f"{context}: confidence must be in [0.0, 1.0]")

    meta = result["meta_data"]
    if not isinstance(meta, dict):
        raise AssertionError(f"{context}: meta_data must be a dict")
    if "image_path" not in meta or "image_name" not in meta:
        raise AssertionError(f"{context}: meta_data must include image_path and image_name")

    # Accept either successful extraction or explicit error payload.
    if "error" not in result and "content" not in result:
        raise AssertionError(f"{context}: expected either content or error")


def run_real_image_test(image_path: Path, is_submission: bool) -> Dict[str, Any]:
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    extractor = QuestionExtractor()
    try:
        result = extractor.process_image(image_path, is_submission=is_submission)
        _assert_result_schema(result, context=f"process_image({image_path.name})")
        return result
    finally:
        extractor.close()


def run_real_batch_test(
    folder_path: Path,
    is_submission: bool,
    save_results: bool,
    output_path: Path | None,
) -> List[Dict[str, Any]]:
    if not folder_path.exists() or not folder_path.is_dir():
        raise NotADirectoryError(f"Invalid folder: {folder_path}")

    extractor = QuestionExtractor()
    try:
        results = extractor.process_exam(
            folder_path=folder_path,
            is_submission=is_submission,
            save_results=save_results,
            output_path=output_path,
        )

        if not isinstance(results, list):
            raise AssertionError("process_exam: results must be a list")
        if len(results) == 0:
            raise AssertionError("process_exam: no images processed")

        for idx, result in enumerate(results):
            _assert_result_schema(result, context=f"process_exam item {idx}")

        return results
    finally:
        extractor.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real integration test for QuestionExtractor (no mocking)."
    )
    parser.add_argument("--image", type=Path, help="Path to a single image to test")
    parser.add_argument("--folder", type=Path, help="Path to a folder of images to batch test")
    parser.add_argument(
        "--submission",
        action="store_true",
        help="Use submission prompts (default is correction prompts)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save batch results JSON when using --folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/Sections/test_output_openai/real_question_extractor_results.json"),
        help="Output JSON path for --folder with --save-results",
    )

    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide at least one of --image or --folder")

    if not AgentsConfig.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY/GEMINI_AI_API_KEY is not configured. "
            "Set it in your environment or .env before running real tests."
        )

    print(f"Model: {AgentsConfig.GEMINI_MODEL_NAME}")
    print(f"Rate limit: {AgentsConfig.RATE_LIMIT} req/min")

    if args.image:
        print(f"\n[REAL TEST] process_image -> {args.image}")
        image_result = run_real_image_test(args.image, is_submission=args.submission)
        print("process_image PASSED")
        print(json.dumps(image_result, ensure_ascii=False, indent=2))

    if args.folder:
        print(f"\n[REAL TEST] process_exam -> {args.folder}")
        batch_results = run_real_batch_test(
            folder_path=args.folder,
            is_submission=args.submission,
            save_results=args.save_results,
            output_path=args.output,
        )
        print(f"process_exam PASSED | items={len(batch_results)}")
        if args.save_results:
            print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
