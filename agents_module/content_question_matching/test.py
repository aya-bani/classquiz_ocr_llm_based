from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running this file directly with: python agents_module\content_question_matching\test.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
AGENTS_MODULE_DIR = Path(__file__).resolve().parents[1]
if str(AGENTS_MODULE_DIR) not in sys.path:
	sys.path.insert(0, str(AGENTS_MODULE_DIR))

from question_classifier import QuestionClassifier
import ocr_gemini


def _extract_student_answer(image_path: Path) -> Dict[str, Any]:
	"""Use ocr_gemini to extract student's handwritten answer from an image."""
	ocr_text = ocr_gemini.run_ocr(str(image_path))
	content = (ocr_text or "").strip()
	if not content:
		content = "[UNK]"

	# Keep flexible structure since answer format depends on question type.
	return {"raw_text": content}


def match_question_and_answer(image_path: Path) -> Dict[str, Any]:
	"""
	Build a single submission item by combining:
	- question extraction from QuestionExtractor
	- student answer OCR from ocr_gemini
	"""
	classifier = QuestionClassifier()
	classification = classifier.classify_question(image_path)
	extracted = classifier.extract_question_content(
		image_path,
		classification["question_type"],
	)
	question_data = {
		"question_type": classification.get("question_type", "UNKNOWN"),
		"confidence": classification.get("confidence", 0.0),
		"content": extracted.get("content", {}),
		"meta_data": {
			"image_path": str(image_path),
			"image_name": image_path.name,
		},
	}

	student_answer = _extract_student_answer(image_path)

	question_type = question_data.get("question_type", "UNKNOWN")
	confidence = float(question_data.get("confidence", 0.0))
	question_content = question_data.get("content", {})
	meta_data = question_data.get(
		"meta_data",
		{
			"image_path": str(image_path),
			"image_name": image_path.name,
		},
	)

	submission_item = {
		"question_type": question_type,
		"confidence": confidence,
		"content": {
			"content": question_content,
			"student_answer": student_answer,
			"confidence": confidence,
		},
		"meta_data": meta_data,
	}

	return {"submission_content": [submission_item]}


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Match extracted question content with student OCR answer for one image"
		)
	)
	parser.add_argument("--image", required=True, help="Path to submission image")
	parser.add_argument(
		"--output",
		default=None,
		help=(
			"Output JSON path. Default: agents_module/content_question_matching/"
			"output_json/<image_name>_matched.json"
		),
	)
	args = parser.parse_args()

	image_path = Path(args.image)
	if not image_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")

	result = match_question_and_answer(image_path)

	if args.output:
		output_path = Path(args.output)
	else:
		output_path = Path(
			"agents_module/content_question_matching/output_json"
		) / f"{image_path.stem}_matched.json"

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2, ensure_ascii=False)

	print(json.dumps(result, indent=2, ensure_ascii=False))
	print(f"\nSaved JSON: {output_path}")


if __name__ == "__main__":
	main()
