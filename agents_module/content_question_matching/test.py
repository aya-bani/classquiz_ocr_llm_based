from __future__ import annotations

import argparse
import json
import os
import re
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


def _normalize_text(text: str) -> str:
	text = (text or "").strip().lower()
	text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
	text = text.replace("ى", "ي").replace("ة", "ه")
	text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
	text = text.replace("**", "")
	text = re.sub(r"\s+", " ", text)
	return text


def _extract_relating_matches(raw_text: str, question_content: Dict[str, Any]) -> list:
	items = question_content.get("items", []) if isinstance(question_content, dict) else []
	options = question_content.get("options", []) if isinstance(question_content, dict) else []

	if not isinstance(items, list) or not isinstance(options, list):
		return []

	item_lookup = {
		_normalize_text(str(item.get("text", ""))): str(item.get("id", ""))
		for item in items
		if isinstance(item, dict)
	}
	option_lookup = {
		_normalize_text(str(opt.get("text", ""))): str(opt.get("id", ""))
		for opt in options
		if isinstance(opt, dict)
	}

	lines = [line.strip() for line in str(raw_text or "").splitlines() if line.strip()]
	matches = []
	for line in lines:
		parts = re.split(r"\s*->\s*|\s*➔\s*|\s*→\s*", line, maxsplit=1)
		if len(parts) != 2:
			continue

		left = _normalize_text(parts[0])
		right = _normalize_text(parts[1])

		item_id = item_lookup.get(left)
		option_id = option_lookup.get(right)
		if item_id and option_id:
			matches.append({"item_id": item_id, "option_id": option_id})

	return matches


def _build_structured_student_answer(
	question_type: str,
	question_content: Dict[str, Any],
	raw_text: str,
) -> Dict[str, Any]:
	structured_answer = {"raw_text": raw_text}

	if str(question_type).upper() == "RELATING":
		structured_answer["matches"] = _extract_relating_matches(raw_text, question_content)

	return structured_answer


def _extract_student_answer(
	image_path: Path,
	question_type: str,
	question_content: Dict[str, Any],
) -> Dict[str, Any]:
	"""Use ocr_gemini to extract student's handwritten answer from an image."""
	ocr_text = ocr_gemini.run_ocr(str(image_path))
	content = (ocr_text or "").strip()
	if not content:
		content = "[UNK]"

	return _build_structured_student_answer(question_type, question_content, content)


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

	question_type = question_data.get("question_type", "UNKNOWN")
	confidence = float(question_data.get("confidence", 0.0))
	question_content = question_data.get("content", {})
	student_answer = _extract_student_answer(image_path, question_type, question_content)
	question_content_with_student = dict(question_content)
	question_content_with_student["student_answer"] = student_answer
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
			"content": question_content_with_student,
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
