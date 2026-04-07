from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
AGENTS_MODULE_DIR = Path(__file__).resolve().parents[1]
if str(AGENTS_MODULE_DIR) not in sys.path:
	sys.path.insert(0, str(AGENTS_MODULE_DIR))

from question_classifier import QuestionClassifier
from extract_correction_content import extract_correction_content


def _extract_correct_answer(image_path: Path) -> Dict[str, Any]:
	"""Use extract_correction_content to get corrected answer from correction image."""
	extracted = extract_correction_content(str(image_path))
	if not extracted:
		return {
			"question_number": None,
			"question_text": None,
			"raw_text": "[UNK]",
		}

	content = extracted.get("content", {})
	correct_answer = content.get("correct_answer", {})
	options = correct_answer.get("correct answer ", [])

	if options and isinstance(options, list):
		raw_text = "\n".join(
			str(item.get("text", ""))
			for item in options
			if isinstance(item, dict)
		).strip()
	else:
		raw_text = "[UNK]"

	return {
		"question_number": correct_answer.get("question_number"),
		"question_text": correct_answer.get("question_text"),
		"raw_text": raw_text or "[UNK]",
	}


def build_exam_content(image_path: Path) -> Dict[str, Any]:
	"""Build one exam_content item from question + corrected answer extraction."""
	classifier = QuestionClassifier()
	classification = classifier.classify_question(image_path)
	question_block = classifier.extract_question_content(
		image_path,
		classification["question_type"],
	)

	correct_answer = _extract_correct_answer(image_path)

	confidence = float(classification.get("confidence", 0.0))
	item = {
		"question_type": classification.get("question_type", "UNKNOWN"),
		"confidence": confidence,
		"content": {
			"content": question_block.get("content", {}),
			"correct_answer": correct_answer,
			"notes": ["1 point"],
			"confidence": confidence,
		},
		"meta_data": {
			"image_path": str(image_path),
			"image_name": image_path.name,
		},
	}

	return {"exam_content": [item]}


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Build exam_content JSON from one correction image"
	)
	parser.add_argument("--image", required=True, help="Path to correction image")
	parser.add_argument(
		"--output",
		default=None,
		help=(
			"Output JSON path. Default: agents_module/correction_question/"
			"output_jso/<image_name>_exam_content.json"
		),
	)
	args = parser.parse_args()

	image_path = Path(args.image)
	if not image_path.exists():
		raise FileNotFoundError(f"Image not found: {image_path}")

	result = build_exam_content(image_path)

	if args.output:
		output_path = Path(args.output)
	else:
		output_path = Path(
			"agents_module/correction_question/output_jso"
		) / f"{image_path.stem}_exam_content.json"

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2, ensure_ascii=False)

	print(json.dumps(result, indent=2, ensure_ascii=False))
	print(f"\nSaved JSON: {output_path}")


if __name__ == "__main__":
	main()
