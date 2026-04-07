from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from google import genai

load_dotenv()

# Keep model/client configuration style aligned with ocr_gemini.py
GEMINI_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
client = genai.Client(vertexai=True, api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.1-pro-preview"

MAX_RETRIES = 3
BASE_DELAY_S = 2


def _parse_json_text(raw: str) -> Dict[str, Any]:
	text = (raw or "").strip()
	text = re.sub(r"^```(?:json)?\s*", "", text)
	text = re.sub(r"\s*```$", "", text)
	return json.loads(text.strip())


def _load_json_file(path: Path) -> Dict[str, Any]:
	raw = path.read_text(encoding="utf-8")
	try:
		return json.loads(raw)
	except json.JSONDecodeError:
		start = raw.find("{")
		end = raw.rfind("}")
		if start != -1 and end != -1 and end > start:
			return json.loads(raw[start : end + 1])
		raise


def _extract_numeric_points(note: str) -> float:
	if not isinstance(note, str):
		return 0.0
	match = re.search(r"(\d+(?:\.\d+)?)", note)
	return float(match.group(1)) if match else 0.0


def _extract_max_points(item: Dict[str, Any]) -> float:
	content = item.get("content", {})
	notes = content.get("notes", [])
	if isinstance(notes, list) and notes:
		points = [_extract_numeric_points(x) for x in notes]
		points = [p for p in points if p > 0]
		if points:
			return max(points)
	return 1.0


def _build_index_by_qnum(items: List[Dict[str, Any]], answer_key: str) -> Dict[str, Dict[str, Any]]:
	indexed: Dict[str, Dict[str, Any]] = {}
	for item in items:
		content = item.get("content", {})
		content_block = content.get("content", {})
		answer_block = content.get(answer_key, {})

		qn = content_block.get("question_number")
		if qn is None:
			qn = answer_block.get("question_number")
		qn_str = str(qn).strip() if qn is not None else "UNKNOWN"
		indexed[qn_str] = item
	return indexed


def _build_grading_payload(
	submission_json: Dict[str, Any],
	correction_json: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], float]:
	submission_items = submission_json.get("submission_content", [])
	correction_items = correction_json.get("exam_content", [])

	sub_index = _build_index_by_qnum(submission_items, "student_answer")
	cor_index = _build_index_by_qnum(correction_items, "correct_answer")

	question_rows: List[Dict[str, Any]] = []
	max_score = 0.0

	for qn, cor_item in cor_index.items():
		sub_item = sub_index.get(qn, {})

		cor_content = cor_item.get("content", {})
		cor_answer = cor_content.get("correct_answer", {})
		cor_text = str(cor_answer.get("raw_text", "")).strip()

		sub_content = sub_item.get("content", {}) if isinstance(sub_item, dict) else {}
		sub_answer = sub_content.get("student_answer", {})
		sub_text = str(sub_answer.get("raw_text", "")).strip()

		question_text = ""
		cor_q = cor_content.get("content", {})
		if isinstance(cor_q, dict):
			question_text = str(cor_q.get("question_text", "")).strip()

		points = _extract_max_points(cor_item)
		max_score += points

		question_rows.append(
			{
				"question_number": qn,
				"question_text": question_text,
				"student_answer": sub_text,
				"correct_answer": cor_text,
				"max_points": points,
			}
		)

	return question_rows, max_score


def _build_grading_prompt(rows: List[Dict[str, Any]]) -> str:
	return (
		"You are an expert and kind primary-school teacher grading answers. "
		"Students are 6-10 years old. Be encouraging, simple, and friendly.\n\n"
		"Grade each question by semantic similarity between student_answer and correct_answer.\n"
		"Scoring rule: points must be one of [0, 0.25, 0.5, 0.75, 1] scaled by max_points.\n"
		"For each question, choose the nearest allowed value and do not exceed max_points.\n"
		"Then provide overall feedback for the child.\n\n"
		"Return ONLY valid JSON in this exact format:\n"
		"{\n"
		"  \"detailed_results\": [\n"
		"    {\n"
		"      \"question_number\": \"...\",\n"
		"      \"max_points\": 1.0,\n"
		"      \"awarded_points\": 0.75,\n"
		"      \"feedback\": \"friendly sentence\"\n"
		"    }\n"
		"  ],\n"
		"  \"overall_feedback\": \"friendly paragraph for a child\"\n"
		"}\n\n"
		"Questions to grade:\n"
		f"{json.dumps(rows, ensure_ascii=False, indent=2)}"
	)


def _call_gemini_with_retry(prompt: str) -> Dict[str, Any]:
	last_error = "Unknown error"
	for attempt in range(1, MAX_RETRIES + 1):
		try:
			response = client.models.generate_content(
				model=GEMINI_MODEL,
				contents=[prompt],
			)
			return _parse_json_text(response.text if response else "")
		except Exception as exc:
			last_error = str(exc)
			if attempt == MAX_RETRIES:
				break
			time.sleep(BASE_DELAY_S * attempt)
	raise RuntimeError(f"Grading failed after retries: {last_error}")


def _compute_grade(percentage: float) -> str:
	if percentage >= 90:
		return "A"
	if percentage >= 80:
		return "B"
	if percentage >= 70:
		return "C"
	if percentage >= 60:
		return "D"
	return "Needs Practice"


def grade_exam(
	submission_json_path: Path,
	correction_json_path: Path,
) -> Dict[str, Any]:
	submission_json = _load_json_file(submission_json_path)
	correction_json = _load_json_file(correction_json_path)

	rows, max_score = _build_grading_payload(submission_json, correction_json)
	prompt = _build_grading_prompt(rows)
	grading_data = _call_gemini_with_retry(prompt)

	details = grading_data.get("detailed_results", [])
	total_score = 0.0
	for row in details:
		try:
			total_score += float(row.get("awarded_points", 0.0))
		except Exception:
			total_score += 0.0

	if max_score <= 0:
		max_score = 1.0
	percentage = (total_score / max_score) * 100.0

	result = {
		"total_score": round(total_score, 2),
		"max_score": round(max_score, 2),
		"percentage": round(percentage, 2),
		"feedback": grading_data.get(
			"overall_feedback",
			"Great effort. Keep practicing, you can do it!",
		),
		"final_grade": _compute_grade(percentage),
	}
	return result


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Grade student submission JSON against correction JSON"
	)
	parser.add_argument("--submission_json", required=True, help="Path to submission JSON")
	parser.add_argument("--correction_json", required=True, help="Path to correction JSON")
	parser.add_argument(
		"--output",
		default=None,
		help=(
			"Output JSON path. Default: agents_module/exam_correction/output json/"
			"<submission_stem>_grading.json"
		),
	)
	args = parser.parse_args()

	submission_json_path = Path(args.submission_json)
	correction_json_path = Path(args.correction_json)

	if not submission_json_path.exists():
		raise FileNotFoundError(f"submission_json not found: {submission_json_path}")
	if not correction_json_path.exists():
		raise FileNotFoundError(f"correction_json not found: {correction_json_path}")

	result = grade_exam(submission_json_path, correction_json_path)

	if args.output:
		output_path = Path(args.output)
	else:
		output_path = (
			Path("agents_module/exam_correction/output json")
			/ f"{submission_json_path.stem}_grading.json"
		)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(result, f, indent=2, ensure_ascii=False)

	print(json.dumps(result, indent=2, ensure_ascii=False))
	print(f"\nSaved JSON: {output_path}")


if __name__ == "__main__":
	main()
