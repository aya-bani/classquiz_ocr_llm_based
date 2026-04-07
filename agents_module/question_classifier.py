from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
	from agents_module.prompts import (
		CLASSIFICATION_PROMPT,
		TEMPLATES_CORRECTION_PROMPT,
	)
except Exception:
	from prompts import (  # type: ignore
		CLASSIFICATION_PROMPT,
		TEMPLATES_CORRECTION_PROMPT,
	)

load_dotenv()

# Keep Gemini model/client configuration aligned with ocr_gemini.py
GEMINI_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
client = genai.Client(vertexai=True, api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.1-pro-preview"

_ALLOWED_QUESTION_TYPES = {
	"ENONCE",
	"WRITING",
	"RELATING",
	"TABLE",
	"MULTIPLE_CHOICE",
	"TRUE_FALSE",
	"FILL_BLANK",
	"SHORT_ANSWER",
	"CALCULATION",
	"DIAGRAM",
	"UNKNOWN",
}


class QuestionClassifier:
	def __init__(self) -> None:
		if not GEMINI_API_KEY:
			raise EnvironmentError("GOOGLE_CLOUD_API_KEY not found in environment")

	@staticmethod
	def _image_part(image_path: Path) -> types.Part:
		if not image_path.exists() or not image_path.is_file():
			raise FileNotFoundError(f"Image not found: {image_path}")

		mime_type, _ = mimetypes.guess_type(str(image_path))
		if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
			mime_type = "image/jpeg"

		with open(image_path, "rb") as f:
			file_bytes = f.read()

		return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

	@staticmethod
	def _parse_json(text: str) -> Dict[str, Any]:
		raw = (text or "").strip()
		raw = re.sub(r"^```(?:json)?\s*", "", raw)
		raw = re.sub(r"\s*```$", "", raw)
		raw = raw.strip()
		try:
			return json.loads(raw)
		except json.JSONDecodeError as exc:
			return {"error": f"Failed to parse JSON: {exc}", "raw_text": raw}

	def classify_question(self, image_path: Path) -> Dict[str, Any]:
		image_part = self._image_part(image_path)
		classify_prompt = (
			f"{CLASSIFICATION_PROMPT}\n\n"
			"CRITICAL: Return strict JSON with keys: question_type, confidence, reasoning. "
			"question_type must be one of: ENONCE, WRITING, RELATING, TABLE, "
			"MULTIPLE_CHOICE, TRUE_FALSE, FILL_BLANK, SHORT_ANSWER, "
			"CALCULATION, DIAGRAM, UNKNOWN."
		)

		response = client.models.generate_content(
			model=GEMINI_MODEL,
			contents=[classify_prompt, image_part],
		)
		data = self._parse_json(response.text if response else "")

		question_type = str(data.get("question_type", "UNKNOWN")).upper().strip()
		confidence = float(data.get("confidence", 0.5))

		if question_type not in _ALLOWED_QUESTION_TYPES:
			question_type = "UNKNOWN"
			confidence = min(confidence, 0.5)

		return {
			"question_type": question_type,
			"confidence": confidence,
			"reasoning": data.get("reasoning", ""),
		}

	def extract_question_content(
		self,
		image_path: Path,
		question_type: str,
	) -> Dict[str, Any]:
		image_part = self._image_part(image_path)
		template = TEMPLATES_CORRECTION_PROMPT.get(question_type)
		if not template:
			template = TEMPLATES_CORRECTION_PROMPT["UNKNOWN"]

		extract_prompt = (
			f"{template}\n\n"
			"Extract only the printed question in original language. "
			"Do not translate. Return only valid JSON."
		)

		response = client.models.generate_content(
			model=GEMINI_MODEL,
			contents=[extract_prompt, image_part],
		)
		data = self._parse_json(response.text if response else "")

		# Match the requested shape where top-level 'content' wraps the payload.
		if isinstance(data, dict) and "content" in data and isinstance(data["content"], dict):
			normalized_content = data["content"]
		else:
			normalized_content = data

		return {"content": normalized_content}

	def process_image(
		self,
		image_path: Path,
		output_json_path: Path,
	) -> Dict[str, Any]:
		classification = self.classify_question(image_path)
		extracted = self.extract_question_content(
			image_path,
			classification["question_type"],
		)

		result = {
			"question_type": classification["question_type"],
			"confidence": classification["confidence"],
			"content": extracted,
			"meta_data": {
				"image_path": str(image_path),
				"image_name": image_path.name,
			},
		}

		output_json_path.parent.mkdir(parents=True, exist_ok=True)
		with open(output_json_path, "w", encoding="utf-8") as f:
			json.dump(result, f, indent=2, ensure_ascii=False)

		return result


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Classify and extract exam question image to JSON"
	)
	parser.add_argument("--image", required=True, help="Path to question image")
	parser.add_argument(
		"--output",
		default=(
			"agents_module/question_classification/"
			"question_classifier_output.json"
		),
		help="Path to output JSON file",
	)
	args = parser.parse_args()

	classifier = QuestionClassifier()
	result = classifier.process_image(Path(args.image), Path(args.output))

	print(json.dumps(result, indent=2, ensure_ascii=False))
	print(f"\nSaved JSON: {args.output}")


if __name__ == "__main__":
	main()
