
import os
import sys
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
client = genai.Client(vertexai=True, api_key=GEMINI_API_KEY)

EXTRACTION_PROMPT = """
You are a STRICT multilingual OCR and content extraction engine specialized in extracting structured information from exam correction images.

Instructions:
- Extract the following fields from the image:
    - question: The full text of the question as it appears.
    - corrected_answer: The full corrected answer as written by the teacher or correction. If the answer is a drawing, diagram, or visual (such as a clock, graph, or sketch), describe in detail what is drawn, including positions, labels, and any handwritten annotations. For visual answers (drawings, diagrams, clocks, etc.), provide a detailed description of the drawing and any relevant features, similar to: "Handwritten blue clock hands drawn inside the left circle. The short hand points to approximately the 10 o'clock position and the long hand points to the 1 o'clock position."
    - subject: The subject of the exam (e.g., Math, Science, etc.).
    - level: The level or grade (e.g., 2eme, 3ème année, etc.).
- Ignore any irrelevant printed text, page numbers, or unrelated marks.
- Keep the original language (Arabic, French, English, Numbers).
- If a field is missing or unreadable, set its value to [UNK].
- Return ONLY a valid JSON object with the extracted fields, nothing else.
"""


def _normalize_arabic_digits(text):
	if not isinstance(text, str):
		return text
	trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
	return text.translate(trans)


def _extract_question_number(question_text):
	if not isinstance(question_text, str):
		return None
	norm = _normalize_arabic_digits(question_text)
	# Examples: تعليمة1, تعليمة 1, تَعْلِيمَة 8
	match = re.search(r"تعليمة\s*([0-9]+)", norm, re.IGNORECASE)
	if match:
		return match.group(1)
	return None


def _to_structured_output(raw_data, image_path):
	question_text = raw_data.get("question")
	corrected_answer = raw_data.get("corrected_answer")
	subject = raw_data.get("subject")
	question_number = _extract_question_number(question_text)

	if corrected_answer and corrected_answer != "[UNK]":
		options = [{"id": "A", "text": corrected_answer}]
	else:
		options = []

	return {
		"question_type": "UNKNOWN",
		"confidence": 0.95,
		"content": {
			"correct_answer": {
				"question_number": question_number,
				"question_text": None,
				"correct answer ": options,
			},
			"notes": ["1 point"],
			"confidence": 0.95,
		},
		"meta_data": {
			"image_path": image_path,
			"image_name": os.path.basename(image_path),
		},
	}


def extract_correction_content(image_path):
	print(f"📤 Processing {image_path}...")
	with open(image_path, "rb") as f:
		file_bytes = f.read()

	ext = os.path.splitext(image_path)[1].lower()
	mime_type = "image/png" if ext == ".png" else "image/jpeg"

	image_part = types.Part.from_bytes(
		data=file_bytes,
		mime_type=mime_type,
	)

	response = client.models.generate_content(
		model="gemini-3.1-pro-preview",
		contents=[EXTRACTION_PROMPT, image_part]
	)

	raw = response.text.strip()
	# Remove markdown code fences if present
	raw = re.sub(r"^```(?:json)?", "", raw)
	raw = re.sub(r"```$", "", raw)
	raw = raw.strip()

	try:
		data = json.loads(raw)
	except Exception:
		print(f"❌ Could not parse JSON for {image_path}:")
		print(raw)
		# Always save the raw output for debugging
		out_txt_path = os.path.splitext(image_path)[0] + "_correction_content_raw.txt"
		with open(out_txt_path, "w", encoding="utf-8") as f:
			f.write(raw)
		print(f"Raw output saved to: {out_txt_path}")
		return None

	return _to_structured_output(data, image_path)


def iter_image_paths(input_path):
	if os.path.isdir(input_path):
		for name in sorted(os.listdir(input_path)):
			path = os.path.join(input_path, name)
			if os.path.isfile(path) and os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg"}:
				yield path
	else:
		yield input_path

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python extract_correction_content.py <image_path_or_folder>")
		sys.exit(1)

	output_dir = os.path.join("Exams", "content_correction_jsons")
	os.makedirs(output_dir, exist_ok=True)
	input_path = sys.argv[1]
	image_paths = list(iter_image_paths(input_path))

	if not image_paths:
		print(f"❌ No supported image files found in: {input_path}")
		sys.exit(1)

	if os.path.isdir(input_path):
		results = []
		for image_path in image_paths:
			result = extract_correction_content(image_path)
			if result:
				results.append(result)
			else:
				print(f"\n❌ JSON file not created for {image_path}. See raw output for details.")

		folder_name = os.path.basename(os.path.normpath(input_path))
		out_path = os.path.join(output_dir, folder_name + "_correction_content.json")
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump(results, f, indent=2, ensure_ascii=False)
		print(f"\n✅ Combined JSON file saved: {out_path}")
	else:
		image_path = image_paths[0]
		result = extract_correction_content(image_path)
		base_name = os.path.splitext(os.path.basename(image_path))[0]
		out_path = os.path.join(output_dir, base_name + "_correction_content.json")

		if result:
			print("\n===== Extracted Correction Content =====\n")
			print(json.dumps(result, indent=2, ensure_ascii=False))
			with open(out_path, "w", encoding="utf-8") as f:
				json.dump(result, f, indent=2, ensure_ascii=False)
			print(f"\n✅ JSON file saved: {out_path}")
		else:
			print(f"\n❌ JSON file not created for {image_path}. See raw output for details.")