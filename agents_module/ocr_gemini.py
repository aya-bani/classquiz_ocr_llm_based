import os
import sys
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
client = genai.Client(vertexai=True, api_key=GEMINI_API_KEY)

OCR_PROMPT = """
You are a STRICT multilingual OCR engine specialized in extracting handwritten answers from exam images.

Instructions:
- Your primary task is to extract ONLY the handwritten answers provided by the student. These answers must be the most important and clearly highlighted in your output.
- Ignore printed text, question headers, and instructions unless they are necessary to understand the handwritten answer.
- Adapt your extraction method based on the question type:
    - For short answer questions, extract the handwritten response as clearly as possible.
    - For 'ارسم' (draw) or graph-based questions, indicate the presence of a drawing or graph and describe any handwritten annotations or labels.
    - For multiple-choice or checkbox questions, extract the handwritten marks or selected options.
- Keep the original language (Arabic, French, English, Numbers).
- Do NOT translate or guess missing content.
- Unreadable → [UNK]
- Return ONLY the extracted handwritten answers, nothing else. Highlight them clearly in your output.
"""

CONF_THRESHOLD = 0.75


# ---------------- OCR ---------------- #
def run_ocr(file_path):
    print(f"📤 Processing {file_path}...")  

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    image_part = types.Part.from_bytes(
        data=file_bytes,
        mime_type="image/jpeg",
    )

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",  
        contents=[OCR_PROMPT, image_part]  
    )

    return response.text


# ---------------- CLEANING ---------------- #
def clean_ocr(ocr_json):
    clean_words = []
    hallucinated = 0
    total = 0

    for w in ocr_json:
        total += 1

        if w["confidence"] < CONF_THRESHOLD:
            clean_words.append("[UNK]")
            hallucinated += 1
        else:
            clean_words.append(w["word"])

    return " ".join(clean_words), hallucinated, total


# ---------------- MAIN ---------------- #
import re
from pathlib import Path


def get_section_number(filename):
    match = re.search(r'section_(\d+)', filename)
    return int(match.group(1)) if match else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_gemini.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    image_files = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        image_files.extend([str(p) for p in Path(folder_path).glob(ext)])

    image_files = sorted(image_files, key=get_section_number)

    results = []


    # Prepare output directory for JSONs
    output_dir = os.path.join("Exams", "content_extraction_jsons")
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_files:
        ext = os.path.splitext(img_path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"

        def run_ocr_with_mime(file_path, mime_type):
            print(f"📤 Processing {file_path}...")
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            image_part = types.Part.from_bytes(
                data=file_bytes,
                mime_type=mime_type,
            )

            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=[OCR_PROMPT, image_part]
            )
            return response.text

        raw = run_ocr_with_mime(img_path, mime_type)
        content = raw.strip() if raw else "[UNK]"

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        section_num = get_section_number(img_path)

        results.append({
            "section": section_num,
            "file": os.path.basename(img_path),
            "content": content,
            "confidence": None
        })

    results.sort(key=lambda x: x["section"])

    print("\n===== Extraction Results by Section =====\n")
    for r in results:
        print(f"Section {r['section']} ({r['file']}):")
        print(f"  Content: {r['content']}")
        print(f"  Confidence: {r['confidence']}")
        print()

    combined_json_path = os.path.join(
        output_dir,
        os.path.basename(os.path.normpath(folder_path)) + "_ocr_content.json",
    )
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ JSON file saved: {combined_json_path}")
