import os
import sys
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

OCR_PROMPT = """
You are a STRICT multilingual OCR engine.

The document may contain:
- Arabic
- French
- English
- Numbers

Rules:
- Keep the original language
- Do NOT translate
- Do NOT guess
- Unreadable → [UNK]
- Return ONLY the extracted text, nothing else
"""

CONF_THRESHOLD = 0.75


# ---------------- OCR ---------------- #
def run_ocr(file_path):
    # Upload the file to Gemini
    print(f"📤 Uploading {file_path}...")
    uploaded_file = client.files.upload(file=file_path)
    
    print(f"✅ File uploaded: {uploaded_file.name}")
    
    # Use the uploaded file URI
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=[OCR_PROMPT, uploaded_file]
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
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python ocr.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    print("🔍 Running OCR...")

    raw = run_ocr(file_path)

    # Convert Gemini text → JSON
    try:
        ocr_json = json.loads(raw)
    except json.JSONDecodeError:
        print("❌ Gemini did not return valid JSON:")
        print(raw)
        sys.exit(1)

    clean_text, hallucinated, total = clean_ocr(ocr_json)

    print("\n📝 Extracted Text:\n")
    print(clean_text)

    print("\n📊 Stats:")
    print(f"Total words: {total}")
    print(f"Hallucinated words: {hallucinated}")
    print(f"Hallucination rate: {hallucinated/total:.2%}")