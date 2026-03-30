import os
import sys
import json
import base64
import re
from pathlib import Path
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("❌ MISTRAL_API_KEY not found in .env")
    sys.exit(1)

client = Mistral(api_key=MISTRAL_API_KEY)

OCR_MODEL  = "mistral-ocr-latest"
LLM_MODEL  = "mistral-large-latest"

CONF_THRESHOLD = 0.75

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

Return ONLY a valid JSON array — no markdown, no extra text:
[
  {"word": "<word>", "confidence": <0.0–1.0>},
  {"word": "<word>", "confidence": <0.0–1.0>}
]

Assign confidence based on legibility:
  1.00 → perfectly clear
  0.75 → mostly clear
  0.50 → uncertain
  0.25 → barely readable
  0.00 → unreadable (use [UNK])
"""

# ============================================
# HELPERS
# ============================================

def encode_image(file_path: str) -> tuple[str, str]:
    """Return (base64_data, mime_type)."""
    suffix = Path(file_path).suffix.lower()
    mime   = "image/png" if suffix == ".png" else "image/jpeg"
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime

def parse_json(raw: str) -> list:
    """Safely extract JSON array from model response."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$",          "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []

# ============================================
# STEP 1 — OCR (Mistral OCR → raw markdown)
# ============================================

def run_ocr(file_path: str) -> str:
    """Upload image to Mistral OCR, return raw markdown text."""
    print(f"📤 Running OCR on {file_path}...")

    b64, mime = encode_image(file_path)

    response = client.ocr.process(
        model=OCR_MODEL,
        document={
            "type":      "image_url",
            "image_url": f"data:{mime};base64,{b64}",
        },
        include_image_base64=False,
    )

    if not response.pages:
        return ""

    raw_text = response.pages[0].markdown or ""
    print(f"✅ OCR complete — {len(raw_text)} characters extracted")
    return raw_text

# ============================================
# STEP 2 — LLM: OCR text → word + confidence JSON
# ============================================

def extract_words(ocr_text: str) -> list[dict]:
    """
    Send raw OCR text to Mistral LLM.
    Returns list of {"word": str, "confidence": float}.
    Same role as Gemini's generate_content in the original code.
    """
    if not ocr_text.strip():
        return []

    print("🧠 Extracting words with confidence scores...")

    response = client.chat.complete(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": OCR_PROMPT},
            {"role": "user",   "content": ocr_text},
        ],
        temperature=0.0,
        max_tokens=2048,
    )

    raw    = response.choices[0].message.content.strip()
    parsed = parse_json(raw)

    if not parsed:
        print("⚠️  Could not parse word list — treating full text as one chunk")
        # Graceful fallback: split by whitespace, assign neutral confidence
        words = ocr_text.split()
        return [{"word": w, "confidence": 0.8} for w in words]

    return parsed

# ============================================
# STEP 3 — CLEANING (identical logic to original)
# ============================================

def clean_ocr(ocr_json: list) -> tuple[str, int, int]:
    """
    Filter words below CONF_THRESHOLD → [UNK].
    Returns (clean_text, hallucinated_count, total_count).
    Exact same logic as the original Gemini version.
    """
    clean_words  = []
    hallucinated = 0
    total        = 0

    for w in ocr_json:
        total += 1
        if w["confidence"] < CONF_THRESHOLD:
            clean_words.append("[UNK]")
            hallucinated += 1
        else:
            clean_words.append(w["word"])

    return " ".join(clean_words), hallucinated, total

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python ocr_mistral.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    # ── Step 1: OCR ──────────────────────────────
    raw_text = run_ocr(file_path)

    if not raw_text:
        print("❌ OCR returned empty text")
        sys.exit(1)

    # ── Step 2: Extract words + confidence ───────
    ocr_json = extract_words(raw_text)

    if not ocr_json:
        print("❌ Could not extract word list")
        sys.exit(1)

    # ── Step 3: Clean (same logic as original) ───
    clean_text, hallucinated, total = clean_ocr(ocr_json)

    # ── Output ───────────────────────────────────
    print("\n📝 Extracted Text:\n")
    print(clean_text)

    print("\n📊 Stats:")
    print(f"  Total words       : {total}")
    print(f"  Hallucinated words: {hallucinated}")
    print(f"  Hallucination rate: {hallucinated / total:.2%}" if total else "  No words found")