import os
import json
import base64
import re
from pathlib import Path
from datetime import datetime
from mistralai.client import Mistral

# ============================================
# CONFIGURATION
# ============================================

api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    print("❌ MISTRAL_API_KEY not found")
    exit(1)

client = Mistral(api_key=api_key)

MODEL = "mistral-ocr-latest"

OUTPUT_FOLDER = "Exams/extraction_results"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ============================================
# PROMPT
# ============================================

EXTRACTION_PROMPT = """
You are analysing a cropped Arabic primary school exam image.

The image contains two types of sections:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. سند  (CONTEXT BLOCK)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Printed text that provides context (a story, table, numbers, etc.)
→ Extract it as-is into "sand"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. تعليمة  (INSTRUCTION + ANSWER ZONE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each تعليمة has TWO parts:

  A) PRINTED instruction text  (the question/task)
  B) HANDWRITTEN student answer — written on or around
     the dotted lines (............)

YOUR GOAL:
  - Extract the printed instruction into "instruction"
  - Extract ONLY the handwritten part into "handwritten_answer"
  - The dots themselves (............) are NOT part of the answer
  - Ignore printed text mixed with handwriting; take only what the
    student wrote by hand
  - There may be MULTIPLE تعليمة blocks per image — extract ALL of them

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARABIC MATH — RIGHT TO LEFT (RTL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Math is written RTL. Preserve the exact order the student wrote.
Example: "3380 = 2870 - 6250" means 6250 - 2870 = 3380 in LTR.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHILD HANDWRITING (ages 5–10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Digits may be reversed or malformed — use math context to resolve
- Crossed-out answers: take the LAST uncrossed answer only
- Ignore teacher correction marks (red pen, checkmarks, X)
- If truly illegible: write [illegible]
- If no answer written at all: use empty string ""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Western digits ONLY: 0 1 2 3 4 5 6 7 8 9
- Operators ONLY: - + x / =
- NO pipe symbols |
- NO dots/ellipsis from the answer lines

Confidence scale (per handwritten_answer):
  0.90–1.00 → Clear and legible
  0.70–0.89 → Mostly clear, minor uncertainty
  0.50–0.69 → Some characters unclear, context helped
  0.30–0.49 → Hard to read, significant guessing
  0.00–0.29 → Cannot extract reliably

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETURN FORMAT — valid JSON only, no markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "sand": "<full printed سند text, or empty string if none>",
  "taalimat": [
    {
      "index": 1,
      "instruction": "<printed تعليمة instruction text>",
      "handwritten_answer": "<only what the student wrote by hand>",
      "confidence": <0.0–1.0>
    },
    {
      "index": 2,
      "instruction": "<printed تعليمة instruction text>",
      "handwritten_answer": "<only what the student wrote by hand>",
      "confidence": <0.0–1.0>
    }
  ]
}

If there is no سند section, set "sand" to "".
If a تعليمة has no handwritten answer, set "handwritten_answer" to "".
Always return a valid JSON object — no extra text outside it.
"""

# ============================================
# HELPERS
# ============================================

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_section_number(filename: str) -> int | str:
    numbers = re.findall(r"\d+", filename)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            return numbers[0]
    return filename

def convert_arabic_digits(text: str) -> str:
    arabic_map = {
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    }
    for ar, w in arabic_map.items():
        text = text.replace(ar, w)
    return text

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = convert_arabic_digits(text)
    text = re.sub(r"[|│┃┊┋]", "", text)
    text = re.sub(r"\.{3,}", "", text)   # remove dot sequences (answer lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_response(raw: str) -> dict:
    """Extract and parse JSON from model response."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

def collect_images(folder: Path) -> list[Path]:
    seen: dict[str, Path] = {}
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        for img in folder.glob(f"*{ext}"):
            seen.setdefault(img.stem, img)
    return sorted(seen.values(), key=lambda p: extract_section_number(p.name))

# ============================================
# CORE — OCR + single vision call per image
# ============================================

def process_image(image_path: str) -> dict:
    """
    1. Run Mistral OCR to get raw markdown text
    2. Send image + OCR text + prompt to pixtral for structured extraction
    Returns structured dict with sand + taalimat list.
    """
    try:
        encoded = encode_image(image_path)
        suffix = Path(image_path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"

        # ── Step 1: OCR ──────────────────────────────────────────
        ocr_response = client.ocr.process(
            model=MODEL,
            document={
                "type": "image_url",
                "image_url": f"data:{mime};base64,{encoded}",
            },
            include_image_base64=False,
        )

        raw_ocr = ""
        if ocr_response.pages:
            raw_ocr = ocr_response.pages[0].markdown or ""

        if not raw_ocr.strip():
            return {
                "sand": "",
                "taalimat": [],
                "error": "OCR returned empty text"
            }

        # ── Step 2: Structured extraction from OCR text ──────────
        user_message = f"""{EXTRACTION_PROMPT}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAW OCR TEXT FROM IMAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{raw_ocr}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Now extract and return the JSON object.
"""

        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured data from Arabic exam OCR text. "
                        "Separate سند (printed context) from تعليمة sections. "
                        "For each تعليمة extract the printed instruction and the "
                        "handwritten student answer separately. "
                        "Return ONLY valid JSON."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=1500,
        )

        raw_response = chat_response.choices[0].message.content.strip()
        parsed = parse_response(raw_response)

        if not parsed:
            return {"sand": "", "taalimat": [], "error": "Failed to parse JSON response"}

        # ── Clean up extracted text ───────────────────────────────
        sand = clean_text(parsed.get("sand", ""))

        taalimat = []
        for t in parsed.get("taalimat", []):
            taalimat.append({
                "index": t.get("index", len(taalimat) + 1),
                "instruction": clean_text(t.get("instruction", "")),
                "handwritten_answer": clean_text(t.get("handwritten_answer", "")),
                "confidence": round(
                    max(0.0, min(1.0, float(t.get("confidence", 0.5)))), 2
                ),
            })

        return {"sand": sand, "taalimat": taalimat}

    except Exception as e:
        return {"sand": "", "taalimat": [], "error": str(e)}

# ============================================
# BATCH PROCESSING
# ============================================

def process_folder(folder_path: str) -> list[dict]:
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return []

    images = collect_images(folder)
    if not images:
        print("❌ No images found.")
        return []

    print(f"\n{'='*60}")
    print(f"📁  {folder_path}")
    print(f"🖼️   {len(images)} images  |  OCR: {MODEL}")
    print(f"🔍  Extracting: سند + تعليمة handwritten answers")
    print(f"{'='*60}\n")

    results = []

    for idx, img in enumerate(images, 1):
        section = extract_section_number(img.name)
        print(f"[{idx}/{len(images)}] Section {section}  →  {img.name}")

        extracted = process_image(str(img))

        result = {
            "section_number": section,
            "filename": img.name,
            "sand": extracted.get("sand", ""),
            "taalimat": extracted.get("taalimat", []),
        }
        if "error" in extracted:
            result["error"] = extracted["error"]
            print(f"  ⚠️  {extracted['error']}")
        else:
            # Print preview
            if result["sand"]:
                print(f"  📜  سند : {result['sand'][:60]}")
            for t in result["taalimat"]:
                ans = t["handwritten_answer"] or "(no answer)"
                print(f"  ✍️  تعليمة {t['index']} ({t['confidence']:.0%}) : {ans[:60]}")
        print()

        results.append(result)

    results.sort(key=lambda r: (isinstance(r["section_number"], str), r["section_number"]))
    return results

# ============================================
# SAVE + SUMMARY
# ============================================

def save_results(results: list[dict], source: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "metadata": {
            "source": source,
            "ocr_model": MODEL,
            "llm_model": "mistral-large-latest",
            "total_images": len(results),
            "date": datetime.now().isoformat(),
            "note": "sand flagged separately from تعليمة handwritten answers",
        },
        "results": results,
    }
    path = Path(OUTPUT_FOLDER) / f"structured_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved → {path}")
    return path


def print_summary(results: list[dict]):
    total_images = len(results)
    total_taalimat = sum(len(r["taalimat"]) for r in results)
    answered = sum(
        1 for r in results
        for t in r["taalimat"]
        if t.get("handwritten_answer")
    )
    with_sand = sum(1 for r in results if r.get("sand"))

    print(f"\n{'='*60}")
    print(f"📊  SUMMARY")
    print(f"{'='*60}")
    print(f"  Images processed   : {total_images}")
    print(f"  Images with سند    : {with_sand}")
    print(f"  Total تعليمة found : {total_taalimat}")
    print(f"  With handwriting   : {answered}")
    print(f"  Empty answers      : {total_taalimat - answered}")

    if answered:
        confidences = [
            t["confidence"]
            for r in results
            for t in r["taalimat"]
            if t.get("handwritten_answer")
        ]
        print(f"  Avg confidence     : {sum(confidences)/len(confidences):.0%}")

    print(f"\n📝 Sample output:")
    print("-" * 40)
    for r in results[:2]:
        print(f"  Section {r['section_number']}")
        if r["sand"]:
            print(f"    سند      : {r['sand'][:50]}")
        for t in r["taalimat"]:
            ans = t["handwritten_answer"] or "(empty)"
            print(f"    تعليمة {t['index']} : {ans[:50]}  [{t['confidence']:.0%}]")

# ============================================
# MAIN
# ============================================

def main():
    INPUT_FOLDER = "Exams/sections/sc"

    print("🔍 Arabic Exam Structured Extractor")
    print("=" * 50)
    print("  سند      → flagged separately (printed context)")
    print("  تعليمة   → instruction + handwritten answer split")
    print("  Dots (......) → stripped from answers")
    print("=" * 50)

    results = process_folder(INPUT_FOLDER)

    if results:
        save_results(results, INPUT_FOLDER)
        print_summary(results)
        print("\n✅ Done!")
    else:
        print("\n❌ No results produced.")


if __name__ == "__main__":
    main()