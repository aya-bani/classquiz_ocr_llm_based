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

MODEL = "pixtral-large-latest"

OUTPUT_FOLDER = "Exams/extraction_results"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ============================================
# PROMPTS
# ============================================

BASE_PROMPT = """
You are an expert at reading Arabic primary school math exam answer zones.
You will be shown a cropped image of a student's handwritten answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL — ARABIC MATH IS READ RIGHT TO LEFT (RTL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Arabic students write and read math RIGHT → LEFT.
The result appears on the RIGHT, operands flow to the LEFT.

Example: student writes "3380 = 2870 - 6250"
Meaning (LTR): 6250 - 2870 = 3380

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHILD HANDWRITING RULES (ages 5–10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Digits may be reversed or malformed — use math context to resolve
- Crossed-out answers: take the LAST uncrossed answer
- Ignore teacher marks (red pen, checkmarks, X's)
- Ignore stray marks or eraser smudges
- Arabic letters: transcribe the INTENDED word from context
- If a value is truly illegible, write [illegible]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Western digits ONLY: 0 1 2 3 4 5 6 7 8 9
- Operators ONLY: - + x / =
- NO pipe symbols |
- NO decimal points unless clearly written
- Preserve the RTL order exactly as written
- If no answer is visible, return empty string ""

Confidence scale:
  0.90–1.00 → Clear and legible
  0.70–0.89 → Mostly clear, minor uncertainty
  0.50–0.69 → Some characters unclear, context helped
  0.30–0.49 → Hard to read, significant guessing
  0.00–0.29 → Cannot extract reliably

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETURN FORMAT — valid JSON only, no markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{"student_answer": "<exactly what the student wrote>", "confidence": <0.0–1.0>}
"""

def build_prompt(user_prompt: str) -> str:
    """Combine base prompt with caller-supplied context."""
    if not user_prompt or not user_prompt.strip():
        return BASE_PROMPT
    return f"{BASE_PROMPT}\n\nADDITIONAL CONTEXT FROM TEACHER:\n{user_prompt.strip()}"

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

def parse_response(raw: str) -> tuple[str, float]:
    """Extract student_answer and confidence from model response."""
    # Strip markdown fences if present
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # Find outermost JSON object
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        data = json.loads(text)
        answer = data.get("student_answer", "") or ""
        confidence = float(data.get("confidence", 0.5))
        confidence = round(max(0.0, min(1.0, confidence)), 2)
        return answer.strip(), confidence
    except (json.JSONDecodeError, ValueError):
        return text.strip(), 0.5

def clean_answer(text: str) -> str:
    """Remove pipe symbols and normalise Arabic-Indic digits."""
    if not text:
        return text

    arabic_map = {
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    }
    for ar, w in arabic_map.items():
        text = text.replace(ar, w)

    text = re.sub(r"[|│┃┊┋↵]", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# ============================================
# CORE — single vision call
# ============================================

def extract_answer(image_path: str, prompt: str) -> dict:
    """
    Send one image to pixtral-large-latest with the combined prompt.
    Returns {"student_answer": str, "confidence": float}.
    """
    try:
        encoded = encode_image(image_path)
        suffix = Path(image_path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"

        full_prompt = build_prompt(prompt)

        response = client.chat.complete(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()
        answer, confidence = parse_response(raw)
        answer = clean_answer(answer)

        return {"student_answer": answer, "confidence": confidence}

    except Exception as e:
        return {"student_answer": "", "confidence": 0.0, "error": str(e)}

# ============================================
# BATCH PROCESSING
# ============================================

def collect_images(folder: Path) -> list[Path]:
    """Gather unique images from folder, sorted by section number."""
    seen: dict[str, Path] = {}
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        for img in folder.glob(f"*{ext}"):
            seen.setdefault(img.stem, img)
    return sorted(seen.values(), key=lambda p: extract_section_number(p.name))


def process_folder(folder_path: str, prompt: str = "") -> list[dict]:
    """
    Process all section images in folder_path.

    Args:
        folder_path: path to folder containing cropped section images
        prompt:      optional teacher-supplied context appended to base prompt

    Returns:
        List of result dicts sorted by section number.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return []

    images = collect_images(folder)
    if not images:
        print("❌ No images found in folder.")
        return []

    print(f"\n{'='*60}")
    print(f"📁  {folder_path}")
    print(f"🖼️   {len(images)} images  |  model: {MODEL}")
    print(f"{'='*60}\n")

    results = []

    for idx, img in enumerate(images, 1):
        section = extract_section_number(img.name)
        print(f"[{idx}/{len(images)}] Section {section}  →  {img.name}")

        extracted = extract_answer(str(img), prompt)

        result = {
            "section_number": section,
            "filename": img.name,
            "student_answer": extracted.get("student_answer", ""),
            "confidence": extracted.get("confidence", 0.0),
        }
        if "error" in extracted:
            result["error"] = extracted["error"]

        results.append(result)

        if result["student_answer"]:
            preview = result["student_answer"][:80].replace("\n", " | ")
            print(f"  ✍️  {preview}")
            print(f"  🎯 {result['confidence']:.0%}")
        else:
            print(f"  📭 No answer detected")
        print()

    # Sort by section number (numeric where possible)
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
            "model": MODEL,
            "total_sections": len(results),
            "date": datetime.now().isoformat(),
            "api_calls": len(results),          # 1 call per image
        },
        "results": results,
    }
    path = Path(OUTPUT_FOLDER) / f"answers_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved → {path}")
    return path


def print_summary(results: list[dict]):
    total = len(results)
    answered = [r for r in results if r["student_answer"]]
    print(f"\n{'='*60}")
    print(f"📊  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total sections : {total}")
    print(f"  With answers   : {len(answered)}")
    print(f"  Empty          : {total - len(answered)}")
    print(f"  API calls used : {total}  (1 per image — no OCR step)")
    if answered:
        avg = sum(r["confidence"] for r in answered) / len(answered)
        print(f"  Avg confidence : {avg:.0%}")
    print(f"\n📝 First 3 results:")
    print("-" * 40)
    for r in results[:3]:
        ans = r["student_answer"] or "(empty)"
        print(f"  Section {r['section_number']:>3} | {r['confidence']:.0%} | {ans[:50]}")

# ============================================
# MAIN
# ============================================

def main():
    # ── Configure these two values ──────────────────────────────
    INPUT_FOLDER = "Exams/google_vision/math/splited images into sections/exam_1"
    PROMPT       = ""          # optional extra context; leave "" for defaults
    # ────────────────────────────────────────────────────────────

    print("🚀 Pixtral Answer Extractor  (1 API call per image)")
    print("=" * 50)

    results = process_folder(INPUT_FOLDER, PROMPT)

    if results:
        save_results(results, INPUT_FOLDER)
        print_summary(results)
        print("\n✅ Done!")
    else:
        print("\n❌ No results produced.")


if __name__ == "__main__":
    main()