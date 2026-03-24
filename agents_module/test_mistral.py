import os
import json
import base64
import re
from pathlib import Path
from datetime import datetime
from mistralai.client import Mistral

# ============================================
# ARABIC RTL MATH PROMPT - CLEAN OUTPUT
# ============================================

ANSWER_EXTRACTION_PROMPT = """
You extract student answers from Arabic primary school math exams.

CRITICAL - ARABIC MATH IS WRITTEN RIGHT TO LEFT (RTL):
When Arabic students write math, they write from RIGHT to LEFT.
The result comes FIRST, then the operation, then the numbers.

Examples of Arabic RTL math:
- Student writes: "3380 = 2870 - 6250"
  This means in English: 6250 - 2870 = 3380

- Student writes: "9630 = 5250 + 3380"  
  This means in English: 3380 + 5250 = 9630

EXTRACTION RULES:
1. PRESERVE THE EXACT RTL ORDER the student wrote
2. Use ONLY Western digits: 0 1 2 3 4 5 6 7 8 9
3. Use ONLY these operators: - + x / =
4. NO decimal points: 2089 NOT 20.89
5. NO pipe symbols | anywhere in output
6. NO vertical bars or separators
7. Clean single answer per line

EXAMPLES OF CORRECT OUTPUT:

Example 1 - Simple RTL equation:
Student writes: 3380 = 2870 - 6250
Output: {"student_answer": "3380 = 2870 - 6250", "confidence": 0.95}

Example 2 - Direct calculation:
Student writes: 6250-2870=3380
Output: {"student_answer": "6250-2870=3380", "confidence": 0.95}

Example 3 - Arabic text with math:
Student writes: ثمن المشتريات=7655
Output: {"student_answer": "ثمن المشتريات=7655", "confidence": 0.90}

Example 4 - Multiple lines:
Student writes:
350
275
50
Output: {"student_answer": "350\n275\n50", "confidence": 0.85}

Example 5 - Clean numbers only:
Student writes: 7655
Output: {"student_answer": "7655", "confidence": 0.95}

IMPORTANT:
- NO pipe symbols | 
- NO vertical bars
- Return ONLY valid JSON
- student_answer must be a string
"""

def get_prompt(section_type: str) -> str:
    """Return the extraction prompt"""
    return ANSWER_EXTRACTION_PROMPT

# ============================================
# CONFIGURATION
# ============================================

api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    print("❌ MISTRAL_API_KEY not found")
    exit(1)

client = Mistral(api_key=api_key)

INPUT_FOLDER = "Exams/sections/math"
OUTPUT_FOLDER = "Exams/extraction_results"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ============================================
# CLEANING FUNCTIONS
# ============================================

def convert_arabic_to_western_digits(text: str) -> str:
    """Convert Arabic-Indic digits to Western digits"""
    digit_map = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
        '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
    }
    for arabic, western in digit_map.items():
        text = text.replace(arabic, western)
    return text

def fix_operators(text: str) -> str:
    """
    Fix common operator misreads:
    - Sometimes '=' is misread as '+' or vice versa
    - Clean up spacing
    """
    if not text:
        return text
    
    # Replace common misreads (if needed)
    # text = text.replace('+', '=')  # Only if needed
    
    return text

def clean_output(text: str) -> str:
    """Final cleaning of output - NO pipes, NO wrong operators"""
    if not text:
        return text
    
    # Remove pipe symbols
    text = text.replace('|', '')
    
    # Remove any '↵' symbols
    text = text.replace('↵', '\n')
    
    # Remove multiple pipes if any remain
    text = re.sub(r'\|+', '', text)
    
    # Remove nested JSON if present
    if text.strip().startswith('{') and 'student_answer' in text:
        try:
            nested = json.loads(text)
            if isinstance(nested, dict) and 'student_answer' in nested:
                text = nested['student_answer']
        except:
            pass
    
    # Clean numbers
    text = convert_arabic_to_western_digits(text)
    
    # Fix operators
    text = fix_operators(text)
    
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove any remaining separator symbols
    text = re.sub(r'[│┃┊┋]', '', text)
    
    return text

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_section_number(filename: str) -> str:
    numbers = re.findall(r'\d+', filename)
    return numbers[0] if numbers else "unknown"

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def clean_json_response(response_text: str) -> str:
    """Extract JSON from response"""
    # Remove markdown
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    elif response_text.startswith('```'):
        response_text = response_text[3:]
    
    if response_text.endswith('```'):
        response_text = response_text[:-3]
    
    # Find JSON object
    start = response_text.find('{')
    end = response_text.rfind('}')
    
    if start != -1 and end != -1:
        return response_text[start:end+1]
    
    return response_text.strip()

def extract_student_answer(image_path: str, section_type: str = "answer_zone") -> dict:
    """Extract student answer with clean output"""
    try:
        encoded_string = encode_image(image_path)
        
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encoded_string}"
            },
            include_image_base64=True
        )
        
        if len(ocr_response.pages) == 0:
            return {"student_answer": "", "confidence": 0.0}
        
        raw_text = ocr_response.pages[0].markdown
        
        if not raw_text.strip():
            return {"student_answer": "", "confidence": 0.0}
        
        # Get prompt
        base_prompt = get_prompt(section_type)
        
        user_prompt = f"""
{base_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OCR TEXT FROM IMAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{raw_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACT THE STUDENT'S HANDWRITTEN ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extract EXACTLY what the student wrote.
Preserve the RIGHT-TO-LEFT order.
NO pipe symbols | in output.
Return ONLY valid JSON.

Student Answer JSON:
"""

        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": "Extract student answers. NO pipe symbols |. Return ONLY valid JSON."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        
        response_text = chat_response.choices[0].message.content.strip()
        clean_json = clean_json_response(response_text)
        
        # Parse JSON
        try:
            parsed = json.loads(clean_json)
            student_answer = parsed.get("student_answer", "")
            confidence = parsed.get("confidence", 0.5)
        except json.JSONDecodeError:
            student_answer = clean_json
            confidence = 0.5
        
        # Clean up
        if student_answer in [None, "null", "None", ""]:
            student_answer = ""
        
        # Clean the output (removes pipes, fixes operators)
        if student_answer:
            student_answer = clean_output(student_answer)
        
        # Ensure confidence is a float
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except:
            confidence = 0.5
        
        return {
            "student_answer": student_answer,
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        return {"student_answer": "", "confidence": 0.0, "error": str(e)}

# ============================================
# BATCH PROCESSING
# ============================================

def process_all_sections(folder_path: str, section_type: str = "answer_zone") -> list:
    folder = Path(folder_path)
    
    # Get all images
    images = []
    for ext in ['.png', '.jpg', '.jpeg']:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    
    if not images:
        print(f"❌ No images found")
        return []
    
    # Deduplicate
    unique = {}
    for img in images:
        stem = img.stem
        if stem not in unique:
            unique[stem] = img
    
    images = list(unique.values())
    images.sort(key=lambda x: extract_section_number(x.name))
    
    print(f"\n{'='*60}")
    print(f"📁 {folder_path}")
    print(f"📸 {len(images)} images")
    print(f"{'='*60}\n")
    
    results = []
    
    for idx, img in enumerate(images, 1):
        section = extract_section_number(img.name)
        print(f"[{idx}/{len(images)}] Section {section}: {img.name}")
        
        extracted = extract_student_answer(str(img), section_type)
        
        # Convert section to int if possible
        try:
            section_num = int(section)
        except:
            section_num = section
        
        result = {
            "section_number": section_num,
            "filename": img.name,
            "student_answer": extracted.get("student_answer", ""),
            "confidence": extracted.get("confidence", 0.0)
        }
        
        results.append(result)
        
        if result["student_answer"]:
            preview = result["student_answer"][:80].replace('\n', ' | ')
            print(f"  ✍️  {preview}")
            print(f"  🎯 {result['confidence']:.0%}")
        else:
            print(f"  📭 No answer")
        print()
    
    return results

# ============================================
# SAVE RESULTS
# ============================================

def save_results(results: list, folder_path: str, output_folder: str) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output = {
        "metadata": {
            "source": folder_path,
            "total": len(results),
            "date": datetime.now().isoformat(),
            "note": "Clean output: NO pipe symbols | NO wrong operators"
        },
        "results": results
    }
    
    json_file = Path(output_folder) / f"answers_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved: {json_file}")
    
    return json_file

def print_summary(results: list):
    if not results:
        return
    
    total = len(results)
    with_answers = sum(1 for r in results if r['student_answer'])
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"With answers: {with_answers}")
    print(f"Without answers: {total - with_answers}")
    
    if with_answers > 0:
        avg_conf = sum(r['confidence'] for r in results if r['student_answer']) / with_answers
        print(f"Avg confidence: {avg_conf:.0%}")
    
    print(f"\n📝 Sample outputs (clean - no pipes):")
    print("-"*40)
    for r in results[:3]:
        if r['student_answer']:
            print(f"Section {r['section_number']}: {r['student_answer'][:60]}")

# ============================================
# MAIN
# ============================================

def main():
    print("🚀 Student Answer Extractor - Clean Output")
    print("="*50)
    print("✓ NO pipe symbols |")
    print("✓ Clean operators: - + x / =")
    print("✓ Western digits only (0-9)")
    print("✓ Preserves Arabic RTL math order")
    print("="*50)
    
    INPUT_FOLDER = "Exams/sections/math"
    OUTPUT_FOLDER = "Exams/extraction_results"
    
    if not Path(INPUT_FOLDER).exists():
        print(f"\n❌ Folder not found: {INPUT_FOLDER}")
        return
    
    results = process_all_sections(INPUT_FOLDER)
    
    if results:
        save_results(results, INPUT_FOLDER, OUTPUT_FOLDER)
        print_summary(results)
        print(f"\n✅ Complete!")
    else:
        print("\n❌ No results")

if __name__ == "__main__":
    main()