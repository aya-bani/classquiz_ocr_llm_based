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

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

INPUT_FOLDER = "Exams/sections/sc"
OUTPUT_FOLDER = "Exams/extraction_results"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_section_number(filename):
    """Extract section number from filename"""
    numbers = re.findall(r'\d+', filename)
    return numbers[0] if numbers else "unknown"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_student_answer(image_path):
    """
    Extract ONLY the student's handwritten answer from التعليمة
    Processes ONLY the FIRST page of each image
    """
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
            return {"student_answer": "", "note": "No pages found"}
        
        first_page = ocr_response.pages[0]
        raw_text = first_page.markdown
        
        if not raw_text.strip():
            return {"student_answer": "", "note": "No text detected on page 1"}
        
        # PROMPT WITH REAL EXAMPLES
        system_prompt = """You are an expert at extracting student answers from Arabic exam papers.

Structure:
- السند (Sind): The story, problem context, or narrative
- التعليمة (Talima): Contains TWO parts:
   1. PRINTED INSTRUCTION: Words like "أَحْسُبُ", "أَكْمِلُ", "أَشْرَحُ", "لِمَاذَا", etc.
   2. STUDENT'S HANDWRITTEN ANSWER: The actual calculation, response, or explanation

Your task: Extract ONLY the student's handwritten answer, NOT the printed instruction."""

        user_prompt = f"""
Extract ONLY the student's handwritten answer from this exam section.

--- COMPLETE EXAM TEXT ---
{raw_text}
--- END ---

**RULES:**
1. Locate the التعليمة section
2. Within التعليمة, identify:
   - The printed instruction (starts with: أحسب, أكمل, أشرح, لماذا, etc.) → IGNORE
   - The student's handwritten answer → EXTRACT THIS
3. The student's answer can be:
   - Mathematical calculations (numbers, +, -, =, ×, ÷)
   - Arabic sentences or phrases
   - Single words or numbers
4. Preserve exact formatting
5. Return empty string if no student answer found

**Student's Handwritten Answer:**
"""

        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        student_answer = chat_response.choices[0].message.content.strip()
        
        return {
            "student_answer": student_answer,
            "note": "Student answer extracted" if student_answer else "No answer found",
            "pages_processed": 1,
            "total_pages_in_image": len(ocr_response.pages)
        }
        
    except Exception as e:
        return {"student_answer": "", "error": str(e)}

# ============================================
# BATCH PROCESSING - FIXED FOR DUPLICATES
# ============================================

def process_all_sections(folder_path):
    folder = Path(folder_path)
    
    # Get all images
    images = []
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        images.extend(folder.glob(f"*{ext}"))
    
    if not images:
        print(f"❌ No images found in {folder_path}")
        return []
    
    # ========== FIX: Remove duplicates by filename stem ==========
    unique_images = {}
    for img in images:
        stem = img.stem  # Get filename without extension (e.g., "p1")
        if stem not in unique_images:
            unique_images[stem] = img
        else:
            print(f"⚠️  Duplicate found: {img.name} (keeping {unique_images[stem].name})")
    
    # Convert back to list
    images = list(unique_images.values())
    
    # Sort by section number
    images.sort(key=lambda x: extract_section_number(x.name))
    
    print(f"\n{'='*60}")
    print(f"📁 Processing folder: {folder_path}")
    print(f"📸 Found {len(images)} unique section images (after removing duplicates)")
    print(f"{'='*60}\n")
    
    results = []
    
    for idx, image_path in enumerate(images, 1):
        section_num = extract_section_number(image_path.name)
        print(f"[{idx}/{len(images)}] Section {section_num}: {image_path.name}")
        print(f"  🔍 Extracting student answer...")
        
        extracted = extract_student_answer(str(image_path))
        
        result = {
            "section_number": section_num,
            "filename": image_path.name,
            "student_answer": extracted.get("student_answer", ""),
            "note": extracted.get("note", ""),
            "pages_processed": extracted.get("pages_processed", 1),
            "total_pages_in_image": extracted.get("total_pages_in_image", 1),
            "error": extracted.get("error", None)
        }
        
        results.append(result)
        
        if result["student_answer"]:
            answer_preview = result["student_answer"][:80]
            print(f"  ✍️  Answer: {answer_preview}...")
        else:
            print(f"  📭 {result['note']}")
        
        if result["total_pages_in_image"] > 1:
            print(f"  📄 Note: Image had {result['total_pages_in_image']} pages, processed only page 1")
        print()
    
    return results

# ============================================
# SAVE RESULTS
# ============================================

def save_results(results, folder_path, output_folder):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_data = {
        "metadata": {
            "source_folder": folder_path,
            "total_sections": len(results),
            "extraction_date": datetime.now().isoformat(),
            "extraction_type": "student_handwritten_answers_only",
            "note": "Each image processed as 1 section (first page only, duplicates removed)"
        },
        "results": results
    }
    
    json_file = Path(output_folder) / f"student_answers_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 JSON saved to: {json_file}")
    
    # Simple text file with only answers
    txt_file = Path(output_folder) / f"student_answers_{timestamp}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("STUDENT HANDWRITTEN ANSWERS\n")
        f.write(f"Extracted from التعليمة (ignoring printed instructions)\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"Section {result['section_number']}:\n")
            f.write(f"File: {result['filename']}\n")
            f.write("-"*40 + "\n")
            if result['student_answer']:
                f.write(result['student_answer'] + "\n")
            else:
                f.write("[No handwritten answer detected]\n")
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"💾 Text file saved to: {txt_file}")
    
    return json_file

def print_summary(results):
    if not results:
        print("\n❌ No results to summarize")
        return
    
    print(f"\n{'='*60}")
    print("📊 STUDENT ANSWERS SUMMARY")
    print(f"{'='*60}")
    print(f"Total sections processed: {len(results)}")
    
    with_answers = sum(1 for r in results if r['student_answer'].strip())
    empty_answers = len(results) - with_answers
    
    print(f"\n✍️  Sections with handwritten answers: {with_answers}")
    print(f"📄 Sections without answers: {empty_answers}")
    
    if with_answers > 0:
        print(f"\n📝 SAMPLE ANSWERS (first 3):")
        print("-"*60)
        sample_count = 0
        for r in results:
            if r['student_answer'].strip() and sample_count < 3:
                print(f"\nSection {r['section_number']}:")
                print(f"Answer: {r['student_answer'][:200]}")
                sample_count += 1

# ============================================
# MAIN
# ============================================

def main():
    print("🚀 Student Handwritten Answer Extractor")
    print("🎯 Mode: Extract ONLY student's answer from التعليمة")
    print("🎯 Ignores: Printed instructions (أَحْسُبُ, أَكْمِلُ, لِمَاذَا, etc.)")
    print("🎯 Note: Processing ONLY first page of each image")
    print("🎯 Fix: Duplicate images removed automatically")
    print("="*60)
    
    INPUT_FOLDER = "Exams/sections/sc"
    OUTPUT_FOLDER = "Exams/extraction_results"
    
    if not Path(INPUT_FOLDER).exists():
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        return
    
    results = process_all_sections(INPUT_FOLDER)
    
    if results:
        save_results(results, INPUT_FOLDER, OUTPUT_FOLDER)
        print_summary(results)
        print(f"\n✅ Extraction complete!")
    else:
        print("\n❌ No results extracted")

if __name__ == "__main__":
    main()