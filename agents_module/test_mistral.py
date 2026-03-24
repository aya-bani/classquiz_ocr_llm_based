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

# Configure paths - CHANGE THESE
INPUT_FOLDER = "Exams/google_vision/math/splited images into sections/exam_1"
OUTPUT_FOLDER = "Exams/extraction_results"

# Create output folder
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_section_number(filename):
    """Extract section number from filename"""
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return numbers[0]
    return "unknown"

def encode_image(image_path):
    """Convert image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_student_answer(image_path):
    """
    Extract ONLY the student's answer from the تعليمة section
    Ignore السند (general question)
    """
    try:
        # Encode image
        encoded_string = encode_image(image_path)
        
        # OCR to get all text with layout
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encoded_string}"
            },
            include_image_base64=True
        )
        
        # Get raw text with structure
        raw_text = ""
        for page in ocr_response.pages:
            raw_text += page.markdown + "\n"
        
        if not raw_text.strip():
            return {
                "student_answer": "",
                "note": "No text detected"
            }
        
        # Extract using LLM - only the student's answer section
        system_prompt = """You extract student answers from exam images.

In Arabic exam papers:
- السند = The question text (what is being asked)
- التعليمة = The student's answer area

Your task: Extract ONLY the student's answer from the تعليمة section.
Ignore السند completely."""

        user_prompt = f"""
From this exam image, extract ONLY the student's handwritten answer:

--- FULL IMAGE TEXT ---
{raw_text}
--- END ---

Rules:
1. Identify which part is السند (question) and which part is التعليمة (student answer)
2. Extract ONLY the student's answer from التعليمة area
3. Do NOT include the question text
4. If multiple answers exist, combine them
5. If no student answer is found, return empty string
6. Return ONLY the student's answer text, no explanations

Student's Answer:
"""

        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        student_answer = chat_response.choices[0].message.content.strip()
        
        return {
            "student_answer": student_answer,
            "full_ocr_text": raw_text[:300]  # Keep for reference
        }
        
    except Exception as e:
        return {
            "student_answer": "",
            "error": str(e)
        }

# ============================================
# BATCH PROCESSING
# ============================================

def process_all_sections(folder_path):
    """
    Process all section images and extract student answers only
    """
    folder = Path(folder_path)
    
    # Get all images
    images = []
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        images.extend(folder.glob(f"*{ext}"))
    
    if not images:
        print(f"❌ No images found in {folder_path}")
        return []
    
    # Sort by section number
    images.sort(key=lambda x: extract_section_number(x.name))
    
    print(f"\n{'='*60}")
    print(f"📁 Processing folder: {folder_path}")
    print(f"📸 Found {len(images)} section images")
    print(f"{'='*60}\n")
    
    results = []
    
    for idx, image_path in enumerate(images, 1):
        section_num = extract_section_number(image_path.name)
        print(f"[{idx}/{len(images)}] Section {section_num}: {image_path.name}")
        print(f"  🔍 Extracting student answer from التعليمة...")
        
        # Extract student answer only
        extracted = extract_student_answer(str(image_path))
        
        result = {
            "section_number": section_num,
            "filename": image_path.name,
            "student_answer": extracted.get("student_answer", ""),
            "error": extracted.get("error", None)
        }
        
        results.append(result)
        
        # Print preview
        answer_preview = result["student_answer"][:100] if result["student_answer"] else "[No student answer detected]"
        print(f"  ✍️  Student answer: {answer_preview}...")
        print()
    
    return results

# ============================================
# SAVE RESULTS
# ============================================

def save_results(results, folder_path, output_folder):
    """
    Save extracted student answers to JSON
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare output
    output_data = {
        "metadata": {
            "source_folder": folder_path,
            "total_sections": len(results),
            "extraction_date": datetime.now().isoformat(),
            "extraction_type": "student_answers_from_talima"
        },
        "results": results
    }
    
    # Save JSON
    json_file = Path(output_folder) / f"student_answers_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 JSON saved to: {json_file}")
    
    # Save simple text file
    txt_file = Path(output_folder) / f"student_answers_{timestamp}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("STUDENT ANSWERS FROM التعليمة\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"Section {result['section_number']}:\n")
            f.write(f"File: {result['filename']}\n")
            f.write("-"*40 + "\n")
            if result['student_answer']:
                f.write(result['student_answer'] + "\n")
            else:
                f.write("[No student answer detected]\n")
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"💾 Text file saved to: {txt_file}")
    
    return json_file

def print_summary(results):
    """
    Print summary of extraction
    """
    if not results:
        print("\n❌ No results to summarize")
        return
    
    print(f"\n{'='*60}")
    print("📊 STUDENT ANSWERS SUMMARY")
    print(f"{'='*60}")
    print(f"Total sections processed: {len(results)}")
    
    # Count sections with answers
    with_answers = sum(1 for r in results if r['student_answer'].strip())
    empty_answers = len(results) - with_answers
    
    print(f"\n✍️  Sections with student answers: {with_answers}")
    print(f"📄 Sections without answers: {empty_answers}")
    
    # Print sample answers
    print(f"\n📝 SAMPLE STUDENT ANSWERS:")
    print("-"*60)
    for r in results[:3]:
        if r['student_answer'].strip():
            print(f"\nSection {r['section_number']}:")
            print(f"Answer: {r['student_answer'][:200]}")
            if len(r['student_answer']) > 200:
                print("...")

# ============================================
# MAIN
# ============================================

def main():
    """
    Main execution - Extract student answers from التعليمة only
    """
    # ============================================
    # CONFIGURE THESE
    # ============================================
    
    INPUT_FOLDER = "Exams/google_vision/math/splited images into sections/exam_1"
    OUTPUT_FOLDER = "Exams/extraction_results"
    
    # ============================================
    # RUN EXTRACTION
    # ============================================
    
    print("🚀 Student Answer Extractor")
    print("🎯 Mode: Extract student answers from التعليمة (ignore السند)")
    print("="*60)
    
    # Check input folder
    if not Path(INPUT_FOLDER).exists():
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        return
    
    # Process all sections
    results = process_all_sections(INPUT_FOLDER)
    
    if results:
        # Save results
        save_results(results, INPUT_FOLDER, OUTPUT_FOLDER)
        
        # Print summary
        print_summary(results)
        
        print(f"\n✅ Extraction complete!")
    else:
        print("\n❌ No results extracted")

if __name__ == "__main__":
    main()