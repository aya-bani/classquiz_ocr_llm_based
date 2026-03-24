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

# Get API key from environment
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

# Configure paths - CHANGE THESE
INPUT_FOLDER = "Exams/google_vision/math/splited images into sections/exam_1"
OUTPUT_FOLDER = "Exams/extraction_results"

# Create output folder if it doesn't exist
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_section_number(filename):
    """Extract section number from filename (e.g., 'image_1.png' -> '1')"""
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return numbers[0]
    return "unknown"

def encode_image(image_path):
    """Convert image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image(image_path):
    """Use Mistral OCR to extract text from image"""
    encoded_string = encode_image(image_path)
    
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{encoded_string}"
        },
        include_image_base64=True
    )
    
    # Combine all pages (usually 1 for section images)
    full_text = ""
    for page in ocr_response.pages:
        full_text += page.markdown + "\n"
    
    return full_text.strip()

def extract_structured_content(image_path, custom_prompt=""):
    """
    Extract structured content using OCR + LLM
    Returns JSON with question, type, options, etc.
    """
    try:
        # Step 1: Get raw text from OCR
        print("  📝 OCR in progress...")
        raw_text = extract_text_from_image(image_path)
        
        if not raw_text:
            print("  ⚠️ No text extracted")
            return None
        
        print(f"  📄 Extracted {len(raw_text)} characters")
        
        # Step 2: Use LLM to structure the content
        system_prompt = """You are an expert in Arabic exam content extraction.
Return ONLY valid JSON with no additional text."""

        user_prompt = f"""
Extract structured information from this exam question:

--- TEXT ---
{raw_text}
--- END TEXT ---

Return a JSON object with these exact fields:
{{
    "question_text": "the main question text",
    "answer_type": "multiple_choice" or "essay" or "true_false" or "fill_blank",
    "options": ["option 1", "option 2", ...] (empty list if not multiple choice),
    "difficulty": "easy" or "medium" or "hard",
    "confidence_score": 0.95 (number between 0 and 1)
}}

{custom_prompt}
"""

        # Call Mistral LLM to structure the content
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1  # Low temperature for consistent output
        )
        
        # Parse the JSON response
        response_text = chat_response.choices[0].message.content.strip()
        
        # Clean markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        extracted = json.loads(response_text)
        
        return extracted
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

# ============================================
# MAIN PROCESSING FUNCTION
# ============================================

def process_folder(folder_path, custom_prompt="", file_extensions=['.png', '.jpg', '.jpeg']):
    """
    Process all images in a folder and extract content
    """
    folder = Path(folder_path)
    
    # Get all images
    images = []
    for ext in file_extensions:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    
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
        
        # Extract content
        extracted = extract_structured_content(str(image_path), custom_prompt)
        
        if extracted:
            result = {
                "section_number": section_num,
                "filename": image_path.name,
                "question_text": extracted.get("question_text", ""),
                "answer_type": extracted.get("answer_type", "unknown"),
                "options": extracted.get("options", []),
                "difficulty": extracted.get("difficulty", "medium"),
                "confidence_score": extracted.get("confidence_score", 0.5)
            }
            results.append(result)
            print(f"  ✅ Question: {result['question_text'][:60]}...")
            print(f"  ✅ Confidence: {result['confidence_score']:.0%}")
        else:
            print(f"  ❌ Failed to extract")
        
        print()
    
    return results

# ============================================
# SAVE RESULTS
# ============================================

def save_results(results, folder_path, output_folder):
    """
    Save extraction results to JSON file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare output data
    output_data = {
        "metadata": {
            "source_folder": folder_path,
            "total_sections": len(results),
            "extraction_date": datetime.now().isoformat(),
            "timestamp": timestamp
        },
        "results": results
    }
    
    # Save JSON
    json_file = Path(output_folder) / f"extraction_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: {json_file}")
    
    # Also save a summary CSV for easy viewing
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_file = Path(output_folder) / f"extraction_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"💾 CSV saved to: {csv_file}")
    except ImportError:
        print("💡 Install pandas for CSV export: pip install pandas")
    
    return json_file

def print_summary(results):
    """
    Print summary statistics
    """
    if not results:
        print("\n❌ No results to summarize")
        return
    
    print(f"\n{'='*60}")
    print("📊 EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total sections processed: {len(results)}")
    
    # Count by type
    types = {}
    difficulties = {}
    total_confidence = 0
    
    for r in results:
        types[r['answer_type']] = types.get(r['answer_type'], 0) + 1
        difficulties[r['difficulty']] = difficulties.get(r['difficulty'], 0) + 1
        total_confidence += r['confidence_score']
    
    avg_confidence = total_confidence / len(results) if results else 0
    
    print(f"\n📋 Question Types:")
    for t, count in types.items():
        print(f"  {t}: {count}")
    
    print(f"\n🎯 Difficulty Levels:")
    for d, count in difficulties.items():
        print(f"  {d}: {count}")
    
    print(f"\n✨ Average Confidence: {avg_confidence:.1%}")
    
    print(f"\n📝 Sample Results (first 3):")
    print("-"*60)
    for r in results[:3]:
        print(f"\nSection {r['section_number']}:")
        print(f"  Question: {r['question_text'][:80]}...")
        print(f"  Type: {r['answer_type']}")
        print(f"  Confidence: {r['confidence_score']:.0%}")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """
    Main execution function
    """
    # ============================================
    # CONFIGURE THESE BEFORE RUNNING
    # ============================================
    
    # Path to folder containing section images
    INPUT_FOLDER = "Exams/google_vision/math/splited images into sections/exam_1"
    
    # Output folder for results (defaults to same folder if not specified)
    OUTPUT_FOLDER = "Exams/extraction_results"
    
    # Optional: Add custom prompt for specific extraction rules
    CUSTOM_PROMPT = """
    Important rules:
    - Questions are in Arabic language
    - For multiple choice, extract all options labeled أ, ب, ج, د
    - If the question has sub-parts (a, b, c), note them in the question_text
    - For essay questions, set answer_type as "essay"
    - If handwriting is unclear, lower confidence_score accordingly
    """
    
    # ============================================
    # RUN EXTRACTION
    # ============================================
    
    print("🚀 Arabic Exam Content Extractor")
    print("="*60)
    
    # Check if input folder exists
    if not Path(INPUT_FOLDER).exists():
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        print("Please update INPUT_FOLDER path")
        return
    
    # Process all images
    results = process_folder(INPUT_FOLDER, CUSTOM_PROMPT)
    
    if results:
        # Save results
        save_results(results, INPUT_FOLDER, OUTPUT_FOLDER)
        
        # Print summary
        print_summary(results)
        
        print(f"\n✅ Extraction complete!")
    else:
        print("\n❌ No results extracted. Check your images and try again.")

# ============================================
# RUN THE SCRIPT
# ============================================

if __name__ == "__main__":
    main()