import re
from pathlib import Path

from extractor import extract_student_answer

def extract_section_number(filename: str) -> str:
    numbers = re.findall(r'\d+', filename)
    return numbers[0] if numbers else "unknown"

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