import json
from pathlib import Path
from datetime import datetime

def save_results(results: list, folder_path: str, output_folder: str) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output = {
        "metadata": {
            "source": folder_path,
            "total": len(results),
            "date": datetime.now().isoformat(),
            "note": "Clean output: NO pipe symbols | NO wrong operators | RTL math preserved"
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