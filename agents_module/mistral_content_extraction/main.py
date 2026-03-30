from pathlib import Path

from config import INPUT_FOLDER, OUTPUT_FOLDER
from processor import process_all_sections
from saver import save_results, print_summary

def main():
    print("🚀 Student Answer Extractor - Clean Output")
    print("="*50)
    print("✓ NO pipe symbols |")
    print("✓ Clean operators: - + x / =")
    print("✓ Western digits only (0-9)")
    print("✓ Preserves Arabic RTL math order")
    print("✓ Optimized for children's handwriting (ages 5-10)")
    print("="*50)
    
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