import os
from pathlib import Path

def find_relative_path():
    # Your current location (where you're running the script from)
    current_dir = Path.cwd()  # This will be C:\Users\ayaba\Documents\classquiz pfe\project\Exam-corrector
    print(f"Current directory: {current_dir}")
    
    # Search for PDF files
    print("\nSearching for PDF files...")
    print("=" * 60)
    
    # Start search from project root
    search_root = Path(r"C:\Users\ayaba\Documents\classquiz pfe\project")
    
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = Path(root) / file
                
                # Get relative path from current directory
                try:
                    relative_path = os.path.relpath(full_path, current_dir)
                    print(f"📄 PDF: {file}")
                    print(f"   Full path: {full_path}")
                    print(f"   Relative path (from current dir): {relative_path}")
                    print(f"   To use in your script: r\"{relative_path}\"")
                    print()
                except:
                    pass

if __name__ == "__main__":
    find_relative_path()