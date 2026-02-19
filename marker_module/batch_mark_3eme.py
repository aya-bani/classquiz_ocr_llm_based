"""
Batch script to mark all 3ème year exams (math, science, arabe)
excluding correction files (those starting with 'corr').

Usage:
    python batch_mark_3eme.py
"""

import sys
from pathlib import Path

# Ensure package imports work when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from marker_module.mark_pdf import mark_pdf


def batch_mark_3eme_exams(start_exam_id: int = 200):
    """
    Mark all 3ème year exam PDFs (excluding corrections) with sequential exam IDs.
    
    Args:
        start_exam_id: Starting exam ID (default: 200)
    """
    base_dir = Path(__file__).parent.parent / "Exams" / "3ème année"
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)
    
    # Paths for each subject
    subjects = {
        'math': base_dir / 'math',
        'science': base_dir / 'science',
        'arabe': base_dir / 'arabe'
    }
    
    # Collect all PDFs to mark (excluding those starting with 'corr')
    pdfs_to_mark = []
    
    for subject, subject_path in subjects.items():
        pdf_files = sorted(subject_path.glob("*.pdf"))
        # Filter out files starting with 'corr'
        non_corr_files = [f for f in pdf_files if not f.name.lower().startswith('corr')]
        
        for pdf_file in non_corr_files:
            pdfs_to_mark.append({
                'path': pdf_file,
                'subject': subject
            })
    
    if not pdfs_to_mark:
        print("No PDF files found (or all files are corrections)")
        sys.exit(1)
    
    print(f"Found {len(pdfs_to_mark)} exam PDF(s) to mark")
    print(f"Starting exam_id: {start_exam_id}\n")
    
    # Mark each PDF with sequential exam IDs
    results = []
    current_exam_id = start_exam_id
    
    for i, item in enumerate(pdfs_to_mark):
        pdf_file = item['path']
        subject = item['subject']
        exam_id = current_exam_id + i
        
        print(f"[{i+1}/{len(pdfs_to_mark)}] [{subject.upper():7s}] Marking {pdf_file.name}...")
        
        try:
            output_path = mark_pdf(str(pdf_file), exam_id)
            results.append({
                'input': pdf_file.name,
                'subject': subject,
                'exam_id': exam_id,
                'output': output_path,
                'status': 'SUCCESS'
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'input': pdf_file.name,
                'subject': subject,
                'exam_id': exam_id,
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*80)
    print("3ÈME YEAR BATCH MARKING SUMMARY")
    print("="*80)
    
    for result in results:
        status = result['status']
        exam_id = result['exam_id']
        subject = result['subject'].upper()
        input_file = result['input']
        
        if status == 'SUCCESS':
            output = result['output']
            print(f"✓ exam_id={exam_id:3d} | {subject:8s} | {input_file:30s} → {output.name}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"✗ exam_id={exam_id:3d} | {subject:8s} | {input_file:30s} | ERROR: {error}")
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\nTotal: {success_count}/{len(pdfs_to_mark)} marked successfully")
    
    # Summary by subject
    print("\nBy subject:")
    for subject in ['math', 'science', 'arabe']:
        subject_results = [r for r in results if r['subject'] == subject]
        if subject_results:
            success = sum(1 for r in subject_results if r['status'] == 'SUCCESS')
            print(f"  {subject.upper():8s}: {success}/{len(subject_results)} ✓")


if __name__ == "__main__":
    start_exam_id = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    batch_mark_3eme_exams(start_exam_id)
