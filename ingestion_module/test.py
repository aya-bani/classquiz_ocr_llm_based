from pathlib import Path
import sys
from PIL import Image
sys.path.append(str(Path(__file__).resolve().parent.parent))
from exam_processing_system import ExamProcessingSystem

processor = ExamProcessingSystem(max_workers=6)
#submission = SubmissionProcessingSystem()


Path_to_exams = Path("Exams")
with ExamProcessingSystem(max_workers=6) as processor:
    result = processor.add_exams(Path_to_exams)
    print(result)
