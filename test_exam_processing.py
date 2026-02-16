from pathlib import Path
from ingestion_module import ExamProcessingSystem



Path_to_exams = Path("Exams")
with ExamProcessingSystem(max_workers=6) as processor:
    result = processor.add_exams(Path_to_exams)
    print(result)
