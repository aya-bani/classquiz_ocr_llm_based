from ingestion_module.exams_repository import ExamsRepository
import json
from agents_module.exam_corrector import ExamCorrector
corrector = ExamCorrector()
repo = ExamsRepository()
result = repo.get_exam_content(3)
content = json.dumps(result.get('content'), indent=2, ensure_ascii=False)
import json

with open("submission_19_exam_3_classification.json", "r", encoding="utf-8") as f:
    data = json.load(f)
data = json.dumps(data, indent=2, ensure_ascii=False)
res = corrector.correct_exam(result.get('content'), data)
print(res)