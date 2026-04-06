
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from question_extractor_google_cloud import QuestionExtractorGoogleCloud

IMAGE_PATH = Path(
    r"Exams/google_vision/math/splited images into sections/exam_1/exam_1_section_12.jpg"
)

with QuestionExtractorGoogleCloud() as extractor:
    result = extractor.process_image(IMAGE_PATH, is_submission=False)

print("Type       :", result.get("question_type"))
print("Confidence :", result.get("confidence"))
print("Content    :", result.get("content"))