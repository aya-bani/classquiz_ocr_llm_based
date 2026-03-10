"""
simple_test.py — test the Cloud Vision OCR + GPT-4o pipeline.
Usage:  python agents_module/simple_test.py
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents_module.question_extractor_google_cloud import QuestionExtractorGoogleCloud

IMAGE_PATH = Path(
    r"Exams\google_vision\prod\splited images into sections\exam_1\exam_1_section_2.jpg"
)

with QuestionExtractorGoogleCloud() as extractor:
    result = extractor.process_image(IMAGE_PATH, is_submission=False)

print(f"Type       : {result['question_type']}")
print(f"Confidence : {result['confidence']}")
print(f"Content    : {result.get('content', {})}")