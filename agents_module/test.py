import json
import sys
from agents_module.question_extractor import QuestionClassifier


classifier = QuestionClassifier()

from pathlib import Path

image_dir = Path("exam_20/exam_20_section_17.jpg")


r = classifier.process_image(image_dir)
print(r)