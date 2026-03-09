from pathlib import Path
import sys

# Allow running this file directly: python agents_module\test.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agents_module.question_extractor import QuestionExtractor

extractor = QuestionExtractor()

image_dir = Path("data\Sections\test_output_openai\math1\exam_section_09.jpg")


r = extractor.process_image(image_dir, is_submission=False)
print(r)