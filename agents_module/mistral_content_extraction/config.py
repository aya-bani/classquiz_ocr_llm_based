import os
from pathlib import Path

# API Configuration
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    print("❌ MISTRAL_API_KEY not found")
    exit(1)

# Folder paths
INPUT_FOLDER = "Exams/google_vision/math/splited images into sections/exam_1"
OUTPUT_FOLDER = "Exams/extraction_results"

# Create output folder
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Model names
OCR_MODEL = "mistral-ocr-latest"
LLM_MODEL = "mistral-large-latest"

# Extraction settings
TEMPERATURE = 0.0
MAX_TOKENS = 1000