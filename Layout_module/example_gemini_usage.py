"""
example_gemini_usage.py
-----------------------
Simple usage example for GeminiImageSplitter.
"""

from pathlib import Path
from Layout_module.gemini_image_splitter import GeminiImageSplitter
from PIL import Image

# Initialize splitter (uses GEMINI_AI_API_KEY from environment)
splitter = GeminiImageSplitter()

# Load your exam image
exam_image = Image.open("path/to/your/exam.jpg")

# Split by keywords (تعليمة, سند)
sections = splitter.split_image_by_keywords(
    image=exam_image,
    min_section_height=100,  # Skip sections smaller than 100px
)

# Print extracted sections
print(f"Found {len(sections)} sections:")
for section in sections:
    keyword = section.keyword_trigger or "header"
    height = section.y_end - section.y_start
    print(f"  Section {section.section_index}: '{keyword}' - {height}px tall")

# Save sections to disk
output_dir = Path("data/Sections/exams")
splitter.save_sections(sections, output_dir, prefix="exam_1")

# Access individual section images
for section in sections:
    # section.image is a PIL Image object
    # You can process it further, e.g.:
    # section.image.show()
    # section.image.save(f"section_{section.section_index}.jpg")
    pass
