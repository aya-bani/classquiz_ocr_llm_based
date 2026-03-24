import os
from mistralai.client import Mistral

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

# === CHANGE THIS TO YOUR LOCAL IMAGE PATH ===
local_image_path = "Exams/google_vision/math/splited images into sections/exam_1/exam_1_section_3.jpg"

# Convert local file to base64
import base64
with open(local_image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{encoded_string}"
    },
    include_image_base64=True
)

# === SEE THE OUTPUT ===
print("=== OCR OUTPUT ===\n")

# Print each page's markdown text
for i, page in enumerate(ocr_response.pages):
    print(f"--- Page {i+1} ---")
    print(page.markdown)
    print()

# Also show the structured data (images, tables, etc.)
print("=== FULL RESPONSE STRUCTURE ===")
print(f"Number of pages: {len(ocr_response.pages)}")
for i, page in enumerate(ocr_response.pages):
    print(f"\nPage {i+1}:")
    print(f"  Markdown length: {len(page.markdown)} chars")
    if hasattr(page, 'images') and page.images:
        print(f"  Images found: {len(page.images)}")
    if hasattr(page, 'tables') and page.tables:
        print(f"  Tables found: {len(page.tables)}")