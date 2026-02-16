from ingestion_module import SubmissionProcessingSystem
from PIL import Image
from pathlib import Path


pages_submitted = []
paths = [Path("10.jpeg")]
for path in paths:
    img = Image.open(path)
    pages_submitted.append(img)




with SubmissionProcessingSystem(max_workers=6) as submission_processor:
    result = submission_processor.process_submission(0, "test", 0, pages_submitted)
    print(result)