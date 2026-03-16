"""Google Cloud Vision based section extraction pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Optional

from google.cloud import vision

from .utils import get_section_number, load_images, sort_results


class GCVSectionExtractor:
    """Extracts text from exam section images using Google Cloud Vision OCR."""

    def __init__(self, credentials_path: Optional[str] = None) -> None:
        cred_path = credentials_path or os.getenv("GOOGLE_CREDENTIALS_PATH")
        if cred_path and Path(cred_path).exists():
            factory = vision.ImageAnnotatorClient.from_service_account_file
            self.client = factory(cred_path)
        else:
            self.client = vision.ImageAnnotatorClient()

    def extract_folder(self, folder_path: str) -> List[Dict]:
        """Run OCR on all section images and return sorted results."""
        image_paths = load_images(Path(folder_path))
        results: List[Dict] = []
        for image_path in image_paths:
            section_number = get_section_number(image_path.name)
            result = self._extract_single_image(image_path, section_number)
            results.append(result)
        return sort_results(results)

    def _extract_single_image(
        self, image_path: Path, section_number: int
    ) -> Dict:
        """Run GCV document_text_detection on one image."""
        content = image_path.read_bytes()
        image = vision.Image(content=content)
        response = self.client.document_text_detection(image=image)

        full_text = ""
        avg_confidence = 0.0
        annotation = response.full_text_annotation
        if annotation:
            full_text = annotation.text
            confidences = [
                block.confidence
                for page in annotation.pages
                for block in page.blocks
            ]
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

        return {
            "section_number": section_number,
            "question": full_text.strip() if full_text else None,
            "student_answer": None,
            "confidence": round(avg_confidence, 4),
        }
