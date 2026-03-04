"""
gemini_image_splitter.py
-----------------------
Splits exam pages into sections using Gemini Vision API for keyword extraction.

Workflow:
1. Extract text with bounding boxes from image using Gemini
2. Find target keywords (تعليمة, سند) in extracted text
3. Determine section boundaries based on keyword Y-coordinates
4. Split image vertically into sections
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_manager import LoggerManager
from Layout_module.layout_config import LayoutConfig


@dataclass
class KeywordMatch:
    """Detected keyword with position in image"""
    keyword: str
    text: str  # full text block containing keyword
    y_position: int  # vertical position (top of bounding box)
    confidence: float = 1.0


@dataclass
class ImageSection:
    """A section of the exam page"""
    section_index: int
    keyword_trigger: Optional[str]  # keyword that started this section
    y_start: int
    y_end: int
    image: Image.Image


class GeminiImageSplitter:
    """
    Splits exam images into sections using Gemini Vision API.
    
    Each section starts with a keyword from KEY_WORDS (تعليمة, سند).
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.logger = LoggerManager.get_logger(__name__)
        
        self.api_key = api_key or LayoutConfig.GEMINI_API_KEY
        self.model_name = model_name or LayoutConfig.GEMINI_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("GEMINI_AI_API_KEY is required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        self.keywords = LayoutConfig.KEY_WORDS
        self.excluded_keywords = LayoutConfig.EXCLUDED_KEYWORDS
        
        self.logger.info(
            f"GeminiImageSplitter initialized with model={self.model_name}, "
            f"keywords={self.keywords}"
        )

    def split_image_by_keywords(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        min_section_height: int = 100,
    ) -> List[ImageSection]:
        """
        Split image into sections, each starting with a keyword.
        
        Args:
            image: Input image (PIL, numpy, or path)
            min_section_height: Minimum height for a section (px)
        
        Returns:
            List of ImageSection objects
        """
        pil_image = self._normalize_image(image)
        width, height = pil_image.size
        
        self.logger.info(f"Splitting image ({width}x{height}) by keywords")
        
        # Extract text with positions using Gemini
        keyword_matches = self._extract_keywords_with_positions(pil_image)
        
        if not keyword_matches:
            self.logger.warning("No keywords found - returning full image as single section")
            return [
                ImageSection(
                    section_index=0,
                    keyword_trigger=None,
                    y_start=0,
                    y_end=height,
                    image=pil_image,
                )
            ]
        
        # Sort matches by vertical position
        keyword_matches.sort(key=lambda m: m.y_position)
        
        self.logger.info(
            f"Found {len(keyword_matches)} keyword matches: "
            f"{[m.keyword for m in keyword_matches]}"
        )
        
        # Determine section boundaries
        sections: List[ImageSection] = []
        
        for i, match in enumerate(keyword_matches):
            y_start = match.y_position
            
            # Find end of this section (start of next keyword or image bottom)
            if i + 1 < len(keyword_matches):
                y_end = keyword_matches[i + 1].y_position
            else:
                y_end = height
            
            section_height = y_end - y_start
            
            # Skip tiny sections
            if section_height < min_section_height:
                self.logger.debug(
                    f"Skipping tiny section (height={section_height}px) at y={y_start}"
                )
                continue
            
            # Crop section from image
            section_img = pil_image.crop((0, y_start, width, y_end))
            
            sections.append(
                ImageSection(
                    section_index=len(sections),
                    keyword_trigger=match.keyword,
                    y_start=y_start,
                    y_end=y_end,
                    image=section_img,
                )
            )
        
        self.logger.info(f"Created {len(sections)} sections")
        return sections

    def _extract_keywords_with_positions(
        self, image: Image.Image
    ) -> List[KeywordMatch]:
        """
        Extract keywords and their vertical positions using Gemini Vision.
        
        Gemini Flash 2.0 doesn't provide exact bounding boxes, so we use a
        text-extraction approach:
        1. Ask Gemini to extract all text from image
        2. Parse the response to find keyword occurrences
        3. Estimate Y-position based on text flow (top-to-bottom)
        """
        self.logger.debug("Extracting text from image with Gemini Vision")
        
        prompt = f"""
Extract ALL text from this Arabic exam page, preserving the vertical order.
For each text block you find, output it on a new line with a line number.

Focus on finding these specific keywords:
{', '.join(self.keywords)}

Exclude any text containing:
{', '.join(self.excluded_keywords)}

Output format:
Line 1: [first text block]
Line 2: [second text block]
...

Be thorough - extract every piece of text you can see.
"""

        try:
            response = self.model.generate_content([prompt, image])
            time.sleep(0.5)  # Rate limiting
            
            if not response or not response.text:
                self.logger.warning("Gemini returned empty response")
                return []
            
            extracted_text = response.text
            self.logger.debug(f"Gemini response:\n{extracted_text[:500]}...")
            
            return self._parse_keywords_from_text(extracted_text, image.height)
        
        except Exception as exc:
            self.logger.error(f"Gemini API call failed: {exc}")
            return []

    def _parse_keywords_from_text(
        self, extracted_text: str, image_height: int
    ) -> List[KeywordMatch]:
        """
        Parse Gemini's text output to find keywords and estimate positions.
        
        Since we don't have exact bounding boxes, we estimate Y-position
        based on line number in the extracted text (assuming top-to-bottom flow).
        """
        matches: List[KeywordMatch] = []
        lines = extracted_text.strip().split('\n')
        
        for line_idx, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Check if this line contains any keyword
            for keyword in self.keywords:
                if keyword in line_clean:
                    # Check for excluded keywords
                    if any(excl in line_clean for excl in self.excluded_keywords):
                        self.logger.debug(
                            f"Excluded keyword found in line {line_idx}: {line_clean}"
                        )
                        continue
                    
                    # Estimate Y-position based on line number
                    # Assume uniform text distribution across image height
                    estimated_y = int((line_idx / max(len(lines), 1)) * image_height)
                    
                    matches.append(
                        KeywordMatch(
                            keyword=keyword,
                            text=line_clean,
                            y_position=estimated_y,
                        )
                    )
                    self.logger.debug(
                        f"Found '{keyword}' at line {line_idx} "
                        f"(est. y={estimated_y}): {line_clean[:50]}"
                    )
        
        # Remove duplicate positions (same keyword appearing multiple times close together)
        matches = self._deduplicate_nearby_matches(matches, proximity_threshold=50)
        
        return matches

    @staticmethod
    def _deduplicate_nearby_matches(
        matches: List[KeywordMatch], proximity_threshold: int = 50
    ) -> List[KeywordMatch]:
        """
        Remove duplicate keyword matches that are too close vertically.
        Keeps the first occurrence.
        """
        if not matches:
            return []
        
        matches_sorted = sorted(matches, key=lambda m: m.y_position)
        deduplicated = [matches_sorted[0]]
        
        for match in matches_sorted[1:]:
            last_y = deduplicated[-1].y_position
            if abs(match.y_position - last_y) > proximity_threshold:
                deduplicated.append(match)
        
        return deduplicated

    @staticmethod
    def _normalize_image(
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Convert various image formats to PIL RGB Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return Image.open(path).convert("RGB")
        
        if isinstance(image, np.ndarray):
            # Assume BGR from OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            return Image.fromarray(image_rgb).convert("RGB")
        
        raise TypeError(f"Unsupported image type: {type(image)}")

    def save_sections(
        self,
        sections: List[ImageSection],
        output_dir: Union[str, Path],
        prefix: str = "section",
    ) -> List[Path]:
        """
        Save image sections to disk.
        
        Args:
            sections: List of ImageSection objects
            output_dir: Output directory path
            prefix: Filename prefix
        
        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths: List[Path] = []
        
        for section in sections:
            keyword_tag = section.keyword_trigger or "header"
            filename = f"{prefix}_{section.section_index:02d}_{keyword_tag}.jpg"
            file_path = output_path / filename
            
            section.image.save(file_path, "JPEG", quality=95)
            saved_paths.append(file_path)
            
            self.logger.debug(
                f"Saved section {section.section_index} "
                f"(y={section.y_start}-{section.y_end}) to {file_path}"
            )
        
        self.logger.info(f"Saved {len(saved_paths)} sections to {output_path}")
        return saved_paths
