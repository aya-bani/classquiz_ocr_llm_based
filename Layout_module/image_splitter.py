from google.cloud import vision
from PIL import Image
import numpy as np
from typing import List, Tuple
import io
from rapidfuzz import fuzz
from logger_manager import LoggerManager
from .layout_config import LayoutConfig
class ImageSplitter:
    """
    Splits exam images into exercises using OCR to detect section markers
    """
    
    def __init__(self):
        """
        Initialize the ImageSplitter with Google Vision API credentials
        """
        self.logger = LoggerManager.get_logger(__name__)
        self.client = vision.ImageAnnotatorClient.from_service_account_file(str(LayoutConfig.CREDENTIALS_PATH))
        self.logger.debug("Google Vision client initialized")
        self.logger.info("Initialized ImageSplitter and created directories")

    def is_keyword_match(self, word_text: str) -> bool:
        """
        Check if a word matches any keyword using fuzzy matching
        
        Args:
            word_text: The word to check
            
        Returns:
            True if word matches any keyword above threshold, False otherwise
        """
        for keyword in LayoutConfig.KEY_WORDS :
            # Calculate similarity ratio (0-100)
            similarity = fuzz.ratio(word_text, keyword)
            if similarity >= LayoutConfig.SIMILARITY_THRESHOLD and word_text not in LayoutConfig.EXCLUDED_KEYWORDS:
                self.logger.debug(f"Keyword match: '{word_text}' ~ '{keyword}' ({similarity}%)")
                return True
        return False
    
    def detect_section_lines(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Use OCR to detect words containing section markers
        
        Args:
            image: PIL Image object
            
        Returns:
            List of bounding box coordinates (x_min, y_min, x_max, y_max) for detected sections
        """
        # Convert PIL Image to bytes
        self.logger.info("Running OCR to detect section markers")

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        content = img_byte_arr.getvalue()
        
        # Perform OCR
        vision_image = vision.Image(content=content)
        response = self.client.document_text_detection(image=vision_image)
        doc = response.full_text_annotation
        
        section_coords = []
        print("-----------------OCR Response-------------------")
        # Extract text and bounding boxes at WORD level
        for page in doc.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        # Build word text from symbols
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        print("-----------------Word detected:-------------------")
                        print(word_text)
                        # Check if word matches any section keyword using fuzzy matching
                        if self.is_keyword_match(word_text):
                            # Get bounding box coordinates for this word
                            box = word.bounding_box.vertices
                            x_min = min(v.x for v in box)
                            y_min = min(v.y for v in box)
                            x_max = max(v.x for v in box)
                            y_max = max(v.y for v in box)
                            
                            section_coords.append((x_min, y_min, x_max, y_max))
                            self.logger.info(f"Detected section keyword '{word_text}' at {x_min},{y_min},{x_max},{y_max}")
        
        # Sort by y-coordinate (top to bottom)
        section_coords.sort(key=lambda coord: coord[1])
        self.logger.info(f"Detected {len(section_coords)} section lines")
        return section_coords
    
    def split_image(self, image: Image.Image) -> List[Image.Image]:
        self.logger.info("Splitting image into sections")

        """
        Split image into sections based on detected markers
        
        Args:
            image: PIL Image to split
            
        Returns:
            List of PIL Image objects, one for each section
        """
        # Detect section lines
        line_coords = self.detect_section_lines(image)
        
        if not line_coords:
            # If no sections detected, return the whole image
            self.logger.warning("No section lines detected; returning full image")
            return [image]
        
        # Convert PIL Image to numpy array for processing
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        sections = []
        y_start = 0
        
        for i, coords in enumerate(line_coords):
            x_min, y_min, x_max, y_max = coords
            
            # First slice: from top to the first line's top (exclude first line)
            if i == 0:
                crop = img_array[y_start:y_min, 0:width]
                if crop.size != 0:
                    sections.append(Image.fromarray(crop))
                # Update start to include first line
                y_start = y_min
            else:
                # Slice: from previous line's start to current line's top
                crop = img_array[y_start:y_min, 0:width]
                if crop.size != 0:
                    sections.append(Image.fromarray(crop))
                # Update start to include current line
                y_start = y_min
        
        # Last slice: from the last line's start to the bottom
        crop = img_array[y_start:height, 0:width]
        if crop.size != 0:
            sections.append(Image.fromarray(crop))
            self.logger.debug(f"Created section {len(sections)} from y={y_start} to y={y_min}")
        self.logger.info(f"Total sections created: {len(sections)}")
        return sections
    
    def split_and_save(self, image: Image.Image, exam_id: int, submission_id:int = None) -> List[str]:
        """
        Split image and save sections to disk
        
        Args:
            image: PIL Image to split
            output_prefix: Prefix for saved image files
            
        Returns:
            List of saved file paths
        """
        self.logger.info("Starting split_and_save")
        sections = self.split_image(image)
        
        if submission_id is not None:
            output_prefix = f"exam_{exam_id}_submission_{submission_id}"
            directory = LayoutConfig.RAW_SUBMISSIONS_PATH / output_prefix
        else:
            output_prefix = f"exam_{exam_id}"
            directory = LayoutConfig.RAW_EXAMS_PATH / output_prefix
        directory.mkdir(parents=True, exist_ok=True)
        for i, section in enumerate(sections):
            filepath = directory / f"{output_prefix}_section_{i}.jpg"
            section.save(filepath, "JPEG")
            self.logger.info(f"Saved section {i} to {filepath}")
        self.logger.info("Finished split_and_save")
        return {'sections_dir': directory, 'number_of_sections': len(sections)}

