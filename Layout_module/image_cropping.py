from PIL import Image
from .layout_config import LayoutConfig
from logger_manager import LoggerManager
from pdf2image import convert_from_path
from pathlib import Path
class ImageCropping:
    def __init__(self):
        self.logger = LoggerManager.get_logger(__name__)

    def remove_margin_from_image(self, img: Image.Image, first_page: bool = False, is_correction: bool = False):
        width, height = img.size
        margin = min(LayoutConfig.BASE_MARGIN, width // 2, height // 2)
        
        left = margin + LayoutConfig.LEFT_OFFSET
        right = width - margin
        bottom = height - margin - LayoutConfig.BOTTOM_OFFSET

        if first_page:
            top_offset = LayoutConfig.FIRST_PAGE_CORRECTION_OFFSET if is_correction else LayoutConfig.FIRST_PAGE_OFFSET
        else:
            top_offset = 0
        top = margin + top_offset
        self.logger.debug(f"Cropping image: left={left}, top={top}, right={right}, bottom={bottom}")
        return img.crop((left, top, right, bottom))



    def process_pdf_to_single_image(self, correction_path:Path, is_correction:bool = False, save_output:bool= False, output_path:str="output.jpeg") -> Image.Image:
        pages = self._load_pdf(correction_path)
        self.logger.info(f"Processing {len(pages)} pages into a single image")
        cropped_pages = []

        for i,page in enumerate(pages) :
            if i == 0 :
                im = self.remove_margin_from_image(page, first_page=True, is_correction=is_correction)
            else :
                im = self.remove_margin_from_image(page, is_correction=is_correction)

            cropped_pages.append(im)

        widths = [img.width for img in cropped_pages]
        heights = [img.height for img in cropped_pages]

        total_height = sum(heights)
        max_width = max(widths)

        final_image = Image.new("RGB", (max_width, total_height), "white")

        y_offset = 0
        
        for img in cropped_pages:
            final_image.paste(img, (0, y_offset))
            y_offset += img.height

        if save_output:
            final_image.save(output_path, "JPEG")
            self.logger.info(f"Saved merged image to {output_path}")
        return final_image
    
    def _load_pdf(self, pdf_path:Path) -> tuple:
        """Load PDF file and return list of pages"""
        pages = convert_from_path(pdf_path, dpi=300)
        return pages