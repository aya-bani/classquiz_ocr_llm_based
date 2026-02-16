from logger_manager import LoggerManager
from .image_cropping import ImageCropping
from .image_splitter import ImageSplitter
from .layout_config import LayoutConfig
from pathlib import Path


class LayoutManager:
    def __init__(self):
        LayoutConfig.validate()
        LayoutConfig.create_directories()
        self.logger = LoggerManager.get_logger(__name__)
        self.logger.info("LayoutManager initialized with provided configuration")
        self.cropping_tool = ImageCropping()
        self.splitter_tool = ImageSplitter()


    def process_correction(self, exam_id: int, correction_path: Path) -> list:
        self.logger.info(f"Processing correction for exam_id={exam_id}")
        merged_image = self.cropping_tool.process_pdf_to_single_image(
            correction_path, is_correction=True, save_output=False
        )
        self.logger.debug("Merged image created for correction")
        section_paths = self.splitter_tool.split_and_save(
            merged_image, exam_id
        )
        self.logger.info(f"Processed correction into {len(section_paths)} sections")
        return section_paths

    def process_submission(self, exam_id: int, submission_id: int, submission_path: Path) -> list:
        self.logger.info(f"Processing submission for exam_id={exam_id}, submission_id={submission_id}")
        merged_image = self.cropping_tool.process_pdf_to_single_image(
            submission_path, is_correction=False, save_output=False
        )
        self.logger.debug("Merged image created for submission")
        section_paths = self.splitter_tool.split_and_save(
            merged_image, exam_id, submission_id
        )
        print("-------------------------------Section paths:-----------------------------------")
        print(section_paths)
        self.logger.info(f"Processed submission into {len(section_paths)} sections")
        return section_paths
    
