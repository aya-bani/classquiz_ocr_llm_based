from pathlib import Path
from logger_manager import LoggerManager
import os
from dotenv import load_dotenv

# Load environment once at module level
load_dotenv()
class LayoutConfig:
    """Configuration for layout module"""
    logger = LoggerManager.get_logger(__name__)
    # File storage paths
    BASE_STORAGE_PATH = Path("data")
    RAW_EXAMS_PATH = BASE_STORAGE_PATH / "Sections" / "exams"
    RAW_SUBMISSIONS_PATH = BASE_STORAGE_PATH / "Sections" / "submissions"

    # Cropping parameters
    BASE_MARGIN = 120
    LEFT_OFFSET = 250
    BOTTOM_OFFSET = 50
    FIRST_PAGE_CORRECTION_OFFSET = 450
    FIRST_PAGE_OFFSET = 560


    # OCR and keyword matching
    CREDENTIALS_PATH = Path(os.getenv("GOOGLE_CREDENTIALS_PATH"))

    SIMILARITY_THRESHOLD = 70
    KEY_WORDS = ["تعليمة", "سند"] 
    EXCLUDED_KEYWORDS = ["تسند"]

    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        
        if not cls.CREDENTIALS_PATH:
            cls.logger.error("GOOGLE_CREDENTIALS_PATH not found in environment variables")
            raise ValueError("GOOGLE_CREDENTIALS_PATH is required")

        cls.logger.info(
            f"Layout config validated - Credentials: {cls.CREDENTIALS_PATH}"
        )
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary storage directories"""
        
        cls.RAW_EXAMS_PATH.mkdir(parents=True, exist_ok=True)
        cls.RAW_SUBMISSIONS_PATH.mkdir(parents=True, exist_ok=True)
        cls.logger.info(f"Created directories: {cls.RAW_EXAMS_PATH}, {cls.RAW_SUBMISSIONS_PATH}")