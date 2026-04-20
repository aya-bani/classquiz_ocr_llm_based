from pathlib import Path
import os
from dotenv import load_dotenv
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_manager import LoggerManager

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

    # layout_config.py - add this
    MAX_NUMBER_POSITION_RATIO = 0.3  # number must be in first 30% of line width (left side)
                                  # or last 30% (right side for Arabic)
    # OCR and keyword matching
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY_VISION")
    
    # # Gemini API for keyword extraction
    # GEMINI_API_KEY = os.getenv("GEMINI_AI_API_KEY")
    # GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.0-flash")

    # #openai config for fallback LLM usage
    # OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
    # OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

    # SIMILARITY_THRESHOLD = 70
    # KEY_WORDS = ["تعليمة", "سند"]
    # EXCLUDED_KEYWORDS = ["تسند"]
    # layout_config.py - updated section

    SIMILARITY_THRESHOLD = 70
    KEY_WORDS = ["تعليمة", "السند"]
    EXCLUDED_KEYWORDS = ["تسند"]

    # Section number patterns - ONLY valid as section starters
    SECTION_NUMBER_PATTERNS = [
    r"^-\d+$",        # -1 -2 -3  (dash BEFORE)
    r"^\d+-$",        # 1- 2- 3-  (dash AFTER)  ← THIS WAS MISSING
    r"^\d+\.$",       # 1. 2. 3.
    r"^\d+/$",        # 1/ 2/ 3/  (sometimes used in Arabic exams)
    ]

    LINE_Y_TOLERANCE = 20  # px — words within this vertical distance = same line

    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        
        if not cls.GEMINI_API_KEY:
            cls.logger.warning("GEMINI_AI_API_KEY not found - Gemini features disabled")
        
        if not cls.CREDENTIALS_PATH:
            cls.logger.warning("GOOGLE_CREDENTIALS_PATH not found - Google Vision disabled")

        cls.logger.info(
            f"Layout config validated - Gemini: {bool(cls.GEMINI_API_KEY)}, "
            f"GCV: {bool(cls.CREDENTIALS_PATH)}"
        )
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary storage directories"""
        
        cls.RAW_EXAMS_PATH.mkdir(parents=True, exist_ok=True)
        cls.RAW_SUBMISSIONS_PATH.mkdir(parents=True, exist_ok=True)
        cls.logger.info(f"Created directories: {cls.RAW_EXAMS_PATH}, {cls.RAW_SUBMISSIONS_PATH}")