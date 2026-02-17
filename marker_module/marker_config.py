from pathlib import Path
from cv2.aruco import DICT_7X7_1000
from logger_manager import LoggerManager



class MarkerConfig:
    """Configuration for layout module"""
    
    # File storage paths
    BASE_STORAGE_PATH = Path("data")
    MARKED_EXAMS_PATH = BASE_STORAGE_PATH / "Marked_Exams"
    SCANNED_SUBMISSIONS_PATH = BASE_STORAGE_PATH / "Scanned_Exams"

    # Generator parameters
    DICT_TYPE = DICT_7X7_1000
    MARKER_SIZE = 90
    MARGIN = 20
    MAX_MARKER_ID = 999
    PAGES_PER_EXAM = 9
    CORNERS_PER_PAGE = 4
    BLOCK_SIZE = PAGES_PER_EXAM * CORNERS_PER_PAGE  
    CORNER_NAMES = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

    # first three markers are always the same across all pages/exams; the
    # generator will only compute a unique fourth ID per page.  These values
    # must fall within the dictionary’s capacity.
    FIXED_MARKER_IDS = [0, 1, 2]

    MAX_EXAMS = (MAX_MARKER_ID + 1) // BLOCK_SIZE  


    #coordinate mapper parameters
    DOC_WIDTH = 1654
    DOC_HEIGHT = 2338



    @classmethod
    def create_directories(cls):
        """Create necessary storage directories"""
        logger = LoggerManager.get_logger(__name__)
        cls.MARKED_EXAMS_PATH.mkdir(parents=True, exist_ok=True)
        cls.SCANNED_SUBMISSIONS_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directories: {cls.MARKED_EXAMS_PATH}, {cls.SCANNED_SUBMISSIONS_PATH}")
