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
    MARKER_SIZE = 60
    MARGIN = 24
    MAX_MARKER_ID = 999
    MAX_PAGES_PER_EXAM = 5  # Maximum pages allowed per exam
    CORNERS_PER_PAGE = 4
    # BLOCK_SIZE: Reserve ID space for up to MAX_PAGES_PER_EXAM dynamic
    # marker IDs (one per actual page) + 3 fixed IDs. Each exam only uses
    # IDs for pages that actually exist, not all MAX_PAGES_PER_EXAM slots.
    BLOCK_SIZE = MAX_PAGES_PER_EXAM + 3  # Reserve 8 IDs per exam
    CORNER_NAMES = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

    # first three markers are always the same across all pages/exams; the
    # generator will only compute a unique fourth ID per page.  These values
    # must fall within the dictionary’s capacity.
    FIXED_MARKER_IDS = [0, 1, 2]

    MAX_EXAMS = (MAX_MARKER_ID + 1) // BLOCK_SIZE
    # coordinate mapper parameters
    DOC_WIDTH = 1654
    DOC_HEIGHT = 2338

    @classmethod
    def create_directories(cls):
        """Create necessary storage directories"""
        logger = LoggerManager.get_logger(__name__)
        cls.MARKED_EXAMS_PATH.mkdir(parents=True, exist_ok=True)
        cls.SCANNED_SUBMISSIONS_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Created directories: {cls.MARKED_EXAMS_PATH}, "
            f"{cls.SCANNED_SUBMISSIONS_PATH}"
        )
