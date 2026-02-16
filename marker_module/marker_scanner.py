import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .marker_config import MarkerConfig
from logger_manager import LoggerManager
from PIL import Image

class ExamScanner:
    """
    Stateless exam page scanner that identifies exam and page numbers using ArUco markers.
    """

    logger = LoggerManager.get_logger(__name__)

    @classmethod
    def scan_page(cls, image: np.ndarray) -> Dict:
        """
        Scan a single page for ArUco markers and extract exam/page information.
        
        Args:
            image: Input image as numpy array (BGR or grayscale)
            
        Returns:
            Dictionary containing scan results with keys:
                - success: bool
                - exam_id: Optional[int]
                - page_number: Optional[int]
                - markers_found: int
                - error: Optional[str]
                - detected_markers: List[Dict]
        """
        cls.logger.debug("Scanning page")

        #gray = cls._convert_to_grayscale(image)
        corners, ids = cls._detect_markers(image)

        if ids is None or len(ids) == 0:
            return cls._create_error_result("No markers detected", 0)

        detected_markers, exam_ids, page_numbers = cls._process_markers(ids)

        # Validate single exam ID
        if len(exam_ids) > 1:
            cls.logger.error(f"Multiple exam IDs detected: {exam_ids}")
            return cls._create_error_result(
                f'Multiple exam IDs detected: {exam_ids}',
                len(ids),
                detected_markers
            )

        # Validate single page number
        if len(page_numbers) > 1:
            cls.logger.error(f"Multiple page numbers detected: {page_numbers}")
            return cls._create_error_result(
                f'Multiple page numbers detected: {page_numbers}',
                len(ids),
                detected_markers,
                exam_id=next(iter(exam_ids))
            )
        if not exam_ids or not page_numbers:
            cls.logger.error("No valid exam ID or page number decoded from markers")
            return cls._create_error_result(
                "Could not decode valid exam ID or page number from detected markers",
                len(ids),
                detected_markers
            )
        exam_id = next(iter(exam_ids))
        page_number = next(iter(page_numbers))
        
        cls.logger.info(f"Scan success exam={exam_id} page={page_number}")

        return {
            'success': True,
            'exam_id': exam_id,
            'page_number': page_number,
            'markers_found': len(ids),
            'expected_markers': MarkerConfig.CORNERS_PER_PAGE,
            'detected_markers': detected_markers,
            'corners': corners,
            'all_corners_detected': len(ids) == MarkerConfig.CORNERS_PER_PAGE
        }

    @classmethod
    def _convert_to_grayscale(cls, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @classmethod
    def _detect_markers(cls, gray_image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """Detect ArUco markers in grayscale image."""
        aruco_dict = cv2.aruco.getPredefinedDictionary(MarkerConfig.DICT_TYPE)
        detector = cv2.aruco.ArucoDetector(
            aruco_dict,
            cv2.aruco.DetectorParameters()
        )
        corners, ids, _ = detector.detectMarkers(gray_image)
        return corners, ids

    @classmethod
    def _process_markers(cls, ids: np.ndarray) -> Tuple[List[Dict], set, set]:
        """Process detected marker IDs and extract exam/page information."""
        detected_markers = []
        exam_ids = set()
        page_numbers = set()

        for marker_id in ids.flatten():
            exam_id, page_num, corner = cls.decode_marker(int(marker_id))
            exam_ids.add(exam_id)
            page_numbers.add(page_num)

            detected_markers.append({
                'marker_id': int(marker_id),
                'exam_id': exam_id,
                'page_number': page_num,
                'corner': corner
            })

        return detected_markers, exam_ids, page_numbers

    @classmethod
    def _create_error_result(
        cls, 
        error: str, 
        markers_found: int,
        detected_markers: Optional[List[Dict]] = None,
        exam_id: Optional[int] = None
    ) -> Dict:
        """Create standardized error result dictionary."""
        result = {
            'success': False,
            'error': error,
            'exam_id': exam_id,
            'page_number': None,
            'markers_found': markers_found
        }
        if detected_markers is not None:
            result['detected_markers'] = detected_markers
        
        if markers_found == 0:
            cls.logger.warning(error)
        
        return result

    @classmethod
    def decode_marker(cls, marker_id: int) -> Tuple[int, int, str]:
        """
        Decode a marker ID into exam ID, page number, and corner position.
        
        Args:
            marker_id: Integer marker ID
            
        Returns:
            Tuple of (exam_id, page_number, corner_name)
        """
        exam_id = marker_id // MarkerConfig.BLOCK_SIZE
        remainder = marker_id % MarkerConfig.BLOCK_SIZE

        page_number = remainder // MarkerConfig.CORNERS_PER_PAGE
        corner_id = remainder % MarkerConfig.CORNERS_PER_PAGE

        corner_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        return exam_id, page_number, corner_names[corner_id]

    @classmethod
    def scan_multiple_pages(cls, images: List[Image.Image]) -> List[Dict]:
        """
        Scan multiple pages and return results for each.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of scan result dictionaries
        """
        cls.logger.info(f"Scanning {len(images)} pages")
        results = []
        
        for i, img in enumerate(images):
            img_array = cls._pil_to_opencv(img)
            result = cls.scan_page(img_array)
            result['image_index'] = i
            results.append(result)
        
        return results

    @classmethod
    def _pil_to_opencv(cls, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)."""
        img_array = np.array(pil_image)
        
        # Convert RGB to BGR if needed (PIL uses RGB, OpenCV uses BGR)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array

    @classmethod
    def organize_by_page(cls, scan_results: List[Dict]) -> Dict[int, Dict[int, Dict]]:
        """
        Organize scan results by exam ID and page number.
        """
        organized = {}
        failed_scans = 0

        for result in scan_results:
            if not result['success']:
                failed_scans += 1
                continue

            exam_id = result.get('exam_id')
            page_number = result.get('page_number')
            
            if exam_id is None or page_number is None:
                cls.logger.warning(f"Missing exam_id or page_number in result: {result}")
                continue
            
            if exam_id not in organized:
                organized[exam_id] = {}
            
            if page_number in organized[exam_id]:
                cls.logger.warning(
                    f"Duplicate page {page_number} found for exam {exam_id}"
                )
            
            organized[exam_id][page_number] = result

        cls.logger.info(
            f"Organized {len(scan_results)} pages: "
            f"{sum(len(pages) for pages in organized.values())} successful, "
            f"{failed_scans} failed"
        )
        return organized