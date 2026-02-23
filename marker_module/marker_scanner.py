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

        # Separate dynamic markers (which encode exam_id + page_number) from fixed markers
        dynamic_markers = [m for m in detected_markers if m['marker_id'] >= max(MarkerConfig.FIXED_MARKER_IDS) + 1]
        fixed_markers = [m for m in detected_markers if m['marker_id'] < max(MarkerConfig.FIXED_MARKER_IDS) + 1]
        
        cls.logger.debug(f"Detected {len(dynamic_markers)} dynamic + {len(fixed_markers)} fixed markers")

        # Extract exam/page from dynamic markers only (fixed ones return dummy 0,0)
        dynamic_exam_ids = {m['exam_id'] for m in dynamic_markers}
        dynamic_page_numbers = {m['page_number'] for m in dynamic_markers}

        # Validate single exam ID from dynamic markers
        if len(dynamic_exam_ids) > 1:
            cls.logger.error(f"Multiple exam IDs detected in dynamic markers: {dynamic_exam_ids}")
            return cls._create_error_result(
                f'Multiple exam IDs detected: {dynamic_exam_ids}',
                len(ids),
                detected_markers
            )

        # Validate single page number from dynamic markers
        if len(dynamic_page_numbers) > 1:
            cls.logger.error(f"Multiple page numbers detected in dynamic markers: {dynamic_page_numbers}")
            return cls._create_error_result(
                f'Multiple page numbers detected: {dynamic_page_numbers}',
                len(ids),
                detected_markers,
                exam_id=next(iter(dynamic_exam_ids)) if dynamic_exam_ids else None
            )

        # At least one dynamic marker must be present
        if not dynamic_exam_ids or not dynamic_page_numbers:
            cls.logger.error("No dynamic markers detected (critical for exam identification)")
            return cls._create_error_result(
                "No dynamic markers detected. Cannot identify exam/page.",
                len(ids),
                detected_markers
            )

        exam_id = next(iter(dynamic_exam_ids))
        page_number = next(iter(dynamic_page_numbers))
        
        cls.logger.info(f"Scan success exam={exam_id} page={page_number} (found {len(dynamic_markers)} dynamic + {len(fixed_markers)} fixed markers)")

        return {
            'success': True,
            'exam_id': exam_id,
            'page_number': page_number,
            'markers_found': len(ids),
            'dynamic_markers_found': len(dynamic_markers),
            'fixed_markers_found': len(fixed_markers),
            'expected_markers': MarkerConfig.CORNERS_PER_PAGE,
            'detected_markers': detected_markers,
            'corners': corners,
            'all_corners_detected': len(ids) == MarkerConfig.CORNERS_PER_PAGE
        }

    @classmethod
    def _preprocess_image(cls, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to enhance ArUco marker visibility.
        
        Steps:
        - Convert to grayscale
        - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Denoise
        - Sharpen
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Sharpen using unsharp mask
        blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
        
        cls.logger.debug("Preprocessing complete: CLAHE + denoise + sharpen")
        return sharpened

    @classmethod
    def _detect_markers(cls, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect ArUco markers with multiple strategies.
        
        Tries multiple strategies and returns the best result 
        (prioritizing detections that include a dynamic marker).
        
        Returns: (corners, ids) tuple
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(MarkerConfig.DICT_TYPE)
        preprocessed = cls._preprocess_image(image)
        
        strategies = [
            ("raw_strict", image, lambda: cv2.aruco.DetectorParameters()),
            ("prep_strict", preprocessed, lambda: cv2.aruco.DetectorParameters()),
            ("prep_lenient", preprocessed, lambda: cls._create_lenient_params()),
            ("prep_very_lenient", preprocessed, lambda: cls._create_very_lenient_params())
        ]
        
        best_result = ([], None)
        best_has_dynamic = False
        
        for name, img_to_use, create_params in strategies:
            try:
                params = create_params()
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners, ids, _ = detector.detectMarkers(img_to_use)
                
                if ids is not None and len(ids) > 0:
                    # Check if this detection includes a dynamic marker (ID >= 3)
                    first_dynamic = max(MarkerConfig.FIXED_MARKER_IDS) + 1
                    has_dynamic = any(mid >= first_dynamic for mid in ids.flatten())
                    
                    cls.logger.debug(f"[{name}] found {len(ids)} markers, dynamic={has_dynamic}")
                    
                    # Keep this result if:
                    # 1. It has a dynamic marker and best so far doesn't
                    # 2. Or same quality but more markers
                    if (has_dynamic and not best_has_dynamic) or \
                       (has_dynamic == best_has_dynamic and len(ids) > len(best_result[1] or [])):
                        best_result = (corners, ids)
                        best_has_dynamic = has_dynamic
                        cls.logger.debug(f"[{name}] is new best result")
            except Exception as e:
                cls.logger.warning(f"Detection [{name}] failed: {e}")
                continue
        
        if best_result[1] is not None and len(best_result[1]) > 0:
            cls.logger.debug(f"Returning best: {len(best_result[1])} markers, dynamic={best_has_dynamic}")
            return best_result
        
        cls.logger.warning("All detection strategies failed")
        return [], None
    
    @classmethod
    def _create_lenient_params(cls) -> cv2.aruco.DetectorParameters:
        """Create lenient detector parameters."""
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshConstant = 5
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.polygonalApproxAccuracyRate = 0.03
        return params
    
    @classmethod
    def _create_very_lenient_params(cls) -> cv2.aruco.DetectorParameters:
        """Create very lenient detector parameters."""
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshConstant = 2
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 40
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        params.polygonalApproxAccuracyRate = 0.1
        return params

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
        
        The fourth (dynamic) marker is calculated as:
            fourth_id = 3 + (exam_id * PAGES_PER_EXAM) + page_number
        
        To decode, we reverse the formula:
            id_offset = marker_id - 3
            exam_id = id_offset // PAGES_PER_EXAM
            page_number = id_offset % PAGES_PER_EXAM
        
        The fixed markers (0, 1, 2) are just corner indicators and use their
        ID directly as the corner position.
        
        Args:
            marker_id: Integer marker ID (0-999)
            
        Returns:
            Tuple of (exam_id, page_number, corner_name)
            
        Examples:
            marker_id=0  → (dummy, dummy, 'top_left')      # Fixed marker
            marker_id=14 → (2, 1, 'bottom_right') # Dynamic: exam 2, page 1
        """
        # validate range first
        if not (0 <= marker_id <= MarkerConfig.MAX_MARKER_ID):
            cls.logger.error(
                f"marker_id {marker_id} out of valid range 0-{MarkerConfig.MAX_MARKER_ID}"
                )
            raise ValueError(
                f"marker_id must be between 0 and {MarkerConfig.MAX_MARKER_ID}"
                )

        first_dynamic = max(MarkerConfig.FIXED_MARKER_IDS) + 1   
        #First dynamic marker ID is 3
        
        if marker_id < first_dynamic:
            # Fixed marker (0, 1, 2) - corner indicator only
            # Use marker_id directly as corner position
            exam_id = 0  # Dummy value for fixed markers
            page_number = 0  # Dummy value for fixed markers
            corner_id = marker_id
        else:
            # Dynamic marker - extract exam_id and page_number
            id_offset = marker_id - first_dynamic
            exam_id = id_offset // MarkerConfig.PAGES_PER_EXAM
            page_number = id_offset % MarkerConfig.PAGES_PER_EXAM
            corner_id = 3  # Dynamic markers are always at the fourth corner
        
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
    def infer_missing_corner(
        cls, 
        detected_markers: List[Dict], 
        corners_data: List[np.ndarray]
    ) -> Optional[Tuple[str, np.ndarray]]:
        """
        If 3 corners are detected, infer the 4th using geometric reasoning.
        
        Assumes the document outline is roughly rectangular. Uses parallelogram rule:
        if corners A, B, C are known, the 4th corner D ≈ A + C - B
        
        Args:
            detected_markers: List of detected marker info dicts with 'corner' field
            corners_data: List of corner arrays from ArUco detection (same order)
        
        Returns:
            Tuple of (missing_corner_name, inferred_corners_array) or None if can't infer
        """
        if len(detected_markers) != 3:
            return None
        
        corner_names = MarkerConfig.CORNER_NAMES  # ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        detected_corners = {m['corner']: corners_data[i][0] for i, m in enumerate(detected_markers)}
        missing = [c for c in corner_names if c not in detected_corners]
        
        if len(missing) != 1:
            return None
        
        missing_corner = missing[0]
        
        # Use vector parallelogram: if A, B, C known and D missing, D ≈ A + C - B
        # Pick the opposite corner pairs smartly
        try:
            corners = detected_corners
            
            if missing_corner == 'bottom_right':
                # D = TL + BR_estimate where BR_estimate ≈ TR + BL - TL
                tl = corners.get('top_left', corners.get('bottom_left'))
                tr = corners.get('top_right', corners.get('top_left'))
                bl = corners.get('bottom_left', corners.get('top_left'))
                inferred = tr + bl - tl
            elif missing_corner == 'bottom_left':
                # Similar logic
                tl = corners.get('top_left', corners.get('top_right'))
                tr = corners.get('top_right', corners.get('bottom_right'))
                br = corners.get('bottom_right', corners.get('top_right'))
                inferred = tl + br - tr
            elif missing_corner == 'top_right':
                tl = corners.get('top_left', corners.get('bottom_right'))
                bl = corners.get('bottom_left', corners.get('top_left'))
                br = corners.get('bottom_right', corners.get('bottom_left'))
                inferred = tl + br - bl
            elif missing_corner == 'top_left':
                tr = corners.get('top_right', corners.get('bottom_left'))
                bl = corners.get('bottom_left', corners.get('bottom_right'))
                br = corners.get('bottom_right', corners.get('top_right'))
                inferred = tr + bl - br
            else:
                return None
            
            # Create corners array in shape (4, 2)
            inferred_corners = np.array([inferred], dtype=np.float32)
            cls.logger.debug(f"Inferred {missing_corner} corner: {inferred}")
            return (missing_corner, inferred_corners)
        except Exception as e:
            cls.logger.warning(f"Failed to infer missing corner {missing_corner}: {e}")
            return None

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