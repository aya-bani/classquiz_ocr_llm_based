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
        
        Implements the full 7-step workflow:
        1. Input image
        2. Preprocess the image (grayscale, denoise, enhance contrast)
        3. Perform strict ArUco marker detection using predefined dictionary
        4. If fewer than 4 markers are detected:
           → Switch to a more lenient detection mode
        5. If markers are still incomplete:
           → Apply geometric inference to estimate missing markers
           → Use parallelogram-based corner estimation
        6. If at least 2 markers are detected:
           → Estimate the paper region from detected markers
           → Apply perspective transformation to straighten the page
           → Re-run marker detection on the warped image for better accuracy
        7. Return final structured output:
           - exam_id, page_number, detected_markers, paper_corners, success flag
        
        Args:
            image: Input image as numpy array (BGR or grayscale)
            
        Returns:
            Dictionary containing scan results with keys:
                - success: bool
                - exam_id: Optional[int]
                - page_number: Optional[int]
                - markers_found: int
                - dynamic_markers_found: int
                - fixed_markers_found: int
                - detected_markers: List[Dict]
                - paper_corners: Optional[np.ndarray]
                - warped_markers: Optional[List[Dict]]
                - error: Optional[str]
        """
        cls.logger.debug("Starting full scan workflow")

        # Step 1-2: Input & Preprocess
        preprocessed = cls._preprocess_image(image)

        # Step 3-4: Detection with fallback strategy (strict → lenient)
        corners, ids = cls._detect_markers_with_fallback(image, preprocessed)
        cls.logger.debug(f"Step 3-4: Detection found {len(ids) if ids is not None else 0} markers")

        # No markers at all = failure
        if ids is None or len(ids) == 0:
            return cls._create_error_result("No markers detected", 0)

        # Process detected markers with corner data
        detected_markers, corners_list, exam_ids, page_numbers = cls._process_markers_with_corners(ids, corners)
        cls.logger.debug(f"Processed {len(detected_markers)} markers")

        # Step 5: Geometric inference if incomplete
        inferred_marker = None
        if len(detected_markers) < 4 and len(detected_markers) >= 3:
            cls.logger.debug("Step 5: Attempting geometric inference")
            inferred = cls._infer_missing_corner(detected_markers, corners_list)
            if inferred is not None:
                missing_corner, inferred_corner_array = inferred
                detected_markers.append({
                    'marker_id': -1,
                    'exam_id': next(iter(exam_ids)) if exam_ids else 0,
                    'page_number': next(iter(page_numbers)) if page_numbers else 0,
                    'corner': missing_corner,
                    'inferred': True
                })
                corners_list.append(inferred_corner_array)
                inferred_marker = missing_corner
                cls.logger.debug(f"Step 5: Inferred corner {missing_corner}")

        # Validate exam/page consistency from dynamic (non-fixed) real markers only.
        # Fixed markers (IDs 0,1,2) intentionally carry dummy exam/page values.
        first_dynamic = max(MarkerConfig.FIXED_MARKER_IDS) + 1
        real_markers = [m for m in detected_markers if 'inferred' not in m]
        dynamic_markers = [m for m in real_markers if m.get('marker_id', -1) >= first_dynamic]
        fixed_markers = [m for m in real_markers if m.get('marker_id', -1) < first_dynamic]

        dynamic_exam_ids = {m['exam_id'] for m in dynamic_markers}
        dynamic_page_numbers = {m['page_number'] for m in dynamic_markers}

        if len(dynamic_exam_ids) > 1:
            cls.logger.error(f"Multiple exam IDs detected: {dynamic_exam_ids}")
            return cls._create_error_result(
                f'Multiple exam IDs detected: {dynamic_exam_ids}',
                len(detected_markers),
                detected_markers
            )

        if len(dynamic_page_numbers) > 1:
            cls.logger.error(f"Multiple page numbers detected: {dynamic_page_numbers}")
            return cls._create_error_result(
                f'Multiple page numbers detected: {dynamic_page_numbers}',
                len(detected_markers),
                detected_markers,
                exam_id=next(iter(dynamic_exam_ids)) if dynamic_exam_ids else None
            )

        if not dynamic_exam_ids or not dynamic_page_numbers:
            cls.logger.error("No dynamic marker detected to decode exam/page")
            return cls._create_error_result(
                "Could not decode valid exam ID or page number from dynamic markers",
                len(detected_markers),
                detected_markers
            )

        exam_id = next(iter(dynamic_exam_ids))
        page_number = next(iter(dynamic_page_numbers))

        # Step 6: Perspective warping + re-detection if at least 2 markers
        warped_markers = None
        paper_corners = None
        if len(detected_markers) >= 2:
            cls.logger.debug("Step 6: Estimating paper region and applying perspective transform")
            paper_corners = cls._estimate_paper_region(detected_markers, corners_list)
            if paper_corners is not None:
                warped_image = cls._apply_perspective_transform(image, paper_corners)
                if warped_image is not None:
                    # Re-run detection on warped image
                    warped_preprocessed = cls._preprocess_image(warped_image)
                    re_corners, re_ids = cls._detect_markers_with_fallback(warped_image, warped_preprocessed)
                    
                    if re_ids is not None and len(re_ids) > 0:
                        warped_markers, _, _, _ = cls._process_markers_with_corners(re_ids, re_corners)
                        cls.logger.debug(f"Step 6: Re-detection found {len(warped_markers)} markers on warped image")

        # Step 7: Return final structured output
        result = {
            'success': True,
            'exam_id': exam_id,
            'page_number': page_number,
            'markers_found': len(detected_markers),
            'dynamic_markers_found': len(dynamic_markers),
            'fixed_markers_found': len(fixed_markers),
            'expected_markers': MarkerConfig.CORNERS_PER_PAGE,
            'detected_markers': detected_markers,
            'paper_corners': paper_corners,
            'warped_markers': warped_markers,
            'inferred_corner': inferred_marker,
            'all_corners_detected': len(detected_markers) == MarkerConfig.CORNERS_PER_PAGE
        }
        
        cls.logger.info(
            f"Scan success exam={exam_id} page={page_number} "
            f"(found {len(detected_markers)} markers, inferred={inferred_marker is not None})"
        )
        
        return result

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
    def _detect_markers_strict(cls, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """Detect ArUco markers with strict parameters."""
        aruco_dict = cv2.aruco.getPredefinedDictionary(MarkerConfig.DICT_TYPE)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(image)
        return corners, ids

    @classmethod
    def _detect_markers_lenient(cls, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """Detect ArUco markers with lenient parameters."""
        aruco_dict = cv2.aruco.getPredefinedDictionary(MarkerConfig.DICT_TYPE)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshConstant = 5
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.polygonalApproxAccuracyRate = 0.03
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(image)
        return corners, ids

    @classmethod
    def _detect_markers_with_fallback(cls, image: np.ndarray, preprocessed: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect markers with multi-strategy fallback (Step 3-4 combined).
        
        Tries: raw_strict → prep_strict → prep_lenient → prep_very_lenient
        Prioritizes detections that include dynamic markers.
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(MarkerConfig.DICT_TYPE)
        
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
        """Process detected marker IDs and extract exam/page information (legacy)."""
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
    def _process_markers_with_corners(
        cls, 
        ids: np.ndarray, 
        corners: List
    ) -> Tuple[List[Dict], List[np.ndarray], set, set]:
        """
        Process detected marker IDs with their corner coordinates.
        
        Returns:
            (detected_markers, corners_list, exam_ids, page_numbers)
        """
        detected_markers = []
        corners_list = []
        exam_ids = set()
        page_numbers = set()

        for i, marker_id in enumerate(ids.flatten()):
            exam_id, page_num, corner = cls.decode_marker(int(marker_id))
            exam_ids.add(exam_id)
            page_numbers.add(page_num)

            detected_markers.append({
                'marker_id': int(marker_id),
                'exam_id': exam_id,
                'page_number': page_num,
                'corner': corner
            })
            corners_list.append(corners[i])

        return detected_markers, corners_list, exam_ids, page_numbers

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
            fourth_id = 3 + (exam_id * MAX_PAGES_PER_EXAM) + page_number
        
        To decode, we reverse the formula:
            id_offset = marker_id - 3
            exam_id = id_offset // MAX_PAGES_PER_EXAM
            page_number = id_offset % MAX_PAGES_PER_EXAM
        
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
            exam_id = id_offset // MarkerConfig.MAX_PAGES_PER_EXAM
            page_number = id_offset % MarkerConfig.MAX_PAGES_PER_EXAM
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
    def _infer_missing_corner(
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
        
        corner_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        detected_corners = {m['corner']: corners_data[i][0] for i, m in enumerate(detected_markers)}
        missing = [c for c in corner_names if c not in detected_corners]
        
        if len(missing) != 1:
            return None
        
        missing_corner = missing[0]
        
        try:
            corners = detected_corners
            
            if missing_corner == 'bottom_right':
                tl = corners.get('top_left')
                tr = corners.get('top_right')
                bl = corners.get('bottom_left')
                if tl is not None and tr is not None and bl is not None:
                    inferred = tr + bl - tl
                else:
                    return None
            elif missing_corner == 'bottom_left':
                tl = corners.get('top_left')
                tr = corners.get('top_right')
                br = corners.get('bottom_right')
                if tl is not None and tr is not None and br is not None:
                    inferred = tl + br - tr
                else:
                    return None
            elif missing_corner == 'top_right':
                tl = corners.get('top_left')
                bl = corners.get('bottom_left')
                br = corners.get('bottom_right')
                if tl is not None and bl is not None and br is not None:
                    inferred = tl + br - bl
                else:
                    return None
            elif missing_corner == 'top_left':
                tr = corners.get('top_right')
                bl = corners.get('bottom_left')
                br = corners.get('bottom_right')
                if tr is not None and bl is not None and br is not None:
                    inferred = tr + bl - br
                else:
                    return None
            else:
                return None
            
            # Create corners array in shape (1, 4, 2) to match ArUco output format
            inferred_corners = np.array([[inferred]], dtype=np.float32)
            cls.logger.debug(f"Inferred {missing_corner} corner: {inferred}")
            return (missing_corner, inferred_corners)
        except Exception as e:
            cls.logger.warning(f"Failed to infer missing corner {missing_corner}: {e}")
            return None

    @classmethod
    def _estimate_paper_region(
        cls, 
        detected_markers: List[Dict], 
        corners_data: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Estimate the four corners of the paper region from detected markers.
        
        Maps marker corners to paper corners:
        - top_left marker → top_left paper corner
        - top_right marker → top_right paper corner
        - bottom_left marker → bottom_left paper corner
        - bottom_right marker → bottom_right paper corner
        
        Args:
            detected_markers: List of marker dicts with 'corner' field
            corners_data: List of corner coordinate arrays
            
        Returns:
            4x2 numpy array of paper corners [TL, TR, BL, BR] or None if insufficient markers
        """
        if len(detected_markers) < 2:
            return None
        
        corner_map = {}
        for i, marker in enumerate(detected_markers):
            if 'inferred' in marker:
                continue  # Only use real markers for region estimation
            corner_name = marker['corner']
            corner_coords = corners_data[i][0]
            corner_map[corner_name] = corner_coords
        
        corner_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        
        # Extract available corners
        paper_corners = []
        for corner_name in corner_names:
            if corner_name in corner_map:
                paper_corners.append(corner_map[corner_name])
            else:
                # If missing, try to infer from others
                available = {k: v for k, v in corner_map.items()}
                if len(available) == 3:
                    missing = [c for c in corner_names if c not in available][0]
                    if missing == 'bottom_right':
                        inferred = available.get('top_left') + available.get('bottom_left') - available.get('top_right')
                    elif missing == 'bottom_left':
                        inferred = available.get('top_left') + available.get('bottom_right') - available.get('top_right')
                    elif missing == 'top_right':
                        inferred = available.get('top_left') + available.get('bottom_right') - available.get('bottom_left')
                    elif missing == 'top_left':
                        inferred = available.get('top_right') + available.get('bottom_left') - available.get('bottom_right')
                    else:
                        return None
                    paper_corners.append(inferred)
                else:
                    # Not enough corners to estimate
                    return None
        
        return np.array(paper_corners, dtype=np.float32)

    @classmethod
    def _apply_perspective_transform(
        cls, 
        image: np.ndarray, 
        paper_corners: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Apply perspective transformation to straighten the page.
        
        Args:
            image: Input image
            paper_corners: 4x2 array of paper corners in order [TL, TR, BL, BR]
            
        Returns:
            Warped image or None on failure
        """
        if paper_corners.shape != (4, 2):
            cls.logger.warning(f"Invalid paper_corners shape: {paper_corners.shape}")
            return None
        
        try:
            # Define destination corners (preserving image dimensions)
            height, width = image.shape[:2]
            dest_corners = np.array([
                [0, 0],
                [width, 0],
                [0, height],
                [width, height]
            ], dtype=np.float32)
            
            # Get perspective transform matrix
            M = cv2.getPerspectiveTransform(paper_corners, dest_corners)
            
            # Apply transformation
            warped = cv2.warpPerspective(image, M, (width, height))
            cls.logger.debug("Perspective transform applied successfully")
            return warped
        except Exception as e:
            cls.logger.warning(f"Failed to apply perspective transform: {e}")
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