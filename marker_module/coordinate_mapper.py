import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .marker_config import MarkerConfig
from PIL import Image


class CoordinateMapper:
    """
    coordinate mapper that transforms coordinates from original document 
    to captured image using ArUco markers. All configuration is passed as parameters.
    """
    
    @staticmethod
    def calculate_original_marker_positions() -> Dict:
        """
        Calculate the center positions of the 4 markers in the original document.
        
        Args:
        
        Returns:
            Dictionary with marker positions for each corner
        """
        m = MarkerConfig.MARGIN
        s = MarkerConfig.MARKER_SIZE
        
        markers = {
            'top_left': {
                'center': (m + s/2, m + s/2),
                'corners': [
                    (m, m),
                    (m + s, m),
                    (m + s, m + s),
                    (m, m + s)
                ]
            },
            'top_right': {
                'center': (MarkerConfig.DOC_WIDTH - m - s/2, m + s/2),
                'corners': [
                    (MarkerConfig.DOC_WIDTH - m - s, m),
                    (MarkerConfig.DOC_WIDTH - m, m),
                    (MarkerConfig.DOC_WIDTH - m, m + s),
                    (MarkerConfig.DOC_WIDTH - m - s, m + s)
                ]
            },
            'bottom_left': {
                'center': (m + s/2, MarkerConfig.DOC_HEIGHT - m - s/2),
                'corners': [
                    (m, MarkerConfig.DOC_HEIGHT - m - s),
                    (m + s, MarkerConfig.DOC_HEIGHT - m - s),
                    (m + s, MarkerConfig.DOC_HEIGHT - m),
                    (m, MarkerConfig.DOC_HEIGHT - m)
                ]
            },
            'bottom_right': {
                'center': (MarkerConfig.DOC_WIDTH - m - s/2, MarkerConfig.DOC_HEIGHT - m - s/2),
                'corners': [
                    (MarkerConfig.DOC_WIDTH - m - s, MarkerConfig.DOC_HEIGHT - m - s),
                    (MarkerConfig.DOC_WIDTH - m, MarkerConfig.DOC_HEIGHT - m - s),
                    (MarkerConfig.DOC_WIDTH - m, MarkerConfig.DOC_HEIGHT - m),
                    (MarkerConfig.DOC_WIDTH - m - s, MarkerConfig.DOC_HEIGHT - m)
                ]
            }
        }
        
        return markers
    
    @staticmethod
    def calculate_marker_center(corners: np.ndarray) -> Tuple[float, float]:
        """
        Calculate center point from ArUco marker corners.
        
        Args:
            corners: ArUco corner array [4 points with (x, y) coordinates]
            
        Returns:
            Tuple (center_x, center_y)
        """
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        return (float(center_x), float(center_y))
    
    @staticmethod
    def compute_homography_from_scan(scan_result: Dict) -> Optional[np.ndarray]:
        """
        Compute homography matrix from ExamScanner scan_page() result.
        
        Args:
            scan_result: Result dictionary from ExamScanner.scan_page()

        
        Returns:
            Homography matrix or None if computation fails
        """
        if not scan_result.get('success', False):
            return None
        
        if scan_result['markers_found'] < 4:
            return None
        
        detected_markers = scan_result['detected_markers']
        corners_data = scan_result['corners']
        
        return CoordinateMapper.compute_homography(detected_markers, corners_data)
    
    @staticmethod
    def compute_homography(detected_markers: List[Dict], corners_data: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Compute homography matrix from detected markers.
        
        Args:
            detected_markers: List of marker info dicts with 'corner' field
            corners_data: List of corner arrays from ArUco detection

        
        Returns:
            Homography matrix or None if computation fails
        """
        if len(detected_markers) < 4:
            return None
        
        # Calculate original marker positions
        original_markers = CoordinateMapper.calculate_original_marker_positions()
        
        # Match detected markers to original positions
        src_points = []  # Points in original document
        dst_points = []  # Points in captured image
        
        for i, marker_info in enumerate(detected_markers):
            corner_name = marker_info['corner']
            
            if corner_name in original_markers:
                # Get center from original document
                orig_center = original_markers[corner_name]['center']
                src_points.append(orig_center)
                
                # Calculate center from detected corners
                marker_corners = corners_data[i][0]  # Shape: (4, 2)
                img_center = CoordinateMapper.calculate_marker_center(marker_corners)
                dst_points.append(img_center)
        
        if len(src_points) < 4:
            return None
        
        # Convert to numpy arrays
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Compute homography matrix
        homography_matrix, status = cv2.findHomography(src_points, dst_points)
        
        return homography_matrix
    
    @staticmethod
    def map_point_to_image(
        doc_x: float,
        doc_y: float,
        homography_matrix: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Map a point from original document coordinates to image coordinates.
        
        Args:
            doc_x: X coordinate in original document
            doc_y: Y coordinate in original document
            homography_matrix: Homography transformation matrix
            
        Returns:
            Tuple (img_x, img_y) or None if transformation fails

        """
        if homography_matrix is None:
            return None
        
        # Create point in homogeneous coordinates
        point = np.array([[[doc_x, doc_y]]], dtype=np.float32)
        
        # Transform point
        transformed = cv2.perspectiveTransform(point, homography_matrix)
        
        img_x = float(transformed[0][0][0])
        img_y = float(transformed[0][0][1])
        
        return (img_x, img_y)
    
    @staticmethod
    def map_points_to_image(
        doc_points: List[Tuple[float, float]],
        homography_matrix: np.ndarray
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Map multiple points from document to image coordinates.
        
        Args:
            doc_points: List of (x, y) tuples in document coordinates
            homography_matrix: Homography transformation matrix
            
        Returns:
            List of (x, y) tuples in image coordinates or None if transformation fails
        """
        if homography_matrix is None:
            return None
        
        # Convert to numpy array
        points = np.array([doc_points], dtype=np.float32)
        
        # Transform all points
        transformed = cv2.perspectiveTransform(points, homography_matrix)
        
        return [(float(x), float(y)) for x, y in transformed[0]]
    
    
    @staticmethod
    def extract_full_document(image: Image.Image, scan_result: Dict) -> Optional[Image.Image]:
        """
        Extract and dewarp the entire document from a captured photo.
        Takes the distorted photo and returns a straight, corrected document as PIL Image.
        """
        if not scan_result.get('success', False) or scan_result['markers_found'] < 4:
            return None
        
        # Compute homography: document coords → photo coords
        homography_matrix = CoordinateMapper.compute_homography_from_scan(scan_result)

        if homography_matrix is None:
            return None
        
        # Invert it to get: photo coords → document coords
        homography_inv = np.linalg.inv(homography_matrix)
        
        # Convert PIL Image to numpy array (RGB)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV processing
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Warp the photo to straighten it
        extracted_document = cv2.warpPerspective(
            image_array, 
            homography_inv, 
            (MarkerConfig.DOC_WIDTH, MarkerConfig.DOC_HEIGHT)
        )
        
        # Convert back from BGR to RGB for PIL
        extracted_document_rgb = cv2.cvtColor(extracted_document, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(extracted_document_rgb)
    
    @staticmethod
    def get_scale_factors(homography_matrix: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Calculate approximate scale factors between document and image.
        
        Args:
            homography_matrix: Homography transformation matrix
        
        Returns:
            Dictionary with scale_x, scale_y, and average_scale or None if calculation fails
        """
        if homography_matrix is None:
            return None
        
        # Map two points to estimate scale
        p1_doc = (0, 0)
        p2_doc = (MarkerConfig.DOC_WIDTH, MarkerConfig.DOC_HEIGHT)
        
        p1_img = CoordinateMapper.map_point_to_image(*p1_doc, homography_matrix)
        p2_img = CoordinateMapper.map_point_to_image(*p2_doc, homography_matrix)
        
        if p1_img is None or p2_img is None:
            return None
        
        scale_x = abs(p2_img[0] - p1_img[0]) / MarkerConfig.DOC_WIDTH
        scale_y = abs(p2_img[1] - p1_img[1]) / MarkerConfig.DOC_HEIGHT
        
        return {
            'scale_x': scale_x,
            'scale_y': scale_y,
            'average_scale': (scale_x + scale_y) / 2
        }

