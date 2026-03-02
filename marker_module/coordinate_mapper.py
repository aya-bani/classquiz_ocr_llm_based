import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .marker_config import MarkerConfig
from PIL import Image


# Corner ordering used throughout this module.
# We always treat the quadrilateral as:
#   TL --- TR
#   |       |
#   BL --- BR
CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]


class CoordinateMapper:
    """
    Coordinate mapper that transforms coordinates from the original document
    to a captured image using ArUco markers.

    Supports robust homography estimation when only 2 or 3 of the 4 expected
    markers are detected, using projective / affine geometry fallbacks.
    """

    # ------------------------------------------------------------------
    # Original document geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_original_marker_positions() -> Dict:
        """
        Calculate the center positions of the 4 markers in the original
        document coordinate system.

        Returns:
            Dictionary keyed by corner name, each with 'center' and 'corners'.
        """
        m = MarkerConfig.MARGIN
        s = MarkerConfig.MARKER_SIZE
        W = MarkerConfig.DOC_WIDTH
        H = MarkerConfig.DOC_HEIGHT

        return {
            "top_left": {
                "center": (m + s / 2, m + s / 2),
                "corners": [(m, m), (m + s, m), (m + s, m + s), (m, m + s)],
            },
            "top_right": {
                "center": (W - m - s / 2, m + s / 2),
                "corners": [
                    (W - m - s, m), (W - m, m),
                    (W - m, m + s), (W - m - s, m + s),
                ],
            },
            "bottom_left": {
                "center": (m + s / 2, H - m - s / 2),
                "corners": [
                    (m, H - m - s), (m + s, H - m - s),
                    (m + s, H - m), (m, H - m),
                ],
            },
            "bottom_right": {
                "center": (W - m - s / 2, H - m - s / 2),
                "corners": [
                    (W - m - s, H - m - s), (W - m, H - m - s),
                    (W - m, H - m), (W - m - s, H - m),
                ],
            },
        }

    @staticmethod
    def calculate_marker_center(corners: np.ndarray) -> Tuple[float, float]:
        """
        Calculate center point from ArUco marker corners.

        Args:
            corners: Array of shape (4, 2) with (x, y) coordinates.

        Returns:
            Tuple (center_x, center_y).
        """
        return (float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1])))

    # ------------------------------------------------------------------
    # Missing-corner estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_missing_corner_3_markers(
        detected: Dict[str, Tuple[float, float]]
    ) -> Tuple[str, Tuple[float, float]]:
        """
        Estimate the position of the missing 4th corner in IMAGE space when
        exactly 3 markers are detected.

        Geometric principle (parallelogram / projective rule):
        -------------------------------------------------------
        A planar quadrilateral with vertices TL, TR, BR, BL satisfies the
        relationship:

            TL + BR = TR + BL   (midpoints of diagonals coincide)

        so the missing corner D can always be recovered as:

            D = A + C - B

        where A and C are the two corners *diagonal* to each other and B is
        the remaining detected corner.  More explicitly:

            missing TL  →  TL  = TR + BL - BR
            missing TR  →  TR  = TL + BR - BL
            missing BR  →  BR  = TR + BL - TL
            missing BL  →  BL  = TL + BR - TR

        These formulae are exact for rectangles and parallelograms and are
        the best linear approximation under mild perspective distortion.

        Args:
            detected: dict mapping corner name → (img_x, img_y) for the 3
                      known corners.

        Returns:
            Tuple (missing_corner_name, (estimated_x, estimated_y)).

        Raises:
            ValueError if fewer than 3 corners are provided.
        """
        if len(detected) < 3:
            raise ValueError("Need at least 3 detected corners.")

        all_corners = {"top_left", "top_right", "bottom_right", "bottom_left"}
        missing_name = (all_corners - set(detected.keys())).pop()

        tl = detected.get("top_left")
        tr = detected.get("top_right")
        br = detected.get("bottom_right")
        bl = detected.get("bottom_left")

        def add(a, b):
            return (a[0] + b[0], a[1] + b[1])

        def sub(a, b):
            return (a[0] - b[0], a[1] - b[1])

        # Apply: D = A + C - B  (diagonal sum rule)
        if missing_name == "top_left":
            # TL is diagonal to BR; TR and BL are the other two
            estimated = add(sub(tr, br), bl)  # = TR + BL - BR
        elif missing_name == "top_right":
            estimated = add(sub(tl, bl), br)  # = TL + BR - BL
        elif missing_name == "bottom_right":
            estimated = add(sub(tr, tl), bl)  # = TR + BL - TL
        else:  # bottom_left
            estimated = add(sub(tl, tr), br)  # = TL + BR - TR

        return missing_name, estimated

    @staticmethod
    def reconstruct_from_two_markers(
        detected: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Reconstruct all 4 corner positions in IMAGE space from just 2 detected
        markers, using the known document aspect ratio and projective geometry.

        Strategy overview:
        ------------------
        Given two detected corners P and Q (in image pixels) and their known
        positions in document space (p_doc, q_doc), we:

        1. Compute the vector v_PQ = Q - P in image space.
        2. Find the document-space vector between the same two points.
        3. Express the remaining two document corners relative to P in doc space.
        4. Use the *same linear mapping* (rotation + scale implied by v_PQ vs
           its document counterpart) to convert those relative doc vectors into
           image-space offsets, then add them to P.

        This is an affine approximation: it captures translation, rotation, and
        uniform scale (the two scale factors may differ along x/y if the detected
        pair spans the document diagonally or along height), but it does NOT
        model perspective (keystone) distortion.  With only 2 points there is
        simply not enough information to recover the full 8-DOF homography, so
        we return the best affine estimate and accept residual perspective error.

        The resulting 4 points are then used with cv2.findHomography (which will
        fall back internally to a least-squares fit).

        Args:
            detected: dict mapping corner name → (img_x, img_y) for the 2
                      known corners.

        Returns:
            Dict mapping all 4 corner names → estimated (img_x, img_y).

        Raises:
            ValueError if fewer than 2 corners are provided.
        """
        if len(detected) < 2:
            raise ValueError("Need at least 2 detected corners.")

        original_positions = CoordinateMapper.calculate_original_marker_positions()

        # Pull the two detected corners and their document-space positions.
        names = list(detected.keys())
        p_name, q_name = names[0], names[1]

        P_img = np.array(detected[p_name], dtype=np.float64)
        Q_img = np.array(detected[q_name], dtype=np.float64)

        P_doc = np.array(original_positions[p_name]["center"], dtype=np.float64)
        Q_doc = np.array(original_positions[q_name]["center"], dtype=np.float64)

        # ---- Build a 2-D linear map: doc_offset → img_offset ------------
        #
        # We need a 2×2 matrix M such that M @ (Q_doc - P_doc) = Q_img - P_img.
        # With a single vector pair we cannot determine M uniquely; we use the
        # simplest solution: a similarity (rotation + uniform scale).
        #
        # Let u = Q_doc - P_doc,  v = Q_img - P_img.
        # We want M u = v.  We choose M as the unique rotation+scale that maps
        # u onto v, which gives:
        #
        #   M = (1/|u|²) * [[u·v,  -u⊥·v],
        #                   [u⊥·v,  u·v  ]]
        #
        # where u⊥ = (-u_y, u_x) (90° rotation of u).
        # This is the standard solution for the 2-D similarity from one vector.

        u = Q_doc - P_doc          # direction vector in document space
        v = Q_img - P_img          # corresponding direction in image space

        u_norm_sq = float(np.dot(u, u))
        if u_norm_sq < 1e-6:
            raise ValueError("The two detected markers are too close in document space.")

        u_perp = np.array([-u[1], u[0]], dtype=np.float64)

        # Build the 2×2 similarity matrix
        dot_uv = float(np.dot(u, v))
        dot_uperp_v = float(np.dot(u_perp, v))

        M = np.array([
            [dot_uv,       -dot_uperp_v],
            [dot_uperp_v,   dot_uv     ],
        ], dtype=np.float64) / u_norm_sq
        # M maps doc_offset → img_offset via a rotation + uniform scale

        # ---- Reconstruct all 4 corners -----------------------------------
        all_corners: Dict[str, Tuple[float, float]] = {}

        for corner_name in ["top_left", "top_right", "bottom_right", "bottom_left"]:
            if corner_name in detected:
                all_corners[corner_name] = detected[corner_name]
            else:
                doc_center = np.array(
                    original_positions[corner_name]["center"], dtype=np.float64
                )
                doc_offset = doc_center - P_doc          # offset in doc space
                img_offset = M @ doc_offset              # map to image space
                estimated = P_img + img_offset
                all_corners[corner_name] = (float(estimated[0]), float(estimated[1]))

        return all_corners

    # ------------------------------------------------------------------
    # Core homography computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_homography(
        detected_markers: List[Dict], corners_data: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute a homography matrix from 2, 3, or 4 detected ArUco markers.

        Dispatch logic:
            4 markers → standard cv2.findHomography (exact + RANSAC)
            3 markers → estimate missing corner via parallelogram rule, then
                        call cv2.findHomography with all 4 points
            2 markers → reconstruct all 4 corners via similarity transform
                        from known aspect ratio, then call cv2.findHomography

        Args:
            detected_markers: List of dicts, each containing at least a
                              'corner' key ('top_left', 'top_right', etc.).
            corners_data:     Parallel list of ArUco corner arrays (shape N×1×4×2
                              or N×4×2 — whichever ArUco returns).

        Returns:
            3×3 homography matrix (np.float64) or None on failure.
        """
        n = len(detected_markers)
        if n < 2:
            return None

        original_positions = CoordinateMapper.calculate_original_marker_positions()

        # ---- Collect detected image-space centers -------------------------
        detected_img: Dict[str, Tuple[float, float]] = {}
        for i, marker_info in enumerate(detected_markers):
            corner_name = marker_info.get("corner")
            if corner_name not in original_positions:
                continue
            marker_corners = corners_data[i][0]   # shape (4, 2)
            center = CoordinateMapper.calculate_marker_center(marker_corners)
            detected_img[corner_name] = center

        n_valid = len(detected_img)
        if n_valid < 2:
            return None

        # ---- Obtain all 4 image-space corners ----------------------------
        if n_valid >= 4:
            # Happy path: use detected centers directly.
            all_img = detected_img

        elif n_valid == 3:
            # Estimate the missing corner with the parallelogram / diagonal rule.
            missing_name, estimated = CoordinateMapper.estimate_missing_corner_3_markers(
                detected_img
            )
            all_img = dict(detected_img)
            all_img[missing_name] = estimated

        else:  # n_valid == 2
            # Reconstruct all 4 corners using the document aspect ratio.
            all_img = CoordinateMapper.reconstruct_from_two_markers(detected_img)

        # ---- Build src / dst point arrays --------------------------------
        # src = original document coordinates
        # dst = image pixel coordinates
        src_points = []
        dst_points = []

        for corner_name in CORNER_ORDER:
            if corner_name not in all_img:
                continue  # Should not happen after reconstruction above
            src_points.append(original_positions[corner_name]["center"])
            dst_points.append(all_img[corner_name])

        if len(src_points) < 4:
            return None

        src_arr = np.array(src_points, dtype=np.float32)
        dst_arr = np.array(dst_points, dtype=np.float32)

        # Use RANSAC only when we have enough real measurements (4+ detected).
        # For reconstructed points we use the algebraic (DLS) method directly.
        method = cv2.RANSAC if n_valid >= 4 else 0
        ransac_thresh = 5.0  # pixels

        homography_matrix, _ = cv2.findHomography(
            src_arr, dst_arr, method, ransacReprojThreshold=ransac_thresh
        )
        return homography_matrix

    @staticmethod
    def compute_homography_from_scan(scan_result: Dict) -> Optional[np.ndarray]:
        """
        Compute homography matrix from ExamScanner.scan_page() result dict.

        Args:
            scan_result: Result dictionary from ExamScanner.scan_page().

        Returns:
            Homography matrix or None if computation fails.
        """
        if not scan_result.get("success", False):
            return None
        if scan_result["markers_found"] < 2:
            return None

        return CoordinateMapper.compute_homography(
            scan_result["detected_markers"], scan_result["corners"]
        )

    # ------------------------------------------------------------------
    # Coordinate mapping utilities
    # ------------------------------------------------------------------

    @staticmethod
    def map_point_to_image(
        doc_x: float, doc_y: float, homography_matrix: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Map a single point from document coordinates to image coordinates.

        Args:
            doc_x: X coordinate in the original document.
            doc_y: Y coordinate in the original document.
            homography_matrix: 3×3 homography matrix.

        Returns:
            (img_x, img_y) or None on failure.
        """
        if homography_matrix is None:
            return None
        point = np.array([[[doc_x, doc_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, homography_matrix)
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))

    @staticmethod
    def map_points_to_image(
        doc_points: List[Tuple[float, float]], homography_matrix: np.ndarray
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Map multiple points from document coordinates to image coordinates.

        Args:
            doc_points: List of (x, y) tuples in document space.
            homography_matrix: 3×3 homography matrix.

        Returns:
            List of (img_x, img_y) tuples or None on failure.
        """
        if homography_matrix is None:
            return None
        points = np.array([doc_points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points, homography_matrix)
        return [(float(x), float(y)) for x, y in transformed[0]]

    # ------------------------------------------------------------------
    # Document extraction (dewarping)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_full_document(
        image: Image.Image, scan_result: Dict
    ) -> Optional[Image.Image]:
        """
        Extract and dewarp the entire document from a captured photo.

        Returns a corrected PIL Image with dimensions DOC_WIDTH × DOC_HEIGHT,
        or None if dewarping is not possible.
        """
        markers_found = scan_result.get("markers_found", 0)
        if not scan_result.get("success", False) or markers_found < 2:
            return None

        homography_matrix = CoordinateMapper.compute_homography_from_scan(scan_result)
        if homography_matrix is None:
            return None

        # Invert homography: photo coords → document coords
        homography_inv = np.linalg.inv(homography_matrix)

        image_array = np.array(image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        extracted = cv2.warpPerspective(
            image_array,
            homography_inv,
            (MarkerConfig.DOC_WIDTH, MarkerConfig.DOC_HEIGHT),
        )

        extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(extracted_rgb)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def get_scale_factors(
        homography_matrix: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        """
        Estimate approximate scale factors between document and image space.

        Args:
            homography_matrix: 3×3 homography matrix.

        Returns:
            Dict with 'scale_x', 'scale_y', 'average_scale', or None.
        """
        if homography_matrix is None:
            return None

        p1_img = CoordinateMapper.map_point_to_image(0, 0, homography_matrix)
        p2_img = CoordinateMapper.map_point_to_image(
            MarkerConfig.DOC_WIDTH, MarkerConfig.DOC_HEIGHT, homography_matrix
        )

        if p1_img is None or p2_img is None:
            return None

        scale_x = abs(p2_img[0] - p1_img[0]) / MarkerConfig.DOC_WIDTH
        scale_y = abs(p2_img[1] - p1_img[1]) / MarkerConfig.DOC_HEIGHT

        return {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "average_scale": (scale_x + scale_y) / 2,
        }