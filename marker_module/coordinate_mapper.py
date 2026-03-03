"""
coordinate_mapper.py
--------------------
Transforms coordinates between the original document space and a captured
image using ArUco markers.

Homography robustness tiers
    4 markers → standard cv2.findHomography + RANSAC
    3 markers → estimate missing corner via parallelogram rule
    2 markers → reconstruct all corners via similarity transform
    0-1 marker → fall back to blue-boundary extraction from visualisation

Blue-boundary fallback (integrated from blue_boundary_extractor)
    When ArUco detection is insufficient, the module can recover the document
    quadrilateral by colour-filtering the blue polyline drawn by the
    visualisation step and re-computing a homography from those corners.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .marker_config import MarkerConfig
from PIL import Image


# ---------------------------------------------------------------------------
# Corner ordering — TL → TR → BR → BL throughout this module
# ---------------------------------------------------------------------------
CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]


# ===========================================================================
# Blue-boundary detection helpers (formerly blue_boundary_extractor.py)
# ===========================================================================

# HSV hue band for the blue polyline drawn by cv2.polylines(..., (255,0,0))
# OpenCV hue is [0, 180]; pure blue ≈ H 110-120, widened to [95, 135] for
# anti-aliasing tolerance.
_BLUE_H_LOW  = 95
_BLUE_H_HIGH = 135
_BLUE_S_LOW  = 80   # reject near-grey pixels
_BLUE_V_LOW  = 80   # reject dark regions (shadows, ArUco markers)


def _bb_threshold_blue(image_bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask isolating the blue boundary pixels via HSV
    thresholding.  HSV separates hue from brightness, making the result
    robust to lighting variation.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([_BLUE_H_LOW,  _BLUE_S_LOW, _BLUE_V_LOW], dtype=np.uint8)
    upper = np.array([_BLUE_H_HIGH, 255,          255         ], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _bb_close_gaps(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Morphological closing (dilate then erode) to bridge small anti-aliasing
    gaps in the drawn polyline without displacing the boundary position.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _bb_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the largest external contour in the mask.  The document boundary
    is always the dominant blue object, so max-area selection is robust.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _bb_approximate_polygon(
    contour: np.ndarray, epsilon_factor: float = 0.02
) -> np.ndarray:
    """
    Simplify a contour to a polygon using Ramer-Douglas-Peucker.
    epsilon = epsilon_factor x arc_length; 0.02 works well for document quads.
    """
    peri = cv2.arcLength(contour, closed=True)
    return cv2.approxPolyDP(contour, epsilon_factor * peri, closed=True)


def _bb_reconstruct_missing_corner(
    pts: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Given exactly 3 corners of a quadrilateral, estimate the 4th using the
    parallelogram diagonal rule:

        TL + BR = TR + BL   ->   missing = A + C - B

    Points are assigned to quadrants via the centroid before the formula
    is applied so the correct role is always used.
    """
    arr = np.array(pts, dtype=np.float64)
    cx, cy = arr.mean(axis=0)

    quadrants: Dict[str, Tuple[int, int]] = {}
    for p in pts:
        qx = "right" if p[0] >= cx else "left"
        qy = "bottom" if p[1] >= cy else "top"
        quadrants[f"{qy}_{qx}"] = p

    all_names = {"top_left", "top_right", "bottom_right", "bottom_left"}
    missing_name = (all_names - set(quadrants.keys())).pop()

    tl = quadrants.get("top_left",     (0, 0))
    tr = quadrants.get("top_right",    (0, 0))
    br = quadrants.get("bottom_right", (0, 0))
    bl = quadrants.get("bottom_left",  (0, 0))

    def add(a, b): return (a[0] + b[0], a[1] + b[1])
    def sub(a, b): return (a[0] - b[0], a[1] - b[1])

    if missing_name == "top_left":
        estimated = add(sub(tr, br), bl)
    elif missing_name == "top_right":
        estimated = add(sub(tl, bl), br)
    elif missing_name == "bottom_right":
        estimated = add(sub(tr, tl), bl)
    else:  # bottom_left
        estimated = add(sub(tl, tr), br)

    quadrants[missing_name] = (int(estimated[0]), int(estimated[1]))
    return [
        quadrants["top_left"],
        quadrants["top_right"],
        quadrants["bottom_right"],
        quadrants["bottom_left"],
    ]


def _bb_select_four_extreme_points(
    points: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    From a convex polygon with > 4 vertices, select the 4 points closest to
    the 4 bounding-box corners, returning them in TL, TR, BR, BL order.
    """
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    bbox_corners = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ], dtype=np.float64)

    selected = []
    for bbox_pt in bbox_corners:
        dists = np.linalg.norm(points.astype(np.float64) - bbox_pt, axis=1)
        selected.append(tuple(points[np.argmin(dists)].tolist()))
    return selected


def _bb_order_corners(
    pts: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Order 4 corner points as [TL, TR, BR, BL] using the sum/difference trick:
        TL -> min(x + y),  BR -> max(x + y)
        TR -> min(x - y),  BL -> max(x - y)
    Correct under perspective distortion for quads with less than ~45 degree keystone.
    """
    arr = np.array(pts, dtype=np.int32)
    s = arr.sum(axis=1)
    d = np.diff(arr, axis=1).ravel()
    return [
        tuple(arr[np.argmin(s)].tolist()),   # TL
        tuple(arr[np.argmin(d)].tolist()),   # TR
        tuple(arr[np.argmax(s)].tolist()),   # BR
        tuple(arr[np.argmax(d)].tolist()),   # BL
    ]


def extract_blue_boundary(
    image: np.ndarray,
) -> Optional[List[Tuple[int, int]]]:
    """
    Detect the blue document-boundary quadrilateral drawn by cv2.polylines
    and return its 4 ordered corner points.

    Pipeline
    --------
    1. HSV colour threshold   -> isolates blue pixels
    2. Morphological closing  -> heals anti-aliasing / drawing gaps
    3. Largest contour        -> selects the document boundary blob
    4. Polygon approximation  -> collapses edge pixels to corner points
    5. Hull fallback          -> handles irregular approximation results
    6. 3-point reconstruction -> recovers a missing corner if needed
    7. 4-extreme selection    -> reduces hull with > 4 points
    8. Corner ordering        -> returns [TL, TR, BR, BL]

    Args:
        image: BGR image array (as returned by cv2.imread).

    Returns:
        List of 4 (x, y) integer tuples ordered [TL, TR, BR, BL],
        or None if the blue boundary cannot be reliably detected.
    """
    mask = _bb_threshold_blue(image)
    mask = _bb_close_gaps(mask, kernel_size=15)

    contour = _bb_largest_contour(mask)
    if contour is None or cv2.contourArea(contour) < 1000:
        return None

    approx = _bb_approximate_polygon(contour, epsilon_factor=0.02)
    pts_raw = approx.reshape(-1, 2)

    # Fall back to convex hull if approximation collapses to fewer than 3 points
    if len(pts_raw) < 3:
        hull = cv2.convexHull(contour)
        pts_raw = hull.reshape(-1, 2)

    if len(pts_raw) < 3:
        return None

    pts_list: List[Tuple[int, int]] = [tuple(p.tolist()) for p in pts_raw]

    # Reconstruct missing corner from 3 known points
    if len(pts_list) == 3:
        pts_list = _bb_reconstruct_missing_corner(pts_list)

    # Select the 4 most extreme hull points when more than 4 remain
    if len(pts_list) > 4:
        pts_arr = np.array(pts_list, dtype=np.int32)
        pts_list = _bb_select_four_extreme_points(pts_arr)

    if len(pts_list) != 4:
        return None

    return _bb_order_corners(pts_list)


# ===========================================================================
# CoordinateMapper
# ===========================================================================

class CoordinateMapper:
    """
    Transforms coordinates between the original document space and a captured
    image using ArUco markers.

    Homography robustness tiers:
        4 markers -> standard homography + RANSAC
        3 markers -> estimate missing corner via parallelogram rule
        2 markers -> reconstruct corners via similarity transform
        0-1 marker, visualisation image available
                  -> extract blue boundary and recompute homography

    The blue-boundary fallback is available via:
        CoordinateMapper.compute_homography_from_blue_boundary(vis_image)
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
    # Missing-corner estimation (ArUco path)
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
        the remaining detected corner. More explicitly:

            missing TL  ->  TL = TR + BL - BR
            missing TR  ->  TR = TL + BR - BL
            missing BR  ->  BR = TR + BL - TL
            missing BL  ->  BL = TL + BR - TR

        These formulae are exact for rectangles and parallelograms and are
        the best linear approximation under mild perspective distortion.

        Args:
            detected: dict mapping corner name -> (img_x, img_y) for the 3
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

        def add(a, b): return (a[0] + b[0], a[1] + b[1])
        def sub(a, b): return (a[0] - b[0], a[1] - b[1])

        if missing_name == "top_left":
            estimated = add(sub(tr, br), bl)
        elif missing_name == "top_right":
            estimated = add(sub(tl, bl), br)
        elif missing_name == "bottom_right":
            estimated = add(sub(tr, tl), bl)
        else:  # bottom_left
            estimated = add(sub(tl, tr), br)

        return missing_name, estimated

    @staticmethod
    def reconstruct_from_two_markers(
        detected: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Reconstruct all 4 corner positions in IMAGE space from just 2 detected
        markers, using the known document aspect ratio and a similarity
        (rotation + uniform scale) transform.

        With only 2 correspondences there is not enough information to recover
        the full 8-DOF homography, so this returns the best affine estimate
        and accepts residual perspective error.

        Args:
            detected: dict mapping corner name -> (img_x, img_y) for the 2
                      known corners.

        Returns:
            Dict mapping all 4 corner names -> estimated (img_x, img_y).

        Raises:
            ValueError if fewer than 2 corners are provided or markers
            are coincident in document space.
        """
        if len(detected) < 2:
            raise ValueError("Need at least 2 detected corners.")

        original_positions = CoordinateMapper.calculate_original_marker_positions()

        names = list(detected.keys())
        p_name, q_name = names[0], names[1]

        P_img = np.array(detected[p_name], dtype=np.float64)
        Q_img = np.array(detected[q_name], dtype=np.float64)
        P_doc = np.array(original_positions[p_name]["center"], dtype=np.float64)
        Q_doc = np.array(original_positions[q_name]["center"], dtype=np.float64)

        u = Q_doc - P_doc   # document-space direction vector
        v = Q_img - P_img   # corresponding image-space direction vector

        u_norm_sq = float(np.dot(u, u))
        if u_norm_sq < 1e-6:
            raise ValueError(
                "The two detected markers are too close in document space."
            )

        u_perp = np.array([-u[1], u[0]], dtype=np.float64)
        dot_uv      = float(np.dot(u, v))
        dot_uperp_v = float(np.dot(u_perp, v))

        # 2x2 similarity matrix: doc_offset -> img_offset
        M = np.array([
            [dot_uv,       -dot_uperp_v],
            [dot_uperp_v,   dot_uv     ],
        ], dtype=np.float64) / u_norm_sq

        all_corners: Dict[str, Tuple[float, float]] = {}
        for corner_name in CORNER_ORDER:
            if corner_name in detected:
                all_corners[corner_name] = detected[corner_name]
            else:
                doc_center = np.array(
                    original_positions[corner_name]["center"], dtype=np.float64
                )
                img_offset = M @ (doc_center - P_doc)
                estimated  = P_img + img_offset
                all_corners[corner_name] = (float(estimated[0]), float(estimated[1]))

        return all_corners

    # ------------------------------------------------------------------
    # Blue-boundary homography fallback
    # ------------------------------------------------------------------

    @staticmethod
    def compute_homography_from_blue_boundary(
        vis_image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute a homography by detecting the blue document-boundary
        quadrilateral drawn in a visualisation image.

        This is the last-resort fallback when fewer than 2 ArUco markers
        were detected. It works by:

        1. Running extract_blue_boundary() to find the 4 image-space corners
           of the blue polyline drawn by test_mapper.py.
        2. Mapping those corners to the known document-space corner positions
           (0,0), (W,0), (W,H), (0,H).
        3. Calling cv2.findHomography on those 4 correspondences.

        Args:
            vis_image: BGR image array of the mapper visualisation output.

        Returns:
            3x3 homography matrix or None if the boundary cannot be found.
        """
        corners_img = extract_blue_boundary(vis_image)
        if corners_img is None:
            return None

        # extract_blue_boundary returns [TL, TR, BR, BL] as integer tuples.
        W = float(MarkerConfig.DOC_WIDTH)
        H = float(MarkerConfig.DOC_HEIGHT)

        src_points = np.array(
            [(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32
        )  # TL, TR, BR, BL in document space

        dst_points = np.array(corners_img, dtype=np.float32)  # image space

        homography_matrix, _ = cv2.findHomography(src_points, dst_points, 0)
        return homography_matrix

    # ------------------------------------------------------------------
    # Core homography computation  (ArUco path with blue-boundary fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_homography(
        detected_markers: List[Dict],
        corners_data: List[np.ndarray],
        vis_image: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute a homography matrix from detected ArUco markers, with an
        automatic fallback to blue-boundary detection when markers are scarce.

        Dispatch order:
            n >= 4  -> standard cv2.findHomography + RANSAC
            n == 3  -> parallelogram rule fills missing corner
            n == 2  -> similarity transform reconstructs all 4 corners
            n <= 1  -> blue-boundary fallback (requires vis_image)

        Args:
            detected_markers: List of dicts with at least a 'corner' key.
            corners_data:     Parallel list of ArUco corner arrays (N x 1 x 4 x 2).
            vis_image:        Optional BGR visualisation image used as a
                              fallback when fewer than 2 markers are detected.

        Returns:
            3x3 homography matrix (np.float64) or None on failure.
        """
        original_positions = CoordinateMapper.calculate_original_marker_positions()

        # ---- Collect valid detected image-space centers ------------------
        detected_img: Dict[str, Tuple[float, float]] = {}
        for i, marker_info in enumerate(detected_markers):
            corner_name = marker_info.get("corner")
            if corner_name not in original_positions:
                continue
            marker_corners = corners_data[i][0]  # shape (4, 2)
            center = CoordinateMapper.calculate_marker_center(marker_corners)
            detected_img[corner_name] = center

        n_valid = len(detected_img)

        # ---- Blue-boundary fallback when ArUco gives fewer than 2 markers
        if n_valid < 2:
            if vis_image is not None:
                return CoordinateMapper.compute_homography_from_blue_boundary(
                    vis_image
                )
            return None

        # ---- Promote to 4 corners via geometry ---------------------------
        if n_valid >= 4:
            all_img = detected_img

        elif n_valid == 3:
            missing_name, estimated = (
                CoordinateMapper.estimate_missing_corner_3_markers(detected_img)
            )
            all_img = dict(detected_img)
            all_img[missing_name] = estimated

        else:  # n_valid == 2
            all_img = CoordinateMapper.reconstruct_from_two_markers(detected_img)

        # ---- Build src / dst point arrays --------------------------------
        src_points, dst_points = [], []
        for corner_name in CORNER_ORDER:
            if corner_name not in all_img:
                continue
            src_points.append(original_positions[corner_name]["center"])
            dst_points.append(all_img[corner_name])

        if len(src_points) < 4:
            return None

        src_arr = np.array(src_points, dtype=np.float32)
        dst_arr = np.array(dst_points, dtype=np.float32)

        # RANSAC only when all 4 points are real measurements
        method = cv2.RANSAC if n_valid >= 4 else 0
        homography_matrix, _ = cv2.findHomography(
            src_arr, dst_arr, method, ransacReprojThreshold=5.0
        )
        return homography_matrix

    @staticmethod
    def compute_homography_from_scan(
        scan_result: Dict,
        vis_image: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute homography matrix from ExamScanner.scan_page() result dict.

        Args:
            scan_result: Result dictionary from ExamScanner.scan_page().
            vis_image:   Optional BGR visualisation image for blue-boundary
                         fallback when marker count is less than 2.

        Returns:
            Homography matrix or None if computation fails.
        """
        if not scan_result.get("success", False):
            return None
        if scan_result["markers_found"] < 2 and vis_image is None:
            return None

        return CoordinateMapper.compute_homography(
            scan_result["detected_markers"],
            scan_result["corners"],
            vis_image=vis_image,
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
            homography_matrix: 3x3 homography matrix.

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
            homography_matrix: 3x3 homography matrix.

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
        image: Image.Image,
        scan_result: Dict,
        vis_image: Optional[np.ndarray] = None,
    ) -> Optional[Image.Image]:
        """
        Extract and dewarp the entire document from a captured photo.

        Returns a corrected PIL Image with dimensions DOC_WIDTH x DOC_HEIGHT,
        or None if dewarping is not possible.

        Args:
            image:       PIL RGB image of the raw captured photo.
            scan_result: Result dict from ExamScanner.scan_page().
            vis_image:   Optional BGR visualisation image; enables the
                         blue-boundary fallback when ArUco markers are scarce.
        """
        markers_found = scan_result.get("markers_found", 0)
        if not scan_result.get("success", False):
            return None
        if markers_found < 2 and vis_image is None:
            return None

        homography_matrix = CoordinateMapper.compute_homography_from_scan(
            scan_result, vis_image=vis_image
        )
        if homography_matrix is None:
            return None

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
            homography_matrix: 3x3 homography matrix.

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