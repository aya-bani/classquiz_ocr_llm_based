"""
coordinate_mapper.py
--------------------
Transforms coordinates between the original document space and a captured
image using ArUco markers.

Homography robustness tiers
    4 markers -> standard cv2.findHomography + RANSAC
    3 markers -> estimate missing corner via parallelogram rule
    2 markers -> reconstruct all corners via similarity transform
    0-1 marker -> fall back to blue-boundary extraction from visualisation

Document boundary computation
    The visible document boundary is NOT computed by mapping abstract (0,0)
    corners through the homography. Instead it is derived directly from the
    marker centers in image space using the known physical offset:

        offset = MARGIN + MARKER_SIZE / 2   (pixels in document space)

    This offset is converted to image-space vectors by reading the local axis
    directions from the marker layout, so the quad remains geometrically
    consistent even when corners are reconstructed from 2 or 3 markers.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .marker_config import MarkerConfig
from PIL import Image


# Corner ordering: TL -> TR -> BR -> BL throughout this module
CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]


# ===========================================================================
# Blue-boundary detection helpers
# ===========================================================================

_BLUE_H_LOW  = 95    # HSV hue lower bound for the drawn blue polyline
_BLUE_H_HIGH = 135   # HSV hue upper bound
_BLUE_S_LOW  = 80    # reject near-grey pixels
_BLUE_V_LOW  = 80    # reject dark regions (shadows, ArUco markers)


def _bb_threshold_blue(image_bgr: np.ndarray) -> np.ndarray:
    """Binary mask of blue pixels via HSV thresholding (lighting-robust)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([_BLUE_H_LOW,  _BLUE_S_LOW, _BLUE_V_LOW], dtype=np.uint8)
    upper = np.array([_BLUE_H_HIGH, 255,          255         ], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _bb_close_gaps(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Morphological closing: bridges anti-aliasing gaps without moving the boundary."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _bb_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest external contour (the document boundary is always dominant)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _bb_approximate_polygon(contour: np.ndarray, epsilon_factor: float = 0.02) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification; epsilon = 2% of perimeter."""
    peri = cv2.arcLength(contour, closed=True)
    return cv2.approxPolyDP(contour, epsilon_factor * peri, closed=True)


def _bb_reconstruct_missing_corner(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Given 3 quad corners, estimate the 4th via the parallelogram diagonal rule:
        missing = A + C - B  (from TL + BR = TR + BL)
    Points are assigned to quadrants via the centroid before applying the formula.
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
    if missing_name == "top_left":       est = add(sub(tr, br), bl)
    elif missing_name == "top_right":    est = add(sub(tl, bl), br)
    elif missing_name == "bottom_right": est = add(sub(tr, tl), bl)
    else:                                est = add(sub(tl, tr), br)
    quadrants[missing_name] = (int(est[0]), int(est[1]))
    return [quadrants["top_left"], quadrants["top_right"],
            quadrants["bottom_right"], quadrants["bottom_left"]]


def _bb_select_four_extreme_points(points: np.ndarray) -> List[Tuple[int, int]]:
    """From >4 hull points, pick the 4 closest to the bounding-box corners."""
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    bbox = np.array([[x_min, y_min], [x_max, y_min],
                     [x_max, y_max], [x_min, y_max]], dtype=np.float64)
    selected = []
    for bp in bbox:
        dists = np.linalg.norm(points.astype(np.float64) - bp, axis=1)
        selected.append(tuple(points[np.argmin(dists)].tolist()))
    return selected


def _bb_order_corners(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Order 4 points as [TL, TR, BR, BL] via min/max of sum and difference."""
    arr = np.array(pts, dtype=np.int32)
    s = arr.sum(axis=1)
    d = np.diff(arr, axis=1).ravel()
    return [tuple(arr[np.argmin(s)].tolist()),   # TL: min(x+y)
            tuple(arr[np.argmin(d)].tolist()),   # TR: min(x-y)
            tuple(arr[np.argmax(s)].tolist()),   # BR: max(x+y)
            tuple(arr[np.argmax(d)].tolist())]   # BL: max(x-y)


def extract_blue_boundary(image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """
    Detect the blue document-boundary quadrilateral drawn by cv2.polylines
    and return its 4 ordered corner points [TL, TR, BR, BL], or None.

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
    """
    mask = _bb_threshold_blue(image)
    mask = _bb_close_gaps(mask, kernel_size=15)
    contour = _bb_largest_contour(mask)
    if contour is None or cv2.contourArea(contour) < 1000:
        return None
    approx = _bb_approximate_polygon(contour, epsilon_factor=0.02)
    pts_raw = approx.reshape(-1, 2)
    if len(pts_raw) < 3:
        hull = cv2.convexHull(contour)
        pts_raw = hull.reshape(-1, 2)
    if len(pts_raw) < 3:
        return None
    pts_list: List[Tuple[int, int]] = [tuple(p.tolist()) for p in pts_raw]
    if len(pts_list) == 3:
        pts_list = _bb_reconstruct_missing_corner(pts_list)
    if len(pts_list) > 4:
        pts_list = _bb_select_four_extreme_points(np.array(pts_list, dtype=np.int32))
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
        3 markers -> parallelogram rule for missing corner
        2 markers -> similarity transform reconstruction
        0-1 marker + vis_image -> blue-boundary fallback

    Key method for visualisation:
        compute_document_boundary_from_markers(all_corners_img)
            Derives the true page boundary directly from marker positions
            and known geometry -- NO dependency on the homography matrix.
    """

    # ------------------------------------------------------------------
    # Original document geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_original_marker_positions() -> Dict:
        """
        Return the center positions of the 4 ArUco markers in the original
        document coordinate system, keyed by corner name.
        """
        m = MarkerConfig.MARGIN
        s = MarkerConfig.MARKER_SIZE
        W = MarkerConfig.DOC_WIDTH
        H = MarkerConfig.DOC_HEIGHT
        return {
            "top_left": {
                "center": (m + s / 2, m + s / 2),
                "corners": [(m, m), (m+s, m), (m+s, m+s), (m, m+s)],
            },
            "top_right": {
                "center": (W - m - s/2, m + s/2),
                "corners": [(W-m-s, m), (W-m, m), (W-m, m+s), (W-m-s, m+s)],
            },
            "bottom_left": {
                "center": (m + s/2, H - m - s/2),
                "corners": [(m, H-m-s), (m+s, H-m-s), (m+s, H-m), (m, H-m)],
            },
            "bottom_right": {
                "center": (W - m - s/2, H - m - s/2),
                "corners": [(W-m-s, H-m-s), (W-m, H-m-s), (W-m, H-m), (W-m-s, H-m)],
            },
        }

    @staticmethod
    def calculate_marker_center(corners: np.ndarray) -> Tuple[float, float]:
        """Return the centroid of an ArUco marker corner array (shape 4x2)."""
        return (float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1])))

    # ------------------------------------------------------------------
    # Missing-corner estimation (ArUco path)
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_missing_corner_3_markers(
        detected: Dict[str, Tuple[float, float]]
    ) -> Tuple[str, Tuple[float, float]]:
        """
        Estimate the missing 4th corner using the parallelogram diagonal rule:

            TL + BR = TR + BL   ->   missing = A + C - B

        Exact for parallelograms; best linear approximation under perspective.

        Args:
            detected: dict mapping corner name -> (img_x, img_y) for 3 corners.

        Returns:
            Tuple (missing_corner_name, (estimated_x, estimated_y)).
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
        if missing_name == "top_left":       estimated = add(sub(tr, br), bl)
        elif missing_name == "top_right":    estimated = add(sub(tl, bl), br)
        elif missing_name == "bottom_right": estimated = add(sub(tr, tl), bl)
        else:                                estimated = add(sub(tl, tr), br)
        return missing_name, estimated

    @staticmethod
    def reconstruct_from_two_markers(
        detected: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Reconstruct all 4 corner positions from 2 detected markers using a
        similarity transform (rotation + uniform scale) built from the known
        document-space layout.

        The similarity matrix M satisfies  M @ (Q_doc - P_doc) = Q_img - P_img
        and is the unique rotation+scale solution for that single vector pair:

            M = (1/|u|^2) * [[ u.v,  -u_perp.v ],
                              [ u_perp.v,  u.v  ]]

        where u = Q_doc - P_doc,  v = Q_img - P_img,  u_perp = (-u_y, u_x).

        Args:
            detected: dict mapping corner name -> (img_x, img_y) for 2 corners.

        Returns:
            Dict mapping all 4 corner names -> (img_x, img_y).
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
        u = Q_doc - P_doc
        v = Q_img - P_img
        u_norm_sq = float(np.dot(u, u))
        if u_norm_sq < 1e-6:
            raise ValueError("The two detected markers are too close in document space.")
        u_perp = np.array([-u[1], u[0]], dtype=np.float64)
        dot_uv      = float(np.dot(u, v))
        dot_uperp_v = float(np.dot(u_perp, v))
        # 2x2 similarity matrix: doc_offset -> img_offset
        M = np.array([[dot_uv, -dot_uperp_v],
                      [dot_uperp_v, dot_uv]], dtype=np.float64) / u_norm_sq
        all_corners: Dict[str, Tuple[float, float]] = {}
        for corner_name in CORNER_ORDER:
            if corner_name in detected:
                all_corners[corner_name] = detected[corner_name]
            else:
                doc_center = np.array(
                    original_positions[corner_name]["center"], dtype=np.float64
                )
                estimated = P_img + M @ (doc_center - P_doc)
                all_corners[corner_name] = (float(estimated[0]), float(estimated[1]))
        return all_corners

    # ------------------------------------------------------------------
    # Geometric document boundary (core new method)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_document_boundary_from_markers(
        all_corners_img: Dict[str, Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Compute the true document boundary quad in IMAGE space directly from
        the four marker center positions and the known physical geometry.

        Geometric principle
        -------------------
        Each marker center is exactly:

            offset = MARGIN + MARKER_SIZE / 2

        away from the nearest true page edge along both axes.
        With MARGIN=24, MARKER_SIZE=60:  offset = 54 document-pixels.

        Under perspective, the document axes are no longer axis-aligned,
        so we must convert the offset into image-space displacement vectors.

        Step 1 -- Axis directions
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        Derive unit vectors for the horizontal and vertical document axes
        directly from the marker quad, averaging both parallel edges:

            h_raw = mean( (TR-TL), (BR-BL) )   -> horizontal axis
            v_raw = mean( (BL-TL), (BR-TR) )   -> vertical axis

        Averaging both parallel edges reduces single-marker noise and adapts
        to perspective foreshortening.

        Step 2 -- Scale  (image-px / doc-px)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Known center-to-center distances in document space:

            inter_h_doc = DOC_WIDTH  - 2 * offset  =  1654 - 108 = 1546
            inter_v_doc = DOC_HEIGHT - 2 * offset  =  2338 - 108 = 2230

        Measured in image space (average of both parallel edges):

            h_scale = mean(|TR-TL|, |BR-BL|) / inter_h_doc
            v_scale = mean(|BL-TL|, |BR-TR|) / inter_v_doc

        Step 3 -- Displacement vectors
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            delta_h = offset * h_scale * h_hat
            delta_v = offset * v_scale * v_hat

        Step 4 -- True document corners
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            doc_TL = marker_TL - delta_h - delta_v
            doc_TR = marker_TR + delta_h - delta_v
            doc_BR = marker_BR + delta_h + delta_v
            doc_BL = marker_BL - delta_h + delta_v

        Properties
        ----------
        * Exact when all 4 markers are detected.
        * Stable under 2/3-marker reconstruction because it depends only on
          image-space marker positions and known document geometry.
        * Never touches the homography matrix -> no extrapolation drift.
        * Correct under perspective: axis directions adapt to local skew.

        Args:
            all_corners_img: Dict mapping all 4 corner names to their
                             (x, y) image-space positions (real or estimated).

        Returns:
            List of 4 (x, y) float tuples ordered [TL, TR, BR, BL]
            representing the true document page boundary in image space.

        Raises:
            ValueError if a corner is missing or the layout is degenerate.
        """
        required = {"top_left", "top_right", "bottom_right", "bottom_left"}
        if not required.issubset(all_corners_img.keys()):
            raise ValueError(
                f"Need all 4 corners, got {set(all_corners_img.keys())}"
            )

        tl = np.array(all_corners_img["top_left"],     dtype=np.float64)
        tr = np.array(all_corners_img["top_right"],    dtype=np.float64)
        br = np.array(all_corners_img["bottom_right"], dtype=np.float64)
        bl = np.array(all_corners_img["bottom_left"],  dtype=np.float64)

        # Physical offset: marker center -> true page edge  (doc-px)
        # MARGIN=24, MARKER_SIZE=60  =>  offset = 54 doc-px
        offset_doc = MarkerConfig.MARGIN + MarkerConfig.MARKER_SIZE / 2.0

        # ---- Step 1: axis directions ------------------------------------
        # Horizontal: average top-edge vector and bottom-edge vector
        h_raw = ((tr - tl) + (br - bl)) / 2.0
        h_len = float(np.linalg.norm(h_raw))
        if h_len < 1e-6:
            raise ValueError("Degenerate marker layout: zero horizontal span.")
        h_hat = h_raw / h_len

        # Vertical: average left-edge vector and right-edge vector
        v_raw = ((bl - tl) + (br - tr)) / 2.0
        v_len = float(np.linalg.norm(v_raw))
        if v_len < 1e-6:
            raise ValueError("Degenerate marker layout: zero vertical span.")
        v_hat = v_raw / v_len

        # ---- Step 2: scale (img-px per doc-px) -------------------------
        inter_h_doc = MarkerConfig.DOC_WIDTH  - 2.0 * offset_doc   # 1546 doc-px
        inter_v_doc = MarkerConfig.DOC_HEIGHT - 2.0 * offset_doc   # 2230 doc-px

        h_span_img = (float(np.linalg.norm(tr - tl)) + float(np.linalg.norm(br - bl))) / 2.0
        v_span_img = (float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr))) / 2.0

        h_scale = h_span_img / inter_h_doc   # img-px / doc-px  (horizontal)
        v_scale = v_span_img / inter_v_doc   # img-px / doc-px  (vertical)

        # ---- Step 3: displacement vectors to true page edges ------------
        delta_h = offset_doc * h_scale * h_hat   # step from marker-x to left/right edge
        delta_v = offset_doc * v_scale * v_hat   # step from marker-y to top/bottom edge

        # ---- Step 4: true document corners in image space ---------------
        doc_tl = tl - delta_h - delta_v
        doc_tr = tr + delta_h - delta_v
        doc_br = br + delta_h + delta_v
        doc_bl = bl - delta_h + delta_v

        return [
            (float(doc_tl[0]), float(doc_tl[1])),
            (float(doc_tr[0]), float(doc_tr[1])),
            (float(doc_br[0]), float(doc_br[1])),
            (float(doc_bl[0]), float(doc_bl[1])),
        ]

    # ------------------------------------------------------------------
    # Blue-boundary homography fallback
    # ------------------------------------------------------------------

    @staticmethod
    def compute_homography_from_blue_boundary(
        vis_image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Last-resort fallback: recover homography from the blue polyline drawn
        in a visualisation image when fewer than 2 ArUco markers are available.
        """
        corners_img = extract_blue_boundary(vis_image)
        if corners_img is None:
            return None
        W = float(MarkerConfig.DOC_WIDTH)
        H = float(MarkerConfig.DOC_HEIGHT)
        src = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)
        dst = np.array(corners_img, dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(src, dst, 0)
        return homography_matrix

    # ------------------------------------------------------------------
    # Core homography computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_homography(
        detected_markers: List[Dict],
        corners_data: List[np.ndarray],
        vis_image: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute a homography matrix from detected ArUco markers.

        Dispatch order:
            n >= 4  -> standard findHomography + RANSAC
            n == 3  -> parallelogram rule, then findHomography
            n == 2  -> similarity transform, then findHomography
            n <= 1  -> blue-boundary fallback (requires vis_image)

        Args:
            detected_markers: List of dicts with at least a 'corner' key.
            corners_data:     Parallel list of ArUco corner arrays (N x 1 x 4 x 2).
            vis_image:        Optional BGR visualisation image for fallback.

        Returns:
            3x3 homography matrix or None on failure.
        """
        original_positions = CoordinateMapper.calculate_original_marker_positions()
        detected_img: Dict[str, Tuple[float, float]] = {}
        for i, marker_info in enumerate(detected_markers):
            corner_name = marker_info.get("corner")
            if corner_name not in original_positions:
                continue
            detected_img[corner_name] = CoordinateMapper.calculate_marker_center(
                corners_data[i][0]
            )

        n_valid = len(detected_img)

        if n_valid < 2:
            if vis_image is not None:
                return CoordinateMapper.compute_homography_from_blue_boundary(vis_image)
            return None

        if n_valid >= 4:
            all_img = detected_img
        elif n_valid == 3:
            missing_name, estimated = (
                CoordinateMapper.estimate_missing_corner_3_markers(detected_img)
            )
            all_img = dict(detected_img)
            all_img[missing_name] = estimated
        else:
            all_img = CoordinateMapper.reconstruct_from_two_markers(detected_img)

        src_points, dst_points = [], []
        for corner_name in CORNER_ORDER:
            if corner_name not in all_img:
                continue
            src_points.append(original_positions[corner_name]["center"])
            dst_points.append(all_img[corner_name])

        if len(src_points) < 4:
            return None

        method = cv2.RANSAC if n_valid >= 4 else 0
        homography_matrix, _ = cv2.findHomography(
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32),
            method, ransacReprojThreshold=5.0,
        )
        return homography_matrix

    @staticmethod
    def compute_homography_from_scan(
        scan_result: Dict,
        vis_image: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute homography from an ExamScanner.scan_page() result dict.

        Args:
            scan_result: Result dictionary from ExamScanner.scan_page().
            vis_image:   Optional BGR visualisation image for fallback.

        Returns:
            Homography matrix or None.
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
            vis_image:   Optional BGR visualisation image for fallback.
        """
        if not scan_result.get("success", False):
            return None
        if scan_result.get("markers_found", 0) < 2 and vis_image is None:
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
            image_array, homography_inv,
            (MarkerConfig.DOC_WIDTH, MarkerConfig.DOC_HEIGHT),
        )
        return Image.fromarray(cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def get_scale_factors(homography_matrix: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Estimate approximate scale factors between document and image space.

        Returns:
            Dict with scale_x, scale_y, average_scale, or None.
        """
        if homography_matrix is None:
            return None
        p1 = CoordinateMapper.map_point_to_image(0, 0, homography_matrix)
        p2 = CoordinateMapper.map_point_to_image(
            MarkerConfig.DOC_WIDTH, MarkerConfig.DOC_HEIGHT, homography_matrix
        )
        if p1 is None or p2 is None:
            return None
        scale_x = abs(p2[0] - p1[0]) / MarkerConfig.DOC_WIDTH
        scale_y = abs(p2[1] - p1[1]) / MarkerConfig.DOC_HEIGHT
        return {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "average_scale": (scale_x + scale_y) / 2,
        }