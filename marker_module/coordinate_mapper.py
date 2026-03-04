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
    The visible document boundary is derived directly from marker centres
    using the known physical offset (MARGIN + MARKER_SIZE/2), NOT by
    mapping abstract (0,0) corners through the homography.

Dewarping
    warpPerspective is called with a homography built as:
        src = the 4 boundary corners in IMAGE space   (where the paper is)
        dst = the 4 corners of the output canvas      (0,0 -> W,H)
    This maps image pixels -> flat document pixels directly, with no
    matrix inversion needed and no orientation ambiguity.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .marker_config import MarkerConfig
from PIL import Image


CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]


# ===========================================================================
# Blue-boundary detection helpers
# ===========================================================================

_BLUE_H_LOW  = 95
_BLUE_H_HIGH = 135
_BLUE_S_LOW  = 80
_BLUE_V_LOW  = 80


def _bb_threshold_blue(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([_BLUE_H_LOW,  _BLUE_S_LOW, _BLUE_V_LOW], dtype=np.uint8)
    upper = np.array([_BLUE_H_HIGH, 255,          255         ], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _bb_close_gaps(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _bb_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _bb_approximate_polygon(contour: np.ndarray, epsilon_factor: float = 0.02) -> np.ndarray:
    peri = cv2.arcLength(contour, closed=True)
    return cv2.approxPolyDP(contour, epsilon_factor * peri, closed=True)


def _bb_reconstruct_missing_corner(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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
    def add(a, b): return (a[0]+b[0], a[1]+b[1])
    def sub(a, b): return (a[0]-b[0], a[1]-b[1])
    if missing_name == "top_left":       est = add(sub(tr, br), bl)
    elif missing_name == "top_right":    est = add(sub(tl, bl), br)
    elif missing_name == "bottom_right": est = add(sub(tr, tl), bl)
    else:                                est = add(sub(tl, tr), br)
    quadrants[missing_name] = (int(est[0]), int(est[1]))
    return [quadrants["top_left"], quadrants["top_right"],
            quadrants["bottom_right"], quadrants["bottom_left"]]


def _bb_select_four_extreme_points(points: np.ndarray) -> List[Tuple[int, int]]:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    bbox = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]], dtype=np.float64)
    selected = []
    for bp in bbox:
        dists = np.linalg.norm(points.astype(np.float64) - bp, axis=1)
        selected.append(tuple(points[np.argmin(dists)].tolist()))
    return selected


def _bb_order_corners(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    arr = np.array(pts, dtype=np.int32)
    s = arr.sum(axis=1)
    d = np.diff(arr, axis=1).ravel()
    return [tuple(arr[np.argmin(s)].tolist()),
            tuple(arr[np.argmin(d)].tolist()),
            tuple(arr[np.argmax(s)].tolist()),
            tuple(arr[np.argmax(d)].tolist())]


def extract_blue_boundary(image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
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
    """

    @staticmethod
    def calculate_original_marker_positions() -> Dict:
        m = MarkerConfig.MARGIN
        s = MarkerConfig.MARKER_SIZE
        W = MarkerConfig.DOC_WIDTH
        H = MarkerConfig.DOC_HEIGHT
        return {
            "top_left":     {"center": (m+s/2,   m+s/2),   "corners": [(m,m),(m+s,m),(m+s,m+s),(m,m+s)]},
            "top_right":    {"center": (W-m-s/2, m+s/2),   "corners": [(W-m-s,m),(W-m,m),(W-m,m+s),(W-m-s,m+s)]},
            "bottom_left":  {"center": (m+s/2,   H-m-s/2), "corners": [(m,H-m-s),(m+s,H-m-s),(m+s,H-m),(m,H-m)]},
            "bottom_right": {"center": (W-m-s/2, H-m-s/2), "corners": [(W-m-s,H-m-s),(W-m,H-m-s),(W-m,H-m),(W-m-s,H-m)]},
        }

    @staticmethod
    def calculate_marker_center(corners: np.ndarray) -> Tuple[float, float]:
        return (float(np.mean(corners[:, 0])), float(np.mean(corners[:, 1])))

    @staticmethod
    def estimate_missing_corner_3_markers(
        detected: Dict[str, Tuple[float, float]]
    ) -> Tuple[str, Tuple[float, float]]:
        """Estimate missing 4th corner: TL+BR = TR+BL -> missing = A+C-B."""
        if len(detected) < 3:
            raise ValueError("Need at least 3 detected corners.")
        all_corners = {"top_left", "top_right", "bottom_right", "bottom_left"}
        missing_name = (all_corners - set(detected.keys())).pop()
        tl = detected.get("top_left")
        tr = detected.get("top_right")
        br = detected.get("bottom_right")
        bl = detected.get("bottom_left")
        def add(a, b): return (a[0]+b[0], a[1]+b[1])
        def sub(a, b): return (a[0]-b[0], a[1]-b[1])
        if missing_name == "top_left":       estimated = add(sub(tr, br), bl)
        elif missing_name == "top_right":    estimated = add(sub(tl, bl), br)
        elif missing_name == "bottom_right": estimated = add(sub(tr, tl), bl)
        else:                                estimated = add(sub(tl, tr), br)
        return missing_name, estimated

    @staticmethod
    def reconstruct_from_two_markers(
        detected: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """Reconstruct all 4 corners from 2 markers via similarity transform."""
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
            raise ValueError("Markers too close in document space.")
        u_perp = np.array([-u[1], u[0]], dtype=np.float64)
        dot_uv      = float(np.dot(u, v))
        dot_uperp_v = float(np.dot(u_perp, v))
        M = np.array([[dot_uv, -dot_uperp_v],
                      [dot_uperp_v, dot_uv]], dtype=np.float64) / u_norm_sq
        all_corners: Dict[str, Tuple[float, float]] = {}
        for corner_name in CORNER_ORDER:
            if corner_name in detected:
                all_corners[corner_name] = detected[corner_name]
            else:
                doc_center = np.array(original_positions[corner_name]["center"], dtype=np.float64)
                estimated  = P_img + M @ (doc_center - P_doc)
                all_corners[corner_name] = (float(estimated[0]), float(estimated[1]))
        return all_corners

    # ------------------------------------------------------------------
    # Document boundary from marker geometry
    # ------------------------------------------------------------------

    @staticmethod
    def compute_document_boundary_from_markers(
        all_corners_img: Dict[str, Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Compute true document boundary corners in IMAGE space.

        Each marker centre is offset = MARGIN + MARKER_SIZE/2 = 54 doc-px
        from the nearest page edge. We step each marker outward along the
        locally measured axis directions by this offset (scaled to image-px).

        Returns [TL, TR, BR, BL] as float tuples in image space.
        """
        required = {"top_left", "top_right", "bottom_right", "bottom_left"}
        if not required.issubset(all_corners_img.keys()):
            raise ValueError(f"Need all 4 corners, got {set(all_corners_img.keys())}")

        tl = np.array(all_corners_img["top_left"],     dtype=np.float64)
        tr = np.array(all_corners_img["top_right"],    dtype=np.float64)
        br = np.array(all_corners_img["bottom_right"], dtype=np.float64)
        bl = np.array(all_corners_img["bottom_left"],  dtype=np.float64)

        offset_doc = MarkerConfig.MARGIN + MarkerConfig.MARKER_SIZE / 2.0  # 54

        h_raw = ((tr - tl) + (br - bl)) / 2.0
        h_len = float(np.linalg.norm(h_raw))
        if h_len < 1e-6:
            raise ValueError("Degenerate layout: zero horizontal span.")
        h_hat = h_raw / h_len

        v_raw = ((bl - tl) + (br - tr)) / 2.0
        v_len = float(np.linalg.norm(v_raw))
        if v_len < 1e-6:
            raise ValueError("Degenerate layout: zero vertical span.")
        v_hat = v_raw / v_len

        inter_h_doc = MarkerConfig.DOC_WIDTH  - 2.0 * offset_doc   # 1546
        inter_v_doc = MarkerConfig.DOC_HEIGHT - 2.0 * offset_doc   # 2230

        h_scale = (
            (float(np.linalg.norm(tr - tl)) + float(np.linalg.norm(br - bl))) / 2.0
        ) / inter_h_doc
        v_scale = (
            (float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr))) / 2.0
        ) / inter_v_doc

        delta_h = offset_doc * h_scale * h_hat
        delta_v = offset_doc * v_scale * v_hat

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
    # Homography (doc-space -> image-space, used for point mapping only)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_homography(
        detected_markers: List[Dict],
        corners_data: List[np.ndarray],
        vis_image: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute H such that  H @ doc_point -> image_point.

        Used by map_point_to_image / map_points_to_image for overlay drawing.
        NOT used for dewarping (see dewarp_document instead).
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
            return None

        if n_valid >= 4:
            all_img = detected_img
        elif n_valid == 3:
            missing_name, estimated = CoordinateMapper.estimate_missing_corner_3_markers(detected_img)
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
        H, _ = cv2.findHomography(
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32),
            method, ransacReprojThreshold=5.0,
        )
        return H

    @staticmethod
    def compute_homography_from_scan(
        scan_result: Dict,
        vis_image: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if not scan_result.get("success", False):
            return None
        return CoordinateMapper.compute_homography(
            scan_result["detected_markers"],
            scan_result["corners"],
            vis_image=vis_image,
        )

    # ------------------------------------------------------------------
    # Dewarping  (the correct approach)
    # ------------------------------------------------------------------

    @staticmethod
    def dewarp_document(
        image: np.ndarray,
        all_corners_img: Dict[str, Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """
        Perspective-correct the document using the 4 boundary corner positions
        in image space.

        Approach
        --------
        Build H as:
            src = 4 document-boundary corners in IMAGE space  (where the paper is)
            dst = 4 corners of the output canvas              (0,0 -> W,H)

        Then call warpPerspective(image, H, (W, H)).

        This directly maps image pixels to their correct location in the
        flattened output -- no matrix inversion, no orientation ambiguity.

        The output canvas is always DOC_WIDTH x DOC_HEIGHT (portrait).
        If the boundary corners indicate the paper is landscape in the photo
        (horizontal span > vertical span), we rotate the output 90 degrees
        so the result is always upright.

        Args:
            image:           BGR numpy array of the original photo.
            all_corners_img: Dict with all 4 corner names -> (x, y) in image space.

        Returns:
            BGR numpy array of shape (DOC_HEIGHT, DOC_WIDTH), or None on failure.
        """
        try:
            boundary = CoordinateMapper.compute_document_boundary_from_markers(
                all_corners_img
            )
        except ValueError:
            return None

        # boundary = [TL, TR, BR, BL] in image space
        src = np.array(boundary, dtype=np.float32)   # 4 x 2, image coords

        W = float(MarkerConfig.DOC_WIDTH)
        H = float(MarkerConfig.DOC_HEIGHT)

        # Detect whether the paper appears landscape in the photo.
        # Use the horizontal and vertical span of the boundary quad.
        tl, tr, br, bl = [np.array(p, dtype=np.float64) for p in boundary]
        h_span = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        v_span = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        landscape_in_photo = h_span < v_span   # paper taller than wide -> portrait orientation

        if landscape_in_photo:
            # Paper is upright in the photo -> output straight portrait
            dst = np.array([
                [0,   0  ],   # TL -> top-left of canvas
                [W,   0  ],   # TR -> top-right
                [W,   H  ],   # BR -> bottom-right
                [0,   H  ],   # BL -> bottom-left
            ], dtype=np.float32)
            out_size = (int(W), int(H))
        else:
            # Paper is on its side in the photo -> rotate output 90 CW
            # so the long axis of the document becomes the vertical of the output
            dst = np.array([
                [H,   0  ],   # TL -> top-right of rotated canvas
                [H,   W  ],   # TR -> bottom-right
                [0,   W  ],   # BR -> bottom-left
                [0,   0  ],   # BL -> top-left
            ], dtype=np.float32)
            out_size = (int(H), int(W))

        H_mat, _ = cv2.findHomography(src, dst, 0)
        if H_mat is None:
            return None

        warped = cv2.warpPerspective(image, H_mat, out_size,
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))

        # Always return portrait (DOC_WIDTH x DOC_HEIGHT)
        if warped.shape[1] > warped.shape[0]:   # still landscape
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return warped

    # ------------------------------------------------------------------
    # extract_full_document (public API, wraps dewarp_document)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_full_document(
        image: Image.Image,
        scan_result: Dict,
        all_corners_img: Optional[Dict[str, Tuple[float, float]]] = None,
        vis_image: Optional[np.ndarray] = None,
    ) -> Optional[Image.Image]:
        """
        Extract and perspective-correct the document from a captured photo.

        Args:
            image:           PIL RGB image of the raw captured photo.
            scan_result:     Result dict from ExamScanner.scan_page().
            all_corners_img: Dict with all 4 corner names -> (x, y) in image
                             space (real + estimated). If None, will be computed
                             from scan_result markers.
            vis_image:       Unused (kept for API compatibility).

        Returns:
            PIL Image, size DOC_WIDTH x DOC_HEIGHT, portrait orientation.
            None on failure.
        """
        if not scan_result.get("success", False):
            return None
        if scan_result.get("markers_found", 0) < 2:
            return None

        # Build all_corners_img from scan_result if not provided
        if all_corners_img is None:
            original_positions = CoordinateMapper.calculate_original_marker_positions()
            detected_img: Dict[str, Tuple[float, float]] = {}
            for i, marker_info in enumerate(scan_result["detected_markers"]):
                cn = marker_info.get("corner")
                if cn not in original_positions:
                    continue
                detected_img[cn] = CoordinateMapper.calculate_marker_center(
                    scan_result["corners"][i][0]
                )
            n = len(detected_img)
            if n < 2:
                return None
            if n >= 4:
                all_corners_img = detected_img
            elif n == 3:
                mn, mp = CoordinateMapper.estimate_missing_corner_3_markers(detected_img)
                all_corners_img = dict(detected_img)
                all_corners_img[mn] = mp
            else:
                all_corners_img = CoordinateMapper.reconstruct_from_two_markers(detected_img)

        image_bgr = np.array(image)
        if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

        warped = CoordinateMapper.dewarp_document(image_bgr, all_corners_img)
        if warped is None:
            return None

        # Ensure output is exactly DOC_WIDTH x DOC_HEIGHT
        target_w = MarkerConfig.DOC_WIDTH
        target_h = MarkerConfig.DOC_HEIGHT
        if warped.shape[1] != target_w or warped.shape[0] != target_h:
            warped = cv2.resize(warped, (target_w, target_h),
                                interpolation=cv2.INTER_LINEAR)

        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    # ------------------------------------------------------------------
    # Blue-boundary fallback
    # ------------------------------------------------------------------

    @staticmethod
    def compute_homography_from_blue_boundary(
        vis_image: np.ndarray,
    ) -> Optional[np.ndarray]:
        corners_img = extract_blue_boundary(vis_image)
        if corners_img is None:
            return None
        W = float(MarkerConfig.DOC_WIDTH)
        H = float(MarkerConfig.DOC_HEIGHT)
        src = np.array([(0,0),(W,0),(W,H),(0,H)], dtype=np.float32)
        dst = np.array(corners_img, dtype=np.float32)
        H_mat, _ = cv2.findHomography(src, dst, 0)
        return H_mat

    # ------------------------------------------------------------------
    # Coordinate mapping utilities (for overlay / zone mapping)
    # ------------------------------------------------------------------

    @staticmethod
    def map_point_to_image(
        doc_x: float, doc_y: float, homography_matrix: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        if homography_matrix is None:
            return None
        t = cv2.perspectiveTransform(
            np.array([[[doc_x, doc_y]]], dtype=np.float32), homography_matrix
        )
        return (float(t[0][0][0]), float(t[0][0][1]))

    @staticmethod
    def map_points_to_image(
        doc_points: List[Tuple[float, float]], homography_matrix: np.ndarray
    ) -> Optional[List[Tuple[float, float]]]:
        if homography_matrix is None:
            return None
        t = cv2.perspectiveTransform(
            np.array([doc_points], dtype=np.float32), homography_matrix
        )
        return [(float(x), float(y)) for x, y in t[0]]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def get_scale_factors(homography_matrix: np.ndarray) -> Optional[Dict[str, float]]:
        if homography_matrix is None:
            return None
        p1 = CoordinateMapper.map_point_to_image(0, 0, homography_matrix)
        p2 = CoordinateMapper.map_point_to_image(
            MarkerConfig.DOC_WIDTH, MarkerConfig.DOC_HEIGHT, homography_matrix
        )
        if p1 is None or p2 is None:
            return None
        sx = abs(p2[0]-p1[0]) / MarkerConfig.DOC_WIDTH
        sy = abs(p2[1]-p1[1]) / MarkerConfig.DOC_HEIGHT
        return {"scale_x": sx, "scale_y": sy, "average_scale": (sx+sy)/2}