"""
coordinate_mapper.py
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


def _bb_reconstruct_missing_corner(pts):
    arr = np.array(pts, dtype=np.float64)
    cx, cy = arr.mean(axis=0)
    quadrants = {}
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


def _bb_select_four_extreme_points(points: np.ndarray):
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    bbox = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]], dtype=np.float64)
    selected = []
    for bp in bbox:
        dists = np.linalg.norm(points.astype(np.float64) - bp, axis=1)
        selected.append(tuple(points[np.argmin(dists)].tolist()))
    return selected


def _bb_order_corners(pts):
    arr = np.array(pts, dtype=np.int32)
    s = arr.sum(axis=1)
    d = np.diff(arr, axis=1).ravel()
    return [tuple(arr[np.argmin(s)].tolist()),
            tuple(arr[np.argmin(d)].tolist()),
            tuple(arr[np.argmax(s)].tolist()),
            tuple(arr[np.argmax(d)].tolist())]


def extract_blue_boundary(image: np.ndarray):
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
    pts_list = [tuple(p.tolist()) for p in pts_raw]
    if len(pts_list) == 3:
        pts_list = _bb_reconstruct_missing_corner(pts_list)
    if len(pts_list) > 4:
        pts_list = _bb_select_four_extreme_points(np.array(pts_list, dtype=np.int32))
    if len(pts_list) != 4:
        return None
    return _bb_order_corners(pts_list)


# ===========================================================================
# Marker validation
# ===========================================================================

# Expected image-space quadrant for each corner name.
# e.g. top_left must be in the LEFT half (x < cx) AND TOP half (y < cy).
_CORNER_QUADRANT = {
    "top_left":     ("left",  "top"),
    "top_right":    ("right", "top"),
    "bottom_right": ("right", "bottom"),
    "bottom_left":  ("left",  "bottom"),
}


def filter_markers_by_image_quadrant(
    raw: Dict[str, Tuple[float, float]],
    image_w: int,
    image_h: int,
    tolerance: float = 0.20,
) -> Dict[str, Tuple[float, float]]:
    """
    Remove any detected marker whose image-space position does not match
    the expected quadrant for its assigned corner name.

    Each corner must lie in its correct half of the image:
        top_left     -> x < cx  AND  y < cy
        top_right    -> x > cx  AND  y < cy
        bottom_right -> x > cx  AND  y > cy
        bottom_left  -> x < cx  AND  y > cy

    A tolerance of 20% of the image dimension is allowed so that a slightly
    off-centre document still passes.

    This is the primary defence against markers from other pages in a stack
    being assigned the wrong corner role.

    Args:
        raw:       detected corners {name: (x, y)}
        image_w:   full image width  in pixels
        image_h:   full image height in pixels
        tolerance: fraction of image size allowed past the centre line

    Returns:
        Filtered dict containing only geometrically plausible corners.
    """
    cx = image_w / 2.0
    cy = image_h / 2.0
    tol_x = tolerance * image_w
    tol_y = tolerance * image_h

    kept: Dict[str, Tuple[float, float]] = {}
    for name, (x, y) in raw.items():
        expected_h, expected_v = _CORNER_QUADRANT.get(name, (None, None))
        if expected_h is None:
            continue

        ok_h = (x <= cx + tol_x) if expected_h == "left" else (x >= cx - tol_x)
        ok_v = (y <= cy + tol_y) if expected_v == "top"  else (y >= cy - tol_y)

        if ok_h and ok_v:
            kept[name] = (x, y)
        else:
            print(f"  [quadrant_filter] Rejected '{name}' at ({x:.0f},{y:.0f}) "
                  f"-- expected {expected_h}/{expected_v} quadrant "
                  f"(image centre {cx:.0f},{cy:.0f})")
    return kept


def _estimate_missing(
    detected: Dict[str, Tuple[float, float]]
) -> Tuple[str, Tuple[float, float]]:
    """Estimate the one missing corner from 3 known ones using diagonal rule."""
    all_corners = {"top_left", "top_right", "bottom_right", "bottom_left"}
    missing_name = (all_corners - set(detected.keys())).pop()
    tl = detected.get("top_left")
    tr = detected.get("top_right")
    br = detected.get("bottom_right")
    bl = detected.get("bottom_left")
    def add(a, b): return (a[0]+b[0], a[1]+b[1])
    def sub(a, b): return (a[0]-b[0], a[1]-b[1])
    if missing_name == "top_left":       est = add(sub(tr, br), bl)
    elif missing_name == "top_right":    est = add(sub(tl, bl), br)
    elif missing_name == "bottom_right": est = add(sub(tr, tl), bl)
    else:                                est = add(sub(tl, tr), br)
    return missing_name, est


def _is_convex_quad(pts: Dict[str, Tuple[float, float]]) -> bool:
    """Return True if the 4 corners form a convex quad in correct spatial order."""
    required = {"top_left", "top_right", "bottom_right", "bottom_left"}
    if not required.issubset(pts.keys()):
        return False
    tl = np.array(pts["top_left"],     dtype=np.float64)
    tr = np.array(pts["top_right"],    dtype=np.float64)
    br = np.array(pts["bottom_right"], dtype=np.float64)
    bl = np.array(pts["bottom_left"],  dtype=np.float64)
    poly = [tl, tr, br, bl]
    cross_signs = []
    n = len(poly)
    for i in range(n):
        p0, p1, p2 = poly[i], poly[(i+1)%n], poly[(i+2)%n]
        cross = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0])
        cross_signs.append(np.sign(cross))
    return len(set(cross_signs)) == 1


# ===========================================================================
# CoordinateMapper
# ===========================================================================

class CoordinateMapper:

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
        if len(detected) < 3:
            raise ValueError("Need at least 3 detected corners.")
        return _estimate_missing(detected)

    @staticmethod
    def reconstruct_from_two_markers(
        detected: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Tuple[float, float]]:
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
    # Core: resolve + validate all 4 corner positions
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_corners(
        detected_markers: List[Dict],
        corners_data: List[np.ndarray],
        image_w: int = 0,
        image_h: int = 0,
    ) -> Tuple[Dict[str, Tuple[float, float]], set]:
        """
        Collect detected marker centres, validate them with quadrant filtering,
        then fill any missing corners by geometric estimation.

        Validation pipeline
        -------------------
        Step 1 — Quadrant filter (primary defence):
            Each corner is only kept if it lies in the correct spatial quadrant
            of the image. A marker labelled 'bottom_left' that appears in the
            top half of the photo is rejected immediately, regardless of which
            ArUco ID was decoded. This eliminates markers from other pages in
            a stack, which are the most common source of false detections.

        Step 2 — Convexity check (secondary defence):
            If all 4 survived step 1, verify they form a convex quad. If not,
            try dropping each one and re-estimating; keep the set that produces
            the most convex arrangement.

        Step 3 — Estimation:
            Fill any remaining missing corners (3 known -> parallelogram rule,
            2 known -> similarity transform).

        Args:
            detected_markers: List of marker dicts with 'corner' key.
            corners_data:     Parallel list of ArUco corner arrays.
            image_w / image_h: Image dimensions for quadrant filtering.
                               If 0, quadrant filter is skipped.

        Returns:
            (all_corners_img, estimated_names)
        """
        original_positions = CoordinateMapper.calculate_original_marker_positions()

        # Collect raw detections
        raw: Dict[str, Tuple[float, float]] = {}
        for i, mi in enumerate(detected_markers):
            cn = mi.get("corner")
            if cn not in original_positions:
                continue
            raw[cn] = CoordinateMapper.calculate_marker_center(corners_data[i][0])

        print(f"  Raw detections: {sorted(raw.keys())}")

        # Step 1: quadrant filter
        if image_w > 0 and image_h > 0:
            clean = filter_markers_by_image_quadrant(raw, image_w, image_h)
        else:
            clean = dict(raw)

        print(f"  After quadrant filter: {sorted(clean.keys())}")

        # Step 2: convexity check on 4 markers
        if len(clean) == 4 and not _is_convex_quad(clean):
            print("  4 markers not convex — trying to drop outlier...")
            best, best_score = None, float('inf')
            for drop in list(clean.keys()):
                candidate = {k: v for k, v in clean.items() if k != drop}
                mn, mp = _estimate_missing(candidate)
                test = dict(candidate); test[mn] = mp
                if _is_convex_quad(test):
                    # Score: distance of dropped marker from its estimated position
                    score = np.linalg.norm(np.array(clean[drop]) - np.array(test[drop]))
                    if score < best_score:
                        best_score = score
                        best = (candidate, drop)
            if best is not None:
                clean, dropped = best
                print(f"  Dropped outlier '{dropped}' (err={best_score:.1f}px)")

        n = len(clean)
        if n < 2:
            raise ValueError(f"Only {n} valid marker(s) after filtering — cannot proceed.")

        # Step 3: estimate missing corners
        estimated_names: set = set()
        all_corners_img: Dict[str, Tuple[float, float]] = {}

        if n >= 4:
            all_corners_img = dict(clean)
        elif n == 3:
            mn, mp = _estimate_missing(clean)
            all_corners_img = dict(clean)
            all_corners_img[mn] = mp
            estimated_names.add(mn)
        else:  # n == 2
            all_corners_img = CoordinateMapper.reconstruct_from_two_markers(clean)
            estimated_names = set(all_corners_img.keys()) - set(clean.keys())

        return all_corners_img, estimated_names

    # ------------------------------------------------------------------
    # Document boundary from marker geometry
    # ------------------------------------------------------------------

    @staticmethod
    def compute_document_boundary_from_markers(
        all_corners_img: Dict[str, Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Step each marker outward by MARGIN + MARKER_SIZE/2 along the locally
        measured document axes to reach the true page edges.
        Returns [TL, TR, BR, BL] in image space.
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

        inter_h_doc = MarkerConfig.DOC_WIDTH  - 2.0 * offset_doc
        inter_v_doc = MarkerConfig.DOC_HEIGHT - 2.0 * offset_doc

        h_scale = (
            (float(np.linalg.norm(tr - tl)) + float(np.linalg.norm(br - bl))) / 2.0
        ) / inter_h_doc
        v_scale = (
            (float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr))) / 2.0
        ) / inter_v_doc

        delta_h = offset_doc * h_scale * h_hat
        delta_v = offset_doc * v_scale * v_hat

        return [
            (float((tl - delta_h - delta_v)[0]), float((tl - delta_h - delta_v)[1])),
            (float((tr + delta_h - delta_v)[0]), float((tr + delta_h - delta_v)[1])),
            (float((br + delta_h + delta_v)[0]), float((br + delta_h + delta_v)[1])),
            (float((bl - delta_h + delta_v)[0]), float((bl - delta_h + delta_v)[1])),
        ]

    # ------------------------------------------------------------------
    # Homography (doc -> image, for point/zone mapping only)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_homography(
        detected_markers: List[Dict],
        corners_data: List[np.ndarray],
        image_w: int = 0,
        image_h: int = 0,
    ) -> Optional[np.ndarray]:
        """Compute H: doc_point -> image_point (for overlay use, not dewarping)."""
        original_positions = CoordinateMapper.calculate_original_marker_positions()
        try:
            all_img, _ = CoordinateMapper.resolve_corners(
                detected_markers, corners_data, image_w, image_h
            )
        except ValueError:
            return None

        src_points, dst_points = [], []
        for corner_name in CORNER_ORDER:
            if corner_name not in all_img:
                continue
            src_points.append(original_positions[corner_name]["center"])
            dst_points.append(all_img[corner_name])

        if len(src_points) < 4:
            return None

        H, _ = cv2.findHomography(
            np.array(src_points, dtype=np.float32),
            np.array(dst_points, dtype=np.float32),
            0,
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
        )

    # ------------------------------------------------------------------
    # Dewarping
    # ------------------------------------------------------------------

    @staticmethod
    def dewarp_document(
        image: np.ndarray,
        all_corners_img: Dict[str, Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """
        Perspective-correct the document.
        src = boundary corners in image space -> dst = canvas corners.
        Auto-corrects portrait/landscape orientation.
        Returns BGR array (DOC_HEIGHT x DOC_WIDTH).
        """
        try:
            boundary = CoordinateMapper.compute_document_boundary_from_markers(
                all_corners_img
            )
        except ValueError:
            return None

        src = np.array(boundary, dtype=np.float32)  # [TL, TR, BR, BL]

        W = float(MarkerConfig.DOC_WIDTH)
        H = float(MarkerConfig.DOC_HEIGHT)

        tl, tr, br, bl = [np.array(p, dtype=np.float64) for p in boundary]
        h_span = (np.linalg.norm(tr-tl) + np.linalg.norm(br-bl)) / 2.0
        v_span = (np.linalg.norm(bl-tl) + np.linalg.norm(br-tr)) / 2.0
        portrait = v_span >= h_span

        if portrait:
            dst      = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
            out_size = (int(W), int(H))
        else:
            dst      = np.array([[0,0],[0,H],[W,H],[W,0]], dtype=np.float32)
            out_size = (int(H), int(W))

        H_mat, _ = cv2.findHomography(src, dst, 0)
        if H_mat is None:
            return None

        warped = cv2.warpPerspective(image, H_mat, out_size,
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))

        if warped.shape[1] > warped.shape[0]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        if warped.shape[1] != int(W) or warped.shape[0] != int(H):
            warped = cv2.resize(warped, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)

        return warped

    @staticmethod
    def extract_full_document(
        image: Image.Image,
        scan_result: Dict,
        all_corners_img: Optional[Dict[str, Tuple[float, float]]] = None,
        vis_image: Optional[np.ndarray] = None,
    ) -> Optional[Image.Image]:
        if not scan_result.get("success", False):
            return None
        if all_corners_img is None:
            try:
                all_corners_img, _ = CoordinateMapper.resolve_corners(
                    scan_result["detected_markers"],
                    scan_result["corners"],
                )
            except ValueError:
                return None

        image_bgr = np.array(image)
        if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

        warped = CoordinateMapper.dewarp_document(image_bgr, all_corners_img)
        if warped is None:
            return None
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    # ------------------------------------------------------------------
    # Blue-boundary fallback
    # ------------------------------------------------------------------

    @staticmethod
    def compute_homography_from_blue_boundary(vis_image: np.ndarray) -> Optional[np.ndarray]:
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
    # Point mapping (for OCR zone overlay)
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