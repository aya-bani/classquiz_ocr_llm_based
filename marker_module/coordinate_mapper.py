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


def _marker_center_expected_ratio() -> float:
    offset = MarkerConfig.MARGIN + MarkerConfig.MARKER_SIZE / 2.0
    w = MarkerConfig.DOC_WIDTH - 2.0 * offset
    h = MarkerConfig.DOC_HEIGHT - 2.0 * offset
    return float(w / h)


def _quad_angle_error(pts: Dict[str, Tuple[float, float]]) -> float:
    tl = np.array(pts["top_left"], dtype=np.float64)
    tr = np.array(pts["top_right"], dtype=np.float64)
    br = np.array(pts["bottom_right"], dtype=np.float64)
    bl = np.array(pts["bottom_left"], dtype=np.float64)

    def corner_err(prev_pt, pt, next_pt):
        v1 = prev_pt - pt
        v2 = next_pt - pt
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            return 1.0
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        return abs(cosang)

    errs = [
        corner_err(bl, tl, tr),
        corner_err(tl, tr, br),
        corner_err(tr, br, bl),
        corner_err(br, bl, tl),
    ]
    return float(np.mean(np.array(errs, dtype=np.float64)))


def _quad_aspect_ratio(pts: Dict[str, Tuple[float, float]]) -> float:
    tl = np.array(pts["top_left"], dtype=np.float64)
    tr = np.array(pts["top_right"], dtype=np.float64)
    br = np.array(pts["bottom_right"], dtype=np.float64)
    bl = np.array(pts["bottom_left"], dtype=np.float64)
    w = (float(np.linalg.norm(tr - tl)) + float(np.linalg.norm(br - bl))) / 2.0
    h = (float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr))) / 2.0
    return float(w / max(h, 1e-6))


def _quad_distance_scale_error(pts: Dict[str, Tuple[float, float]]) -> float:
    tl = np.array(pts["top_left"], dtype=np.float64)
    tr = np.array(pts["top_right"], dtype=np.float64)
    br = np.array(pts["bottom_right"], dtype=np.float64)
    bl = np.array(pts["bottom_left"], dtype=np.float64)

    offset = MarkerConfig.MARGIN + MarkerConfig.MARKER_SIZE / 2.0
    inter_w = MarkerConfig.DOC_WIDTH - 2.0 * offset
    inter_h = MarkerConfig.DOC_HEIGHT - 2.0 * offset
    if inter_w <= 1e-6 or inter_h <= 1e-6:
        return 1.0

    sx = ((float(np.linalg.norm(tr - tl)) + float(np.linalg.norm(br - bl))) / 2.0) / inter_w
    sy = ((float(np.linalg.norm(bl - tl)) + float(np.linalg.norm(br - tr))) / 2.0) / inter_h
    return abs(float(np.log((sx + 1e-6) / (sy + 1e-6))))


def _geometry_error_score(pts: Dict[str, Tuple[float, float]]) -> float:
    expected_ratio = _marker_center_expected_ratio()
    aspect = _quad_aspect_ratio(pts)
    aspect_err = abs(float(np.log((aspect + 1e-6) / (expected_ratio + 1e-6))))
    angle_err = _quad_angle_error(pts)
    dist_err = _quad_distance_scale_error(pts)
    return dist_err + angle_err + aspect_err


def _is_plausible_quad(
    pts: Dict[str, Tuple[float, float]],
    aspect_tol_log: float = 0.45,
    max_angle_err: float = 0.55,
    max_scale_err: float = 0.75,
) -> bool:
    if not _is_convex_quad(pts):
        return False

    expected_ratio = _marker_center_expected_ratio()
    aspect = _quad_aspect_ratio(pts)
    aspect_err = abs(float(np.log((aspect + 1e-6) / (expected_ratio + 1e-6))))
    if aspect_err > aspect_tol_log:
        return False

    if _quad_angle_error(pts) > max_angle_err:
        return False

    if _quad_distance_scale_error(pts) > max_scale_err:
        return False

    return True


def _refine_marker_center_subpixel(
    marker_corners: np.ndarray,
    gray: Optional[np.ndarray],
) -> Tuple[float, float]:
    pts = marker_corners.astype(np.float32).reshape(-1, 2)

    if gray is not None and gray.size > 0:
        sp = pts.reshape(-1, 1, 2).copy()
        term = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.001,
        )
        cv2.cornerSubPix(gray, sp, (5, 5), (-1, -1), term)
        pts = sp.reshape(-1, 2)

    mean_center = np.mean(pts, axis=0)
    moments = cv2.moments(pts.astype(np.float32))
    if abs(moments["m00"]) > 1e-6:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        center = 0.5 * mean_center + 0.5 * np.array([cx, cy], dtype=np.float32)
    else:
        center = mean_center

    return (float(center[0]), float(center[1]))


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

    @staticmethod
    def reconstruct_from_one_marker(
        detected: Dict[str, Tuple[float, float]],
        image_w: int,
        image_h: int,
    ) -> Dict[str, Tuple[float, float]]:
        """Estimate all 4 corners from a single detected marker.

        Uses the known document aspect ratio and the marker's expected
        position to compute a uniform scale + translation mapping from
        document space to image space (no rotation assumed).
        """
        original_positions = CoordinateMapper.calculate_original_marker_positions()
        name = list(detected.keys())[0]
        P_img = np.array(detected[name], dtype=np.float64)
        P_doc = np.array(original_positions[name]["center"], dtype=np.float64)

        W = float(MarkerConfig.DOC_WIDTH)
        H = float(MarkerConfig.DOC_HEIGHT)

        # Estimate scale from image dimensions vs document dimensions
        scale = min(image_w / W, image_h / H)

        # Translation: P_img = scale * P_doc + offset
        offset = P_img - scale * P_doc

        all_corners: Dict[str, Tuple[float, float]] = {}
        for corner_name in CORNER_ORDER:
            doc_center = np.array(original_positions[corner_name]["center"], dtype=np.float64)
            est = scale * doc_center + offset
            all_corners[corner_name] = (float(est[0]), float(est[1]))
        # Keep the original detected position
        all_corners[name] = detected[name]
        return all_corners

    @staticmethod
    def resolve_corners_from_blue_boundary(
        image_bgr: np.ndarray,
    ) -> Optional[Tuple[Dict[str, Tuple[float, float]], set]]:
        """Fallback: detect 4 corners from blue boundary lines in the image."""
        corners_img = extract_blue_boundary(image_bgr)
        if corners_img is None:
            return None
        names = ["top_left", "top_right", "bottom_right", "bottom_left"]
        all_corners = {n: (float(c[0]), float(c[1])) for n, c in zip(names, corners_img)}
        estimated_names = set(names)  # all from boundary, none from markers
        return all_corners, estimated_names

    @staticmethod
    def resolve_corners_from_contour(
        image_bgr: np.ndarray,
    ) -> Optional[Tuple[Dict[str, Tuple[float, float]], set]]:
        """Last-resort fallback: detect paper edges via grayscale contour/Otsu."""
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=5)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.15 * h * w:
            return None

        hull = cv2.convexHull(largest)
        approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
        if len(approx) != 4:
            # Try with larger epsilon
            approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
        if len(approx) != 4:
            return None

        pts = approx.reshape(4, 2).astype(np.float64)
        ordered = _bb_order_corners([tuple(p.tolist()) for p in pts])

        names = ["top_left", "top_right", "bottom_right", "bottom_left"]
        all_corners = {n: (float(c[0]), float(c[1])) for n, c in zip(names, ordered)}
        return all_corners, set(names)

    # ------------------------------------------------------------------
    # Core: resolve + validate all 4 corner positions
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_corners(
        detected_markers: List[Dict],
        corners_data: List[np.ndarray],
        image_w: int = 0,
        image_h: int = 0,
        image_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Tuple[float, float]], set]:
        """
        Collect detected marker centres, validate them with quadrant filtering,
        then fill any missing corners by geometric estimation.

        Validation pipeline
        -------------------
        Step 1 — Quadrant filter (primary defence):
            Each corner is only kept if it lies in the correct spatial quadrant
            of the image.

        Step 2 — Convexity check (secondary defence):
            If all 4 survived step 1, verify they form a convex quad. If not,
            try dropping each one and re-estimating.

        Step 3 — Estimation:
            4 markers -> use directly
            3 markers -> parallelogram rule
            2 markers -> similarity transform
            1 marker  -> scale + translate from document layout

        Step 4 — Fallbacks (if < 1 marker or image provided):
            a) Blue boundary detection (HSV thresholding)
            b) Contour-based paper detection (Otsu + convex hull)

        Args:
            detected_markers: List of marker dicts with 'corner' key.
            corners_data:     Parallel list of ArUco corner arrays.
            image_w / image_h: Image dimensions for quadrant filtering.
            image_bgr:         Original BGR image for fallback detection.

        Returns:
            (all_corners_img, estimated_names)
        """
        original_positions = CoordinateMapper.calculate_original_marker_positions()

        gray = None
        if image_bgr is not None and len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        elif image_bgr is not None:
            gray = image_bgr

        # Collect raw detections
        raw: Dict[str, Tuple[float, float]] = {}
        for i, mi in enumerate(detected_markers):
            cn = mi.get("corner")
            if cn not in original_positions:
                continue
            raw[cn] = _refine_marker_center_subpixel(corners_data[i][0], gray)

        print(f"  Raw detections: {sorted(raw.keys())}")

        # Step 1: quadrant filter
        if image_w > 0 and image_h > 0:
            clean = filter_markers_by_image_quadrant(raw, image_w, image_h)
        else:
            clean = dict(raw)

        print(f"  After quadrant filter: {sorted(clean.keys())}")

        # Step 2: full geometry consistency on 4 markers
        if len(clean) == 4 and not _is_plausible_quad(clean):
            print("  4 markers geometrically inconsistent — trying outlier drop...")
            best, best_score = None, float("inf")
            for drop in list(clean.keys()):
                candidate = {k: v for k, v in clean.items() if k != drop}
                mn, mp = _estimate_missing(candidate)
                test = dict(candidate)
                test[mn] = mp
                if _is_plausible_quad(test):
                    score = _geometry_error_score(test)
                    if score < best_score:
                        best_score = score
                        best = test
            if best is not None:
                clean = best
                print(f"  Resolved outlier via min-geometry-error (score={best_score:.4f})")

        n = len(clean)

        # Step 3: estimate missing corners from markers
        estimated_names: set = set()
        all_corners_img: Dict[str, Tuple[float, float]] = {}

        if n >= 4:
            all_corners_img = dict(clean)
        elif n == 3:
            mn, mp = _estimate_missing(clean)
            all_corners_img = dict(clean)
            all_corners_img[mn] = mp
            estimated_names.add(mn)
            if not _is_plausible_quad(all_corners_img):
                raise ValueError("3-marker reconstruction failed geometry validation.")
        elif n == 2:
            all_corners_img = CoordinateMapper.reconstruct_from_two_markers(clean)
            estimated_names = set(all_corners_img.keys()) - set(clean.keys())
            if not _is_plausible_quad(all_corners_img):
                raise ValueError("2-marker reconstruction failed geometry validation.")
        elif n == 1 and image_w > 0 and image_h > 0:
            print("  1 marker — estimating from document layout + image size")
            all_corners_img = CoordinateMapper.reconstruct_from_one_marker(
                clean, image_w, image_h
            )
            estimated_names = set(all_corners_img.keys()) - set(clean.keys())
            if not _is_plausible_quad(all_corners_img):
                raise ValueError("1-marker reconstruction failed geometry validation.")
        else:
            # Step 4: no usable markers — try image-based fallbacks
            if image_bgr is not None:
                print("  0 usable markers — trying blue boundary fallback...")
                result = CoordinateMapper.resolve_corners_from_blue_boundary(image_bgr)
                if result is not None:
                    print("  Blue boundary detected successfully")
                    return result

                print("  Blue boundary failed — trying contour fallback...")
                result = CoordinateMapper.resolve_corners_from_contour(image_bgr)
                if result is not None:
                    print("  Contour-based paper detection succeeded")
                    return result

            raise ValueError(
                f"Only {n} valid marker(s) and no image fallback available — cannot proceed."
            )

        return all_corners_img, estimated_names

    # ------------------------------------------------------------------
    # Document boundary from marker geometry
    # ------------------------------------------------------------------

    @staticmethod
    def compute_document_boundary_from_markers(
        all_corners_img: Dict[str, Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Compute page boundary by fitting doc->image homography from marker centers
        and projecting the true document rectangle.
        Returns [TL, TR, BR, BL] in image space.
        """
        required = {"top_left", "top_right", "bottom_right", "bottom_left"}
        if not required.issubset(all_corners_img.keys()):
            raise ValueError(f"Need all 4 corners, got {set(all_corners_img.keys())}")

        original_positions = CoordinateMapper.calculate_original_marker_positions()
        src_doc_marker_centers = np.array(
            [original_positions[name]["center"] for name in CORNER_ORDER],
            dtype=np.float32,
        )
        dst_img_marker_centers = np.array(
            [all_corners_img[name] for name in CORNER_ORDER],
            dtype=np.float32,
        )

        H_doc_to_img = cv2.getPerspectiveTransform(
            src_doc_marker_centers,
            dst_img_marker_centers,
        )

        W = float(MarkerConfig.DOC_WIDTH)
        H = float(MarkerConfig.DOC_HEIGHT)
        doc_page_corners = np.array(
            [[[0.0, 0.0], [W, 0.0], [W, H], [0.0, H]]],
            dtype=np.float32,
        )
        proj = cv2.perspectiveTransform(doc_page_corners, H_doc_to_img)[0]

        return [
            (float(proj[0][0]), float(proj[0][1])),
            (float(proj[1][0]), float(proj[1][1])),
            (float(proj[2][0]), float(proj[2][1])),
            (float(proj[3][0]), float(proj[3][1])),
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

        H_mat = cv2.getPerspectiveTransform(src, dst)

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