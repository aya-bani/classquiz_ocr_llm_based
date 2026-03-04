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

Marker validation
    Before computing homography, detected markers are validated to ensure
    they belong to the TARGET document (not markers from other pages in the
    stack). Validation checks:
      1. Geometric consistency: the 4 corners must form a convex quadrilateral
         with the correct TL/TR/BR/BL spatial relationships.
      2. Aspect ratio: the quad's shape must be plausible for a document of
         the known DOC_WIDTH x DOC_HEIGHT dimensions.
      3. Outlier rejection: any single marker that is far from its expected
         position (given the others) is removed and re-estimated.

Dewarping
    warpPerspective is called with H built as:
        src = 4 boundary corners in IMAGE space
        dst = output canvas corners (0,0 -> DOC_W, DOC_H)
    No matrix inversion. Orientation corrected automatically.
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
# Geometry validation helpers
# ===========================================================================

def _is_convex_quad(pts: Dict[str, Tuple[float, float]]) -> bool:
    """
    Return True if the 4 corners form a convex quadrilateral with the
    correct spatial ordering: TL upper-left, TR upper-right, etc.

    Checks:
      1. Convexity: all cross-products of consecutive edge vectors have
         the same sign (all clockwise or all counter-clockwise).
      2. Spatial sanity: TL.x < TR.x, BL.x < BR.x, TL.y < BL.y, TR.y < BR.y
         (allowing up to 40% tolerance for perspective tilt).
    """
    required = {"top_left", "top_right", "bottom_right", "bottom_left"}
    if not required.issubset(pts.keys()):
        return False

    tl = np.array(pts["top_left"],     dtype=np.float64)
    tr = np.array(pts["top_right"],    dtype=np.float64)
    br = np.array(pts["bottom_right"], dtype=np.float64)
    bl = np.array(pts["bottom_left"],  dtype=np.float64)

    # Convexity check (vertices in order TL, TR, BR, BL)
    poly = [tl, tr, br, bl]
    cross_signs = []
    n = len(poly)
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i+1) % n]
        p2 = poly[(i+2) % n]
        cross = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0])
        cross_signs.append(np.sign(cross))
    if len(set(cross_signs)) > 1:  # mixed signs -> not convex
        return False

    # Spatial sanity: each corner should be on the correct side
    # Allow up to 40% of the diagonal as tolerance for perspective tilt
    h_span = max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl))
    v_span = max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr))
    tol_h = 0.4 * h_span
    tol_v = 0.4 * v_span

    if not (tl[0] < tr[0] + tol_h):  return False  # TL left of TR
    if not (bl[0] < br[0] + tol_h):  return False  # BL left of BR
    if not (tl[1] < bl[1] + tol_v):  return False  # TL above BL
    if not (tr[1] < br[1] + tol_v):  return False  # TR above BR

    return True


def _aspect_ratio_ok(pts: Dict[str, Tuple[float, float]],
                     max_ratio_error: float = 0.6) -> bool:
    """
    Check that the quad's aspect ratio is plausible for DOC_WIDTH x DOC_HEIGHT.

    We allow a wide tolerance (0.6) because perspective distortion can
    significantly change the apparent ratio.
    """
    tl = np.array(pts["top_left"],     dtype=np.float64)
    tr = np.array(pts["top_right"],    dtype=np.float64)
    br = np.array(pts["bottom_right"], dtype=np.float64)
    bl = np.array(pts["bottom_left"],  dtype=np.float64)

    w = (np.linalg.norm(tr-tl) + np.linalg.norm(br-bl)) / 2.0
    h = (np.linalg.norm(bl-tl) + np.linalg.norm(br-tr)) / 2.0

    if w < 1e-3 or h < 1e-3:
        return False

    observed_ratio = w / h
    expected_ratio = MarkerConfig.DOC_WIDTH / MarkerConfig.DOC_HEIGHT  # ~0.707

    # In portrait: observed ~ expected. In landscape photo: observed ~ 1/expected.
    err1 = abs(observed_ratio - expected_ratio) / expected_ratio
    err2 = abs(observed_ratio - 1.0/expected_ratio) / (1.0/expected_ratio)

    return min(err1, err2) <= max_ratio_error


def validate_and_clean_markers(
    detected_img: Dict[str, Tuple[float, float]]
) -> Dict[str, Tuple[float, float]]:
    """
    Given detected marker centres keyed by corner name, remove any markers
    that are geometrically inconsistent with the others.

    Algorithm
    ---------
    1. If all 4 markers present and form a valid convex quad -> return as-is.
    2. If all 4 present but quad is invalid: try dropping each marker one at
       a time and check if the remaining 3 are consistent (can estimate the
       missing one to form a valid quad). Keep the best 3.
    3. If 3 markers present: check consistency directly.
    4. If 2 markers present: return as-is (no validation possible).

    Returns
    -------
    Cleaned dict (may have fewer markers than input if outliers were removed).
    """
    valid_names = {"top_left", "top_right", "bottom_right", "bottom_left"}
    clean = {k: v for k, v in detected_img.items() if k in valid_names}

    if len(clean) < 2:
        return clean

    if len(clean) == 2:
        return clean  # nothing to validate

    if len(clean) == 3:
        # Estimate missing corner and check convexity
        try:
            mn, mp = _estimate_missing(clean)
            test = dict(clean); test[mn] = mp
            if _is_convex_quad(test) and _aspect_ratio_ok(test):
                return clean  # all 3 are good
        except Exception:
            pass
        # Try dropping each of the 3 and see if the remaining 2 + estimated
        # give a better quad -- return the pair that works
        for drop in list(clean.keys()):
            remaining = {k: v for k, v in clean.items() if k != drop}
            try:
                mn, mp = _estimate_missing(remaining)
                test = dict(remaining); test[mn] = mp
                # Still need 4 to check
                mn2, mp2 = _estimate_missing(test)  # will error if already 4
            except Exception:
                pass
        return clean  # return as-is if we can't improve

    # len == 4
    if _is_convex_quad(clean) and _aspect_ratio_ok(clean):
        return clean  # perfect

    # 4 markers but invalid quad: try dropping each one
    best_clean = None
    best_score = float('inf')

    for drop in list(clean.keys()):
        candidate = {k: v for k, v in clean.items() if k != drop}
        try:
            mn, mp = _estimate_missing(candidate)
            test = dict(candidate); test[mn] = mp
            if _is_convex_quad(test) and _aspect_ratio_ok(test):
                # Score = distance the dropped marker is from its estimated position
                score = np.linalg.norm(
                    np.array(clean[drop]) - np.array(test[drop])
                )
                if score < best_score:
                    best_score = score
                    best_clean = candidate
        except Exception:
            continue

    if best_clean is not None:
        dropped = set(clean.keys()) - set(best_clean.keys())
        print(f"  [validate_markers] Dropped outlier marker: {dropped} "
              f"(error={best_score:.1f}px)")
        return best_clean

    # Could not fix by dropping one — return original and let caller handle
    return clean


def _estimate_missing(
    detected: Dict[str, Tuple[float, float]]
) -> Tuple[str, Tuple[float, float]]:
    """Internal helper: estimate missing corner from 3 detected ones."""
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
        return _estimate_missing(detected)

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

        Each marker centre is offset = MARGIN + MARKER_SIZE/2 from the
        nearest page edge. We step each marker outward along locally-measured
        axis directions by this offset (scaled to image-px).

        Returns [TL, TR, BR, BL] as float tuples.
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
    # Homography (doc-space -> image-space, for point mapping only)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_homography(
        detected_markers: List[Dict],
        corners_data: List[np.ndarray],
        vis_image: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute H such that H @ doc_point -> image_point.
        Used by map_point_to_image / map_points_to_image.
        NOT used for dewarping (see dewarp_document).
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

        # Validate and clean markers before using them
        detected_img = validate_and_clean_markers(detected_img)
        n_valid = len(detected_img)

        if n_valid < 2:
            return None

        if n_valid >= 4:
            all_img = detected_img
        elif n_valid == 3:
            mn, mp = _estimate_missing(detected_img)
            all_img = dict(detected_img); all_img[mn] = mp
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
    # Core: resolve + validate all 4 corner positions
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_corners(
        detected_markers: List[Dict],
        corners_data: List[np.ndarray],
    ) -> Tuple[Dict[str, Tuple[float, float]], set]:
        """
        Collect detected marker centres, validate them, and fill in missing
        corners by estimation.

        Returns
        -------
        all_corners_img : dict  -- all 4 corner names -> (x, y) image coords
        estimated_names : set   -- corners that were estimated (not measured)

        Raises ValueError if fewer than 2 valid markers remain after cleaning.
        """
        original_positions = CoordinateMapper.calculate_original_marker_positions()

        # Collect raw detections
        raw: Dict[str, Tuple[float, float]] = {}
        for i, mi in enumerate(detected_markers):
            cn = mi.get("corner")
            if cn not in original_positions:
                continue
            raw[cn] = CoordinateMapper.calculate_marker_center(corners_data[i][0])

        # Validate: remove geometrically inconsistent markers
        clean = validate_and_clean_markers(raw)
        n = len(clean)

        if n < 2:
            raise ValueError(f"Only {n} valid marker(s) after cleaning -- cannot proceed.")

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
    # Dewarping
    # ------------------------------------------------------------------

    @staticmethod
    def dewarp_document(
        image: np.ndarray,
        all_corners_img: Dict[str, Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """
        Perspective-correct the document.

        Builds H as:
            src = document-boundary corners in IMAGE space
            dst = output canvas corners (0,0 -> DOC_W, DOC_H)
        Calls warpPerspective directly -- no matrix inversion.
        Auto-detects and corrects landscape orientation.

        Returns BGR numpy array (DOC_HEIGHT x DOC_WIDTH) or None.
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
        paper_is_portrait = v_span >= h_span

        if paper_is_portrait:
            dst = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
            out_size = (int(W), int(H))
        else:
            # Paper on its side -> rotate output so portrait is upright
            dst = np.array([[0,0],[0,H],[W,H],[W,0]], dtype=np.float32)
            out_size = (int(H), int(W))

        H_mat, _ = cv2.findHomography(src, dst, 0)
        if H_mat is None:
            return None

        warped = cv2.warpPerspective(image, H_mat, out_size,
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))

        # Ensure portrait output
        if warped.shape[1] > warped.shape[0]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        # Resize to exact doc dimensions
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
        """
        Extract and perspective-correct the document (public API).

        Args:
            image:           PIL RGB image of the raw photo.
            scan_result:     Result dict from ExamScanner.scan_page().
            all_corners_img: Pre-computed corners (from resolve_corners).
                             If None, computed internally from scan_result.
            vis_image:       Unused (kept for API compatibility).
        """
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
    # Coordinate mapping (for overlay / OCR zone mapping)
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