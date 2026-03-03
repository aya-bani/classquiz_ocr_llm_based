"""
test_mapper.py
--------------
Integration test for the CoordinateMapper pipeline.

Reads an exam scan from disk, detects ArUco markers, computes a homography
(with automatic fallback for 2/3 markers), dewarps the document, and writes
visualisation outputs to Exams/output_mapper/.

Usage
-----
    python marker_module/test_mapper/test_mapper.py [image_path]

If no path is supplied the first image found in Exams/new_real_exams/ is used.
"""

import sys
import re
from pathlib import Path

# Ensure package imports work when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image

from marker_module.marker_scanner import ExamScanner
from marker_module.coordinate_mapper import CoordinateMapper
from marker_module.marker_config import MarkerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_stem_from_input(input_path: Path) -> str:
    """Build normalised output stem as ex<number> from input file name."""
    match = re.search(r"(\d+)", input_path.stem)
    if match:
        return f"ex{match.group(1)}"
    return "ex0"


def _resolve_input_image(project_root: Path) -> Path:
    """Return the image to process: from CLI arg or first file in base dir."""
    base_dir = project_root / "Exams" / "new_real_exams"
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        return candidate
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(
        [p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    )
    if not images:
        return base_dir / "ex14.jpg"
    return images[0]


def _detection_mode_label(n: int) -> str:
    """Human-readable description of the active homography path."""
    if n >= 4:
        return "FULL (4 markers) -- standard homography"
    if n == 3:
        return "PARTIAL (3 markers) -- missing corner estimated via parallelogram rule"
    if n == 2:
        return "PARTIAL (2 markers) -- corners reconstructed via similarity transform"
    return f"INSUFFICIENT ({n} markers) -- cannot compute homography"


def _draw_corner_label(
    vis: np.ndarray,
    center: tuple,
    name: str,
    color: tuple,
    estimated: bool = False,
) -> None:
    """Draw a filled circle + text label at a corner position."""
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(vis, (cx, cy), 8, color, -1)
    label = f"{name} [est]" if estimated else name
    cv2.putText(
        vis, label, (cx + 10, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    input_path  = _resolve_input_image(project_root)
    output_dir  = project_root / "Exams" / "output_mapper"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem        = input_path.stem
    output_stem = _output_stem_from_input(input_path)

    print("=" * 72)
    print("COORDINATE MAPPER TEST")
    print("=" * 72)
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")

    # ------------------------------------------------------------------ #
    # 1. Load image & detect ArUco markers                               #
    # ------------------------------------------------------------------ #
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"ERROR: cannot read image: {input_path}")
        return 1

    preprocessed = ExamScanner._preprocess_image(image)
    corners, ids = ExamScanner._detect_markers_with_fallback(image, preprocessed)

    if ids is None or len(ids) == 0:
        print("ERROR: no markers detected")
        return 1

    detected_markers, corners_list, exam_ids, page_numbers = (
        ExamScanner._process_markers_with_corners(ids, corners)
    )

    n_detected = len(detected_markers)
    print(f"\nDetected markers : {n_detected}")
    print(f"Mode             : {_detection_mode_label(n_detected)}")
    print(f"Exam IDs         : {sorted(exam_ids) if exam_ids else 'N/A'}")
    print(f"Page numbers     : {sorted(page_numbers) if page_numbers else 'N/A'}")

    if n_detected < 2:
        print("ERROR: fewer than 2 markers detected -- cannot estimate homography")
        return 1

    # ------------------------------------------------------------------ #
    # 2. Compute homography (2 / 3 / 4 marker paths)                     #
    # ------------------------------------------------------------------ #
    homography = CoordinateMapper.compute_homography(detected_markers, corners_list)
    if homography is None:
        print("ERROR: failed to compute homography")
        return 1

    homography_path = output_dir / f"homography_{stem}.npy"
    np.save(homography_path, homography)

    # ------------------------------------------------------------------ #
    # 3. Collect image-space marker centers for reporting & visualisation #
    # ------------------------------------------------------------------ #
    detected_img: dict[str, tuple] = {}
    for i, marker_info in enumerate(detected_markers):
        name   = marker_info.get("corner", "unknown")
        center = CoordinateMapper.calculate_marker_center(corners_list[i][0])
        detected_img[name] = center

    # Debug: show what corners were actually detected
    valid_corners = {k: v for k, v in detected_img.items() if k in {"top_left", "top_right", "bottom_left", "bottom_right"}}
    print(f"\nValid corners detected: {list(valid_corners.keys())}")
    if len(valid_corners) != len(detected_img):
        print(f"WARNING: {len(detected_img) - len(valid_corners)} marker(s) with invalid corner name detected")

    all_corners_img: dict[str, tuple] = {}
    estimated_names: set[str]         = set()

    # Use only valid corners for geometry estimation
    n_valid = len(valid_corners)

    if n_valid >= 4:
        all_corners_img = dict(valid_corners)

    elif n_valid == 3:
        missing_name, missing_pt = CoordinateMapper.estimate_missing_corner_3_markers(
            valid_corners
        )
        all_corners_img = dict(valid_corners)
        all_corners_img[missing_name] = missing_pt
        estimated_names.add(missing_name)
        print(
            f"\nEstimated corner : {missing_name} @ "
            f"({missing_pt[0]:.2f}, {missing_pt[1]:.2f})"
        )

    elif n_valid == 2:
        all_corners_img = CoordinateMapper.reconstruct_from_two_markers(valid_corners)
        estimated_names = set(all_corners_img.keys()) - set(valid_corners.keys())
        print("\nReconstructed corners:")
        for name in estimated_names:
            pt = all_corners_img[name]
            print(f"  {name} @ ({pt[0]:.2f}, {pt[1]:.2f})")

    else:
        print(f"ERROR: only {n_valid} valid corner(s) detected -- cannot estimate homography")
        return 1

    # ------------------------------------------------------------------ #
    # 4. Dewarping                                                        #
    # ------------------------------------------------------------------ #
    scan_result = {
        "success":          True,
        "markers_found":    n_detected,
        "detected_markers": detected_markers,
        "corners":          corners_list,
    }

    pil_image = Image.open(input_path).convert("RGB")
    dewarped  = CoordinateMapper.extract_full_document(pil_image, scan_result)
    if dewarped is None:
        print("ERROR: dewarping failed")
        return 1

    dewarped_path = output_dir / f"{output_stem}_dewarped.jpg"
    dewarped.save(dewarped_path, quality=95)

    # ------------------------------------------------------------------ #
    # 5. Visualisation                                                    #
    # ------------------------------------------------------------------ #
    vis = image.copy()

    COLOR_DETECTED  = (0, 220, 0)    # green  -- real marker
    COLOR_ESTIMATED = (0, 165, 255)  # orange -- estimated / reconstructed
    COLOR_OUTLINE   = (255, 0, 0)    # blue   -- document boundary

    # Draw all four corner labels
    for name, center in all_corners_img.items():
        is_est = name in estimated_names
        _draw_corner_label(
            vis, center, name,
            COLOR_ESTIMATED if is_est else COLOR_DETECTED,
            estimated=is_est,
        )

    # ------------------------------------------------------------------
    # Draw document boundary derived directly from marker geometry.
    #
    # We use compute_document_boundary_from_markers() instead of mapping
    # abstract (0,0) corners through the homography because:
    #   * The homography is calibrated at marker *centers*, not page edges.
    #   * Mapping (0,0) requires extrapolation outside the calibration region.
    #   * Under 2/3-marker reconstruction the extrapolation error grows.
    #
    # The method steps each marker center outward by the known physical
    # offset (MARGIN + MARKER_SIZE/2 = 54 doc-px, scaled to image-px via
    # measured inter-marker distances) along locally inferred axis directions.
    # No homography matrix is used -- geometrically anchored to the markers.
    # ------------------------------------------------------------------
    inside_crop_path = None
    try:
        boundary_pts = CoordinateMapper.compute_document_boundary_from_markers(
            all_corners_img
        )
        pts = np.array(boundary_pts, dtype=np.int32)

        # Save a crop of only the content inside the boundary
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        inside_only = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h  = cv2.boundingRect(pts)
        inside_crop  = inside_only[y:y + h, x:x + w]
        inside_crop_path = output_dir / f"{output_stem}_inside_boundary.jpg"
        cv2.imwrite(str(inside_crop_path), inside_crop)

        # Draw the boundary outline
        cv2.polylines(vis, [pts], isClosed=True, color=COLOR_OUTLINE, thickness=3)

    except ValueError as exc:
        print(f"WARNING: boundary computation failed: {exc}")

    # Legend
    legend_y = 30
    for label, color in [
        ("Detected marker",           COLOR_DETECTED),
        ("Estimated / reconstructed", COLOR_ESTIMATED),
        ("Document boundary",         COLOR_OUTLINE),
    ]:
        cv2.circle(vis, (20, legend_y), 7, color, -1)
        cv2.putText(vis, label, (35, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        legend_y += 25

    vis_path = output_dir / f"{output_stem}_mapper_visualisation.jpg"
    cv2.imwrite(str(vis_path), vis)

    # ------------------------------------------------------------------ #
    # 6. Summary report                                                   #
    # ------------------------------------------------------------------ #
    print("\nMarker centers (image coords):")
    for name, (x, y) in all_corners_img.items():
        tag = " [estimated]" if name in estimated_names else ""
        print(f"  {name:<16}: ({x:.2f}, {y:.2f}){tag}")

    scale = CoordinateMapper.get_scale_factors(homography)
    if scale:
        print("\nScale factors:")
        print(f"  scale_x       : {scale['scale_x']:.4f}")
        print(f"  scale_y       : {scale['scale_y']:.4f}")
        print(f"  average_scale : {scale['average_scale']:.4f}")

    print("\nSaved outputs:")
    print(f"  {homography_path}")
    print(f"  {dewarped_path}")
    print(f"  {vis_path}")
    if inside_crop_path is not None:
        print(f"  {inside_crop_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())