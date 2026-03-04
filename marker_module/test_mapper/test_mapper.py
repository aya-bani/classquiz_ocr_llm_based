"""
test_mapper.py
--------------
Integration test for the CoordinateMapper pipeline.

Three output images — in pipeline order:

    1_mapper_visualisation.jpg
        Original photo annotated with detected/estimated marker corners
        and the computed document boundary quad (blue).

    2_inside_boundary.jpg
        Raw paper content cropped from the original photo using the boundary.
        Still perspective-distorted. Shows what the boundary captures.

    3_dewarped.jpg
        Perspective-corrected document, DOC_WIDTH x DOC_HEIGHT, portrait.
        Built directly from the boundary corners — no homography inversion.
"""

import sys
import re
from pathlib import Path

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
    match = re.search(r"(\d+)", input_path.stem)
    return f"ex{match.group(1)}" if match else "ex0"


def _resolve_input_image(project_root: Path) -> Path:
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
    return images[0] if images else base_dir / "ex14.jpg"


def _detection_mode_label(n: int) -> str:
    if n >= 4: return "FULL (4 markers) -- standard homography"
    if n == 3: return "PARTIAL (3 markers) -- missing corner via parallelogram rule"
    if n == 2: return "PARTIAL (2 markers) -- corners via similarity transform"
    return f"INSUFFICIENT ({n} markers)"


def _draw_corner_label(vis, center, name, color, estimated=False):
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(vis, (cx, cy), 8, color, -1)
    cv2.putText(vis, f"{name} [est]" if estimated else name,
                (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------

def stage1_visualisation(image, all_corners_img, estimated_names, boundary_pts):
    """Annotate the original photo: marker dots + boundary quad."""
    vis = image.copy()
    COLOR_OK  = (0, 220, 0)
    COLOR_EST = (0, 165, 255)
    COLOR_BND = (255, 0, 0)

    for name, center in all_corners_img.items():
        _draw_corner_label(vis, center, name,
                           COLOR_EST if name in estimated_names else COLOR_OK,
                           estimated=(name in estimated_names))

    if boundary_pts is not None:
        cv2.polylines(vis, [boundary_pts], isClosed=True, color=COLOR_BND, thickness=3)

    # Legend
    y = 30
    for label, col in [("Detected marker", COLOR_OK),
                        ("Estimated / reconstructed", COLOR_EST),
                        ("Document boundary", COLOR_BND)]:
        cv2.circle(vis, (20, y), 7, col, -1)
        cv2.putText(vis, label, (35, y+5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
        y += 25
    return vis


def stage2_inside_boundary(image, boundary_pts):
    """
    Mask the original photo to the boundary polygon, then crop to its
    bounding rectangle.  Source is the clean original (no annotations).
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [boundary_pts], 255)
    inside = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(boundary_pts)
    x = max(0, x); y = max(0, y)
    return inside[y:min(image.shape[0], y+h), x:min(image.shape[1], x+w)]


def stage3_dewarped(image_bgr, all_corners_img):
    """
    Perspective-correct the document.

    Builds H directly as:
        src = boundary corners in IMAGE space
        dst = output canvas corners (0,0 -> DOC_W, DOC_H)
    No matrix inversion. Orientation is corrected automatically.
    """
    warped = CoordinateMapper.dewarp_document(image_bgr, all_corners_img)
    if warped is None:
        return None
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


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
    print()
    print("Pipeline:")
    print("  Stage 1 -> 1_mapper_visualisation  (annotated original photo)")
    print("  Stage 2 -> 2_inside_boundary       (raw paper crop, pre-correction)")
    print("  Stage 3 -> 3_dewarped              (perspective-corrected, portrait)")

    # ------------------------------------------------------------------ #
    # Detect markers                                                      #
    # ------------------------------------------------------------------ #
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"\nERROR: cannot read {input_path}")
        return 1

    preprocessed = ExamScanner._preprocess_image(image)
    corners, ids = ExamScanner._detect_markers_with_fallback(image, preprocessed)

    if ids is None or len(ids) == 0:
        print("\nERROR: no markers detected")
        return 1

    detected_markers, corners_list, exam_ids, page_numbers = (
        ExamScanner._process_markers_with_corners(ids, corners)
    )

    n_detected = len(detected_markers)
    print(f"\nDetected markers : {n_detected}")
    print(f"Mode             : {_detection_mode_label(n_detected)}")
    print(f"Exam IDs         : {sorted(exam_ids) if exam_ids else 'N/A'}")
    print(f"Page numbers     : {sorted(page_numbers) if page_numbers else 'N/A'}")

    # ------------------------------------------------------------------ #
    # Resolve all 4 corner positions in image space                       #
    # ------------------------------------------------------------------ #
    detected_img: dict = {}
    for i, mi in enumerate(detected_markers):
        cn = mi.get("corner", "")
        if cn not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
            continue
        detected_img[cn] = CoordinateMapper.calculate_marker_center(corners_list[i][0])

    n_valid = len(detected_img)
    print(f"Valid corners    : {sorted(detected_img.keys())}")

    if n_valid < 2:
        print(f"\nERROR: only {n_valid} valid corner(s) -- cannot continue")
        return 1

    all_corners_img: dict = {}
    estimated_names: set  = set()

    if n_valid >= 4:
        all_corners_img = dict(detected_img)
    elif n_valid == 3:
        mn, mp = CoordinateMapper.estimate_missing_corner_3_markers(detected_img)
        all_corners_img = dict(detected_img)
        all_corners_img[mn] = mp
        estimated_names.add(mn)
        print(f"Estimated        : {mn} @ ({mp[0]:.1f}, {mp[1]:.1f})")
    else:
        all_corners_img = CoordinateMapper.reconstruct_from_two_markers(detected_img)
        estimated_names = set(all_corners_img.keys()) - set(detected_img.keys())
        for nm in estimated_names:
            pt = all_corners_img[nm]
            print(f"Reconstructed    : {nm} @ ({pt[0]:.1f}, {pt[1]:.1f})")

    # ------------------------------------------------------------------ #
    # Compute document boundary                                           #
    # ------------------------------------------------------------------ #
    boundary_pts = None
    try:
        bpts = CoordinateMapper.compute_document_boundary_from_markers(all_corners_img)
        boundary_pts = np.array(bpts, dtype=np.int32)
    except ValueError as e:
        print(f"\nWARNING: boundary failed: {e}")

    # ------------------------------------------------------------------ #
    # STAGE 1 -- annotated visualisation                                  #
    # ------------------------------------------------------------------ #
    vis = stage1_visualisation(image, all_corners_img, estimated_names, boundary_pts)
    vis_path = output_dir / f"{output_stem}_1_mapper_visualisation.jpg"
    cv2.imwrite(str(vis_path), vis)
    print(f"\nStage 1 saved : {vis_path.name}")

    # ------------------------------------------------------------------ #
    # STAGE 2 -- raw paper crop                                           #
    # ------------------------------------------------------------------ #
    if boundary_pts is not None:
        crop = stage2_inside_boundary(image, boundary_pts)
        crop_path = output_dir / f"{output_stem}_2_inside_boundary.jpg"
        cv2.imwrite(str(crop_path), crop)
        print(f"Stage 2 saved : {crop_path.name}")
    else:
        print("Stage 2 skipped: no boundary")

    # ------------------------------------------------------------------ #
    # STAGE 3 -- perspective correction                                   #
    # ------------------------------------------------------------------ #
    dewarped = stage3_dewarped(image, all_corners_img)
    if dewarped is None:
        print("Stage 3 FAILED: dewarp returned None")
        return 1
    dw_path = output_dir / f"{output_stem}_3_dewarped.jpg"
    dewarped.save(dw_path, quality=95)
    print(f"Stage 3 saved : {dw_path.name}  "
          f"({dewarped.width}x{dewarped.height}px)")

    # Also save homography for diagnostic use
    scan_result = {
        "success": True, "markers_found": n_detected,
        "detected_markers": detected_markers, "corners": corners_list,
    }
    H = CoordinateMapper.compute_homography(detected_markers, corners_list)
    if H is not None:
        np.save(output_dir / f"homography_{stem}.npy", H)
        scale = CoordinateMapper.get_scale_factors(H)
        if scale:
            print(f"\nScale  x={scale['scale_x']:.4f}  "
                  f"y={scale['scale_y']:.4f}  "
                  f"avg={scale['average_scale']:.4f}")

    print("\nMarker centres (image coords):")
    for name, (x, y) in all_corners_img.items():
        tag = " [est]" if name in estimated_names else ""
        print(f"  {name:<16}: ({x:.1f}, {y:.1f}){tag}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())