"""
test_mapper.py — three-stage pipeline with quadrant-aware marker validation.
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


def _stem(p: Path) -> str:
    m = re.search(r"(\d+)", p.stem)
    return f"ex{m.group(1)}" if m else "ex0"


def _resolve_input(project_root: Path) -> Path:
    base = project_root / "Exams" / "examen_corrige"
    if len(sys.argv) > 1:
        c = Path(sys.argv[1])
        return c if c.is_absolute() else base / c
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in allowed])
    return imgs[0] if imgs else base / "ex14.jpg"


def _mode(n: int) -> str:
    if n >= 4: return "4 markers (full)"
    if n == 3: return "3 markers (1 estimated)"
    if n == 2: return "2 markers (2 reconstructed)"
    if n == 1: return "1 marker (3 reconstructed)"
    return "0 markers (fallback detection)"


def draw_corner(vis, center, name, color, estimated=False):
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(vis, (cx, cy), 10, color, -1)
    cv2.circle(vis, (cx, cy), 10, (0, 0, 0), 2)
    lbl = f"{name} [est]" if estimated else name
    cv2.putText(vis, lbl, (cx+13, cy-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def build_vis(image, all_corners, estimated_names, boundary_pts):
    vis = image.copy()
    C_OK  = (0, 220, 0)
    C_EST = (0, 165, 255)
    C_BND = (255, 50, 50)
    for name, center in all_corners.items():
        draw_corner(vis, center, name,
                    C_EST if name in estimated_names else C_OK,
                    estimated=(name in estimated_names))
    if boundary_pts is not None:
        cv2.polylines(vis, [boundary_pts], True, C_BND, 3)
    y = 35
    for lbl, col in [("Detected marker", C_OK),
                     ("Estimated / reconstructed", C_EST),
                     ("Document boundary", C_BND)]:
        cv2.circle(vis, (22, y), 8, col, -1)
        cv2.putText(vis, lbl, (38, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        y += 30
    return vis


def build_crop(image, boundary_pts):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [boundary_pts], 255)
    inside = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(boundary_pts)
    x = max(0, x); y = max(0, y)
    return inside[y:min(image.shape[0], y+h), x:min(image.shape[1], x+w)]


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    input_path = _resolve_input(project_root)
    output_dir = project_root / "Exams" / "output_mapper"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(input_path)

    print("=" * 72)
    print(f"COORDINATE MAPPER  |  {input_path.name}")
    print("=" * 72)

    # ---- Load ---------------------------------------------------------
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"ERROR: cannot read {input_path}"); return 1
    img_h, img_w = image.shape[:2]

    # ---- Detect -------------------------------------------------------
    preprocessed = ExamScanner._preprocess_image(image)
    corners, ids = ExamScanner._detect_markers_with_fallback(image, preprocessed)

    detected_markers, corners_list, exam_ids, page_numbers = [], [], set(), set()
    if ids is not None and len(ids) > 0:
        detected_markers, corners_list, exam_ids, page_numbers = (
            ExamScanner._process_markers_with_corners(ids, corners)
        )
    print(f"Raw detections   : {len(detected_markers)} markers")
    print(f"Exam IDs         : {sorted(exam_ids) if exam_ids else 'N/A'}")
    print(f"Page numbers     : {sorted(page_numbers) if page_numbers else 'N/A'}")

    # ---- Resolve & validate corners (with quadrant filter) ------------
    try:
        all_corners_img, estimated_names = CoordinateMapper.resolve_corners(
            detected_markers, corners_list,
            image_w=img_w, image_h=img_h,
            image_bgr=image,   # <-- enables fallback detection
        )
    except ValueError as e:
        print(f"ERROR: {e}"); return 1

    n_real = len(all_corners_img) - len(estimated_names)
    print(f"Valid markers    : {n_real}  ({_mode(n_real)})")
    if estimated_names:
        print(f"Estimated        : {sorted(estimated_names)}")

    print("\nCorner positions (image coords):")
    for name in ["top_left", "top_right", "bottom_right", "bottom_left"]:
        if name in all_corners_img:
            x, y = all_corners_img[name]
            tag = " [est]" if name in estimated_names else ""
            print(f"  {name:<16}: ({x:.1f}, {y:.1f}){tag}")

    # ---- Boundary -----------------------------------------------------
    boundary_pts = None
    try:
        bpts = CoordinateMapper.compute_document_boundary_from_markers(all_corners_img)
        boundary_pts = np.array(bpts, dtype=np.int32)
    except ValueError as e:
        print(f"WARNING: boundary failed: {e}")

    # ---- Stage 1: visualisation ---------------------------------------
    vis = build_vis(image, all_corners_img, estimated_names, boundary_pts)
    p1 = output_dir / f"{stem}_1_mapper_visualisation.jpg"
    cv2.imwrite(str(p1), vis)
    print(f"\nStage 1 -> {p1.name}")

    # ---- Stage 2: raw paper crop --------------------------------------
    if boundary_pts is not None:
        crop = build_crop(image, boundary_pts)
        p2 = output_dir / f"{stem}_2_inside_boundary.jpg"
        cv2.imwrite(str(p2), crop)
        print(f"Stage 2 -> {p2.name}")
    else:
        print("Stage 2 skipped")

    # ---- Stage 3: dewarped --------------------------------------------
    warped = CoordinateMapper.dewarp_document(image, all_corners_img)
    if warped is None:
        print("Stage 3 FAILED"); return 1
    dewarped = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    p3 = output_dir / f"{stem}_3_dewarped.jpg"
    dewarped.save(p3, quality=95)
    print(f"Stage 3 -> {p3.name}  ({dewarped.width}x{dewarped.height})")

    # ---- Diagnostics --------------------------------------------------
    H = CoordinateMapper.compute_homography(
        detected_markers, corners_list, image_w=img_w, image_h=img_h
    )
    if H is not None:
        np.save(output_dir / f"homography_{input_path.stem}.npy", H)
        sc = CoordinateMapper.get_scale_factors(H)
        if sc:
            print(f"\nScale: x={sc['scale_x']:.4f}  y={sc['scale_y']:.4f}  "
                  f"avg={sc['average_scale']:.4f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())