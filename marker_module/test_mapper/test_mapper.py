import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from marker_module.marker_scanner import ExamScanner
from marker_module.coordinate_mapper import CoordinateMapper
from marker_module.marker_config import MarkerConfig


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
    if not images:
        return base_dir / "ex10.jpg"
    return images[0]


def _detection_mode_label(n: int) -> str:
    """Human-readable label for how many markers were detected."""
    if n >= 4:
        return "FULL (4 markers) — standard homography"
    if n == 3:
        return "PARTIAL (3 markers) — missing corner estimated via parallelogram rule"
    if n == 2:
        return "PARTIAL (2 markers) — corners reconstructed via similarity transform"
    return f"INSUFFICIENT ({n} markers) — cannot compute homography"


def _draw_corner_label(
    vis: np.ndarray,
    center: tuple,
    name: str,
    color: tuple,
    estimated: bool = False,
) -> None:
    """Draw a filled circle + text label at a corner position."""
    cx, cy = int(center[0]), int(center[1])
    radius = 8
    cv2.circle(vis, (cx, cy), radius, color, -1)

    label = name if not estimated else f"{name} [est]"
    cv2.putText(
        vis,
        label,
        (cx + 10, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    input_path = _resolve_input_image(project_root)
    output_dir = project_root / "Exams" / "output_mapper"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print("=" * 72)
    print("COORDINATE MAPPER TEST")
    print("=" * 72)
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")

    # ------------------------------------------------------------------ #
    # 1. Load & detect markers                                            #
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
        print("ERROR: fewer than 2 markers detected — cannot estimate homography")
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
    # 3. Collect detected image-space centers for reporting               #
    # ------------------------------------------------------------------ #
    detected_img: dict[str, tuple] = {}
    for i, marker_info in enumerate(detected_markers):
        name = marker_info.get("corner", "unknown")
        center = CoordinateMapper.calculate_marker_center(corners_list[i][0])
        detected_img[name] = center

    # Determine estimated corners for visualisation
    all_corners_img: dict[str, tuple] = {}
    estimated_names: set[str] = set()

    if n_detected >= 4:
        all_corners_img = detected_img
    elif n_detected == 3:
        missing_name, missing_pt = CoordinateMapper.estimate_missing_corner_3_markers(
            detected_img
        )
        all_corners_img = dict(detected_img)
        all_corners_img[missing_name] = missing_pt
        estimated_names.add(missing_name)
        print(f"\nEstimated corner : {missing_name} @ "
              f"({missing_pt[0]:.2f}, {missing_pt[1]:.2f})")
    else:  # 2 markers
        all_corners_img = CoordinateMapper.reconstruct_from_two_markers(detected_img)
        estimated_names = set(all_corners_img.keys()) - set(detected_img.keys())
        print("\nReconstructed corners:")
        for name in estimated_names:
            pt = all_corners_img[name]
            print(f"  {name} @ ({pt[0]:.2f}, {pt[1]:.2f})")

    # ------------------------------------------------------------------ #
    # 4. Dewarping                                                        #
    # ------------------------------------------------------------------ #
    scan_result = {
        "success": True,
        "markers_found": n_detected,
        "detected_markers": detected_markers,
        "corners": corners_list,
    }

    pil_image = Image.open(input_path).convert("RGB")
    dewarped = CoordinateMapper.extract_full_document(pil_image, scan_result)
    if dewarped is None:
        print("ERROR: dewarping failed")
        return 1

    dewarped_path = output_dir / f"{stem}_dewarped.jpg"
    dewarped.save(dewarped_path, quality=95)

    # ------------------------------------------------------------------ #
    # 5. Visualisation                                                    #
    # ------------------------------------------------------------------ #
    vis = image.copy()

    COLOR_DETECTED  = (0, 220, 0)    # green  — real marker
    COLOR_ESTIMATED = (0, 165, 255)  # orange — estimated / reconstructed
    COLOR_OUTLINE   = (255, 0, 0)    # blue   — document boundary

    # Draw all four corners
    for name, center in all_corners_img.items():
        is_estimated = name in estimated_names
        color = COLOR_ESTIMATED if is_estimated else COLOR_DETECTED
        _draw_corner_label(vis, center, name, color, estimated=is_estimated)

    # Draw the projected document boundary
    doc_corners_doc = [
        (0.0, 0.0),
        (float(MarkerConfig.DOC_WIDTH), 0.0),
        (float(MarkerConfig.DOC_WIDTH), float(MarkerConfig.DOC_HEIGHT)),
        (0.0, float(MarkerConfig.DOC_HEIGHT)),
    ]
    mapped_corners = CoordinateMapper.map_points_to_image(doc_corners_doc, homography)
    if mapped_corners:
        pts = np.array(mapped_corners, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=COLOR_OUTLINE, thickness=3)

    # Legend
    legend_y = 30
    for label, color in [
        ("Detected marker", COLOR_DETECTED),
        ("Estimated / reconstructed", COLOR_ESTIMATED),
        ("Document boundary", COLOR_OUTLINE),
    ]:
        cv2.circle(vis, (20, legend_y), 7, color, -1)
        cv2.putText(vis, label, (35, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        legend_y += 25

    vis_path = output_dir / f"{stem}_mapper_visualization.jpg"
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
        print(f"\nScale factors:")
        print(f"  scale_x       : {scale['scale_x']:.4f}")
        print(f"  scale_y       : {scale['scale_y']:.4f}")
        print(f"  average_scale : {scale['average_scale']:.4f}")

    print("\nSaved outputs:")
    print(f"  {homography_path}")
    print(f"  {dewarped_path}")
    print(f"  {vis_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())