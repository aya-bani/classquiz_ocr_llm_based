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
	images = sorted([p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed])
	if not images:
		return base_dir / "ex4.jpg"
	return images[0]


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

	image = cv2.imread(str(input_path))
	if image is None:
		print(f"ERROR: cannot read image: {input_path}")
		return 1

	preprocessed = ExamScanner._preprocess_image(image)
	corners, ids = ExamScanner._detect_markers_with_fallback(image, preprocessed)

	if ids is None or len(ids) == 0:
		print("ERROR: no markers detected")
		return 1

	detected_markers, corners_list, exam_ids, page_numbers = ExamScanner._process_markers_with_corners(ids, corners)

	print(f"Detected markers: {len(detected_markers)}")
	print(f"Exam IDs: {sorted(exam_ids) if exam_ids else 'N/A'}")
	print(f"Page numbers: {sorted(page_numbers) if page_numbers else 'N/A'}")

	if len(detected_markers) < 4:
		print("ERROR: fewer than 4 markers detected, cannot compute homography")
		return 1

	homography = CoordinateMapper.compute_homography(detected_markers, corners_list)
	if homography is None:
		print("ERROR: failed to compute homography")
		return 1

	homography_path = output_dir / f"homography_{stem}.npy"
	np.save(homography_path, homography)

	scan_result = {
		"success": True,
		"markers_found": len(detected_markers),
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

	vis = image.copy()
	marker_centers = []
	for i, marker_info in enumerate(detected_markers):
		center = CoordinateMapper.calculate_marker_center(corners_list[i][0])
		marker_centers.append((marker_info.get("corner", "unknown"), center))
		cx, cy = int(center[0]), int(center[1])
		cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)
		cv2.putText(
			vis,
			marker_info.get("corner", "?"),
			(cx + 10, cy - 8),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 255, 0),
			2,
			cv2.LINE_AA,
		)

	doc_corners = [
		(0.0, 0.0),
		(float(MarkerConfig.DOC_WIDTH), 0.0),
		(float(MarkerConfig.DOC_WIDTH), float(MarkerConfig.DOC_HEIGHT)),
		(0.0, float(MarkerConfig.DOC_HEIGHT)),
	]
	mapped_corners = CoordinateMapper.map_points_to_image(doc_corners, homography)
	if mapped_corners:
		pts = np.array(mapped_corners, dtype=np.int32)
		cv2.polylines(vis, [pts], isClosed=True, color=(255, 0, 0), thickness=3)

	vis_path = output_dir / f"{stem}_mapper_visualization.jpg"
	cv2.imwrite(str(vis_path), vis)

	print("\nSaved outputs:")
	print(f"- {homography_path}")
	print(f"- {dewarped_path}")
	print(f"- {vis_path}")
	print("\nMarker centers (image coords):")
	for name, (x, y) in marker_centers:
		print(f"- {name}: ({x:.2f}, {y:.2f})")

	print("\nDone.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
