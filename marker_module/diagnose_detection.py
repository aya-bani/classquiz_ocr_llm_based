"""Diagnostic script to visualize preprocessing and detection attempts."""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marker_module.marker_scanner import ExamScanner
from marker_module.marker_config import MarkerConfig


def save_debug_image(data, name):
    """Save a single-channel or RGB image for inspection."""
    if len(data.shape) == 2:
        # Grayscale
        img = Image.fromarray(data).convert('L')
    else:
        # RGB
        img = Image.fromarray(data)
    img.save(f'Exams/real_exams/debug_{name}.png')
    print(f'Saved: debug_{name}.png')


def diagnose(img_path):
    print(f'Diagnosing {img_path}...')
    
    # Load and convert
    pil = Image.open(img_path).convert('RGB')
    img_cv = ExamScanner._pil_to_opencv(pil)
    
    # Step 1: Preprocess
    preprocessed = ExamScanner._preprocess_image(img_cv)
    print(f'Original shape: {img_cv.shape}, Preprocessed shape: {preprocessed.shape}')
    print(f'Original range: [{img_cv.min()}, {img_cv.max()}], Preprocessed range: [{preprocessed.min()}, {preprocessed.max()}]')
    
    save_debug_image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), '01_original')
    save_debug_image(preprocessed, '02_preprocessed')
    
    # Step 2: Try thresholding to see marker regions
    _, binary = cv2.threshold(preprocessed, 127, 255, cv2.THRESH_BINARY)
    save_debug_image(binary, '03_binary_threshold')
    
    # Step 3: Apply morphological ops to enhance
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    save_debug_image(binary_cleaned, '04_binary_morphed')
    
    # Step 4: Try edge detection
    edges = cv2.Canny(preprocessed, 30, 100)
    save_debug_image(edges, '05_canny_edges')
    
    # Step 5: Try Hough line detection to find document boundaries
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
    print(f'Found {len(lines) if lines is not None else 0} lines')
    
    # Visual: draw lines on original
    vis = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).copy()
    if lines is not None:
        for line in lines[:50]:  # limit to first 50
            x1, y1, x2, y2 = line[0]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    save_debug_image(vis, '06_detected_lines')
    
    # Step 6: Manual ArUco detection with explicit params
    aruco_dict = cv2.aruco.getPredefinedDictionary(MarkerConfig.DICT_TYPE)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshConstant = 5
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rejected = detector.detectMarkers(preprocessed)
    
    print(f'Detected markers: {len(ids) if ids is not None else 0}')
    print(f'Rejected candidates: {len(rejected) if rejected else 0}')
    
    # Visualize rejected markers
    vis_rejected = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB).copy()
    if rejected:
        cv2.aruco.drawDetectedMarkers(vis_rejected, rejected, borderColor=(255, 0, 0))
    save_debug_image(vis_rejected, '07_rejected_markers')
    
    # Visualize detected
    vis_detected = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB).copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis_detected, corners, ids)
    save_debug_image(vis_detected, '08_detected_markers')


if __name__ == '__main__':
    diagnose('Exams/real_exams/IMG_6168.jpg')
    print('\nDebug images saved to Exams/real_exams/debug_*.png')
