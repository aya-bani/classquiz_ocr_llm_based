import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, '..')
from marker_module.marker_generator import MarkerGenerator
from marker_module.marker_config import MarkerConfig
from logger_manager import LoggerManager


def test_marker_generator():
    """Test MarkerGenerator functionality"""
    logger = LoggerManager.get_logger(__name__)
    logger.info("Starting MarkerGenerator tests")

    # Initialize
    generator = MarkerGenerator()
    logger.info("✓ MarkerGenerator initialized")

    # Test 1: Calculate markers for a page
    exam_id = 5
    page_number = 0
    marker_ids = generator.calculate_markers(exam_id, page_number)
    logger.info(f"✓ Marker IDs calculated for exam {exam_id}, page {page_number}: {marker_ids}")
    assert len(marker_ids) == 4, "Should have 4 markers per page"
    assert marker_ids[:3] == MarkerConfig.FIXED_MARKER_IDS, "First 3 should be fixed markers"

    # Test 2: Generate single marker
    marker_image = generator.generate_marker(marker_ids[0])
    logger.info(f"✓ Single marker generated: shape={marker_image.shape}")
    assert marker_image.shape == (MarkerConfig.MARKER_SIZE, MarkerConfig.MARKER_SIZE)

    # Test 3: Create a test image
    test_width, test_height = 800, 600
    test_image = Image.new('RGB', (test_width, test_height), color='white')
    logger.info(f"✓ Test image created: {test_width}x{test_height}")

    # Test 4: Add markers to page
    marked_image = generator.add_markers_to_page(exam_id, test_image, page_number)
    logger.info(f"✓ Markers added to page: output shape={marked_image.shape}")
    assert marked_image.shape[:2] == (test_height, test_width)
    assert len(marked_image.shape) == 3, "Output should be BGR image"

    # Test 5: Get marker range for exam
    first, last = generator.get_exam_marker_range(exam_id)
    logger.info(f"✓ Marker range for exam {exam_id}: {first}-{last}")

    # Test 6: Generate marked pages (simulate PDF pages)
    pages = [
        Image.new('RGB', (test_width, test_height), color='white'),
        Image.new('RGB', (test_width, test_height), color='lightgray'),
        Image.new('RGB', (test_width, test_height), color='white'),
    ]
    marked_pages = generator.generate_marked_exam(exam_id, pages)
    logger.info(f"✓ Generated marked pages for exam {exam_id}: {len(marked_pages)} pages")
    assert len(marked_pages) == len(pages), "Should have same number of output pages"

    for idx, page in enumerate(marked_pages):
        assert isinstance(page, Image.Image), f"Page {idx} should be PIL Image"
        logger.info(f"  - Page {idx}: {page.size}, mode={page.mode}")

    logger.info("\n" + "="*60)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("="*60)


if __name__ == "__main__":
    test_marker_generator()
