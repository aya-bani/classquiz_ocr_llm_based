import pytest
import numpy as np
from PIL import Image

from marker_module.marker_generator import MarkerGenerator
from marker_module.marker_config import MarkerConfig


@pytest.fixture
def generator():
    """Returns a fresh MarkerGenerator for each test."""
    return MarkerGenerator()


def test_calculate_markers_basic(generator):
    # first three IDs must always equal the fixed constants
    ids = generator.calculate_markers(0, 0)
    assert ids[:3] == MarkerConfig.FIXED_MARKER_IDS
    # fourth ID should follow the old formula for exam 0 page 0
    assert ids[3] == 3

    # different exam still uses same fixed values
    ids2 = generator.calculate_markers(1, 0)
    assert ids2[:3] == MarkerConfig.FIXED_MARKER_IDS
    expected_fourth = 1 * MarkerConfig.BLOCK_SIZE + (
        MarkerConfig.CORNERS_PER_PAGE - 1)
    assert ids2[3] == expected_fourth
    assert len(ids2) == MarkerConfig.CORNERS_PER_PAGE

    # last valid page/exam combination should not raise an error
    max_exam = MarkerConfig.MAX_EXAMS - 1
    max_page = MarkerConfig.PAGES_PER_EXAM - 1
    _ = generator.calculate_markers(max_exam, max_page)


def test_calculate_markers_invalid():
    gen = MarkerGenerator()
    with pytest.raises(ValueError):
        gen.calculate_markers(-1, 0)
    with pytest.raises(ValueError):
        gen.calculate_markers(0, MarkerConfig.PAGES_PER_EXAM)
    with pytest.raises(ValueError):
        gen.calculate_markers(MarkerConfig.MAX_EXAMS, 0)


def test_generate_marker_bounds():
    gen = MarkerGenerator()
    with pytest.raises(ValueError):
        gen.generate_marker(-1)
    with pytest.raises(ValueError):
        gen.generate_marker(MarkerConfig.MAX_MARKER_ID + 1)


def test_generate_marker_image_shape(generator):
    img = generator.generate_marker(10)
    assert isinstance(img, np.ndarray)
    assert img.shape == (MarkerConfig.MARKER_SIZE, MarkerConfig.MARKER_SIZE)
    # should be a binary pattern (0 or 255)
    assert set(np.unique(img)).issubset({0, 255})


def test_add_markers_to_page_effect(generator):
    # create a plain white page
    blank = Image.new("RGB", (500, 500), "white")
    result = generator.add_markers_to_page(0, blank, 0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (500, 500, 3)

    # the top-left corner should now contain a marker (not all white)
    tl = result[0:MarkerConfig.MARKER_SIZE, 0:MarkerConfig.MARKER_SIZE]
    assert not np.all(tl == 255)


def test_get_exam_marker_range(generator):
    # range should still include the fixed identifiers and the dynamic block
    first, last = generator.get_exam_marker_range(0)
    assert first == min(MarkerConfig.FIXED_MARKER_IDS)
    assert last == MarkerConfig.BLOCK_SIZE - 1

    with pytest.raises(ValueError):
        generator.get_exam_marker_range(-1)


def test_generate_marked_exam_roundtrip(generator):
    # generate a few dummy pages (large enough for markers) and ensure each
    # returned page is an Image
    size = 500  # comfortably larger than MarkerConfig.MARKER_SIZE + 2*MARGIN
    pages = [Image.new("RGB", (size, size), "white") for _ in range(3)]
    marked = generator.generate_marked_exam(2, pages)
    assert len(marked) == 3
    for p in marked:
        assert isinstance(p, Image.Image)

    # markers should differ from plain white
    arr = np.array(marked[0])
    assert not np.all(arr == 255)


def test_scan_real_exam_image():
    """Load a real exam photo and run the scanner on it."""
    import os
    import cv2

    from marker_module.marker_scanner import ExamScanner

    img_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'Exams', 'real_exams', 'IMG_6168.jpg'
    )
    img_path = os.path.normpath(img_path)

    img = cv2.imread(img_path)
    assert img is not None, f"could not load image at {img_path}"

    result = ExamScanner.scan_page(img)
    # basic expectations
    assert isinstance(result, dict)
    assert 'success' in result
    # at least one marker must be found for a valid test
    assert result['markers_found'] > 0
    # scanner should report success if markers were present
    assert result['success'] is True
    # dynamic markers should exist for a real exam page
    assert result.get('dynamic_markers_found', 0) > 0
    # exam_id/page_number should be integers
    assert isinstance(result.get('exam_id'), int)
    assert isinstance(result.get('page_number'), int)

