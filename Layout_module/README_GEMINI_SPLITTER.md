# Gemini Image Splitter

Splits exam pages into sections using Google Gemini Vision API.

## Features

- **Keyword Detection**: Finds Arabic keywords (تعليمة, سند) in exam images
- **Smart Splitting**: Divides pages into logical sections based on keyword positions
- **Flexible Input**: Accepts PIL Images, numpy arrays, or file paths
- **Auto-saving**: Save sections as individual images

## Quick Start

### 1. Set your API key

Add to your `.env` file:
```env
GEMINI_AI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-2.0-flash-exp
```

### 2. Basic usage

```python
from Layout_module.gemini_image_splitter import GeminiImageSplitter

# Initialize
splitter = GeminiImageSplitter()

# Split image by keywords
sections = splitter.split_image_by_keywords("path/to/exam.jpg")

# Save sections
splitter.save_sections(sections, "output_dir", prefix="exam_1")
```

### 3. Test with CLI

```bash
python Layout_module/test_gemini_splitter.py --image "path/to/exam.jpg" --output "data/Sections/test"
```

## Configuration

Keywords and excluded words are defined in `layout_config.py`:

```python
KEY_WORDS = ["تعليمة", "سند"]          # Section start keywords
EXCLUDED_KEYWORDS = ["تسند"]           # Words to ignore
```

## How It Works

1. **Text Extraction**: Gemini Vision API extracts all text from image
2. **Keyword Matching**: Finds target keywords and their approximate positions
3. **Section Boundaries**: Determines Y-coordinates for splitting
4. **Image Cropping**: Splits original image at boundary points
5. **Output**: Returns list of `ImageSection` objects

## Section Structure

```python
@dataclass
class ImageSection:
    section_index: int          # 0, 1, 2, ...
    keyword_trigger: str        # "تعليمة" or "سند"
    y_start: int                # Top Y-coordinate
    y_end: int                  # Bottom Y-coordinate
    image: Image.Image          # PIL Image of section
```

## Advanced Options

```python
sections = splitter.split_image_by_keywords(
    image="exam.jpg",
    min_section_height=100,     # Skip sections < 100px
)
```

## Requirements

```bash
pip install google-generativeai pillow opencv-python
```

## Notes

- Gemini Flash 2.0 doesn't provide exact bounding boxes
- Y-positions are estimated based on text flow order
- Works best with clear, well-formatted exams
- Rate limited to avoid API quota issues
