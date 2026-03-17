"""Utility helpers for the content extraction pipeline."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_section_number(filename: str) -> int:
    """Extract section number from common section-image file names."""
    patterns = [
        r"section[_\s-](\d+)",
        r"(?:^|[_\s-])p(\d+)(?:\.[^.]+)?$",
        r"(?:^|[_\s-])(\d+)(?:\.[^.]+)?$",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return -1


def load_images(folder: Path) -> List[Path]:
    """Load image files from folder sorted by section number."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    paths = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=lambda p: get_section_number(p.name))
    return paths


def sort_results(results: List[Dict]) -> List[Dict]:
    """Sort results by section number ascending."""
    return sorted(results, key=lambda r: int(r.get("section_number", -1)))


def _looks_like_table(text: str) -> bool:
    """Heuristic for table-like content using separators or keywords."""
    if re.search(
        r"(?:^|\s)(table|row|col|column|جدول|عمود|سطر)(?:\s|$)",
        text,
    ):
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False

    separator_rich_lines = 0
    for line in lines:
        sep_count = line.count("|") + line.count(";") + line.count(":")
        if sep_count >= 2:
            separator_rich_lines += 1

    return separator_rich_lines >= 2


def detect_section_type(
    image_path: Path,
    extracted_text: Optional[str] = None,
) -> str:
    """Best-effort section type detection from filename and optional text."""
    name = image_path.stem.lower()
    text = (extracted_text or "").lower()
    source = f"{name} {text}"

    if _looks_like_table(source):
        return "TABLE"

    if any(k in source for k in ["اختر", "choose", "multiple", "choice"]):
        return "MULTIPLE_CHOICE"

    if any(k in source for k in ["اربط", "match", "matching", "relat"]):
        return "RELATING"

    if any(k in source for k in ["أكمل", "اكمل", "fill", "blank", "___"]):
        return "FILL_BLANK"

    if (
        "صح أو خطأ" in source
        or "صح او خطا" in source
        or "true false" in source
        or ("صح" in source and "خط" in source)
    ):
        return "TRUE_FALSE"

    if any(k in source for k in ["ارسم", "diagram", "shape", "شكل", "label"]):
        return "DIAGRAM"

    if any(
        k in source
        for k in ["اكتب", "writing", "paragraph", "expression"]
    ):
        return "WRITING"

    if any(
        k in source
        for k in [
            "احسب",
            "عملية",
            "calculate",
            "sum",
            "-",
            "+",
            "=",
            "×",
            "÷",
        ]
    ):
        return "CALCULATION"

    if any(
        k in source
        for k in ["تعليمة", "تعليمات", "enonce", "instruction", "instr"]
    ):
        return "ENONCE"

    if any(
        k in source
        for k in [
            "علل",
            "فسر",
            "اذكر",
            "short",
            "answer",
            "question",
            "q_",
            "q-",
        ]
    ):
        return "SHORT_ANSWER"

    if "answer_zone" in source or "response" in source or "student" in source:
        return "WRITING"

    return "unknown"


def extract_json_from_text(text: str) -> Dict:
    """Parse JSON from raw model output (plain JSON or fenced block)."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return {
                "question": None,
                "student_answer": None,
                "confidence": 0.0,
                "raw_response": cleaned,
            }
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "question": None,
                "student_answer": None,
                "confidence": 0.0,
                "raw_response": cleaned,
            }
