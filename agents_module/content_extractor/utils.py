"""Utility helpers for the content extraction pipeline."""

import json
import re
from pathlib import Path
from typing import Dict, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_section_number(filename: str) -> int:
    """Extract section number from names like exam_1_section_2.jpg."""
    match = re.search(r"section[_\s-](\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else -1


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


def detect_section_type(image_path: Path) -> str:
    """Best-effort section type detection from file name only."""
    name = image_path.stem.lower()
    if "instr" in name or "enonce" in name or "instruction" in name:
        return "instruction"
    if "answer" in name or "response" in name or "student" in name:
        return "answer_zone"
    if "question" in name or "q_" in name or "q-" in name:
        return "question"
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
