from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
AGENTS_MODULE_DIR = Path(__file__).resolve().parents[1]
if str(AGENTS_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_MODULE_DIR))

GEMINI_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
client = genai.Client(vertexai=True, api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.1-pro-preview"


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627")
    text = text.replace("\u0622", "\u0627")
    text = text.replace("\u0649", "\u064a").replace("\u0629", "\u0647")
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("**", "")
    text = text.replace("\u2794", "->").replace("\u2192", "->")
    text = text.replace("\u21d2", "->")
    text = re.sub(r"\s+", " ", text)
    return text


def _image_part(image_path: Path) -> types.Part:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        file_bytes = f.read()
    return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)


def _parse_json_text(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def _detect_handwritten_content(image_path: Path) -> Dict[str, Any]:
    prompt = (
        "Analyze this exam image and decide if there is meaningful handwritten "
        "content. Return only JSON with keys: has_handwritten (boolean), "
        "confidence (0..1), reason (short string)."
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, _image_part(image_path)],
        )
        data = _parse_json_text(response.text if response else "")
        return {
            "has_handwritten": bool(data.get("has_handwritten", False)),
            "confidence": float(data.get("confidence", 0.0)),
            "reason": str(data.get("reason", "")),
        }
    except Exception:
        return {
            "has_handwritten": False,
            "confidence": 0.0,
            "reason": "handwriting_detection_failed",
        }


def _infer_is_submission(
    image_path: Path,
    manual_flag: Optional[bool],
) -> Dict[str, Any]:
    if manual_flag is not None:
        return {
            "is_submission": manual_flag,
            "decision_source": "manual_flag",
            "handwriting": None,
        }

    path_norm = _normalize_text(str(image_path).replace("\\", "/"))
    if "_is_submission" in path_norm or "submission" in path_norm:
        return {
            "is_submission": True,
            "decision_source": "path_rule",
            "handwriting": None,
        }

    correction_tokens = ["correction", "corrige", "corrig", "model_answer"]
    if any(tok in path_norm for tok in correction_tokens):
        return {
            "is_submission": False,
            "decision_source": "path_rule",
            "handwriting": None,
        }

    handwriting = _detect_handwritten_content(image_path)
    return {
        "is_submission": bool(handwriting.get("has_handwritten", False)),
        "decision_source": "handwriting_rule",
        "handwriting": handwriting,
    }


def _extract_relating_matches(
    raw_text: str,
    question_content: Dict[str, Any],
) -> List[Dict[str, str]]:
    items = question_content.get("items", []) if isinstance(question_content, dict) else []
    options = question_content.get("options", []) if isinstance(question_content, dict) else []
    if not isinstance(items, list) or not isinstance(options, list):
        return []

    item_lookup = {
        _normalize_text(str(item.get("text", ""))): str(item.get("id", ""))
        for item in items
        if isinstance(item, dict)
    }
    option_lookup = {
        _normalize_text(str(opt.get("text", ""))): str(opt.get("id", ""))
        for opt in options
        if isinstance(opt, dict)
    }

    lines = [line.strip() for line in str(raw_text or "").splitlines() if line.strip()]
    matches: List[Dict[str, str]] = []
    for line in lines:
        parts = re.split(r"\s*->\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue

        left = _normalize_text(parts[0])
        right = _normalize_text(parts[1])
        item_id = item_lookup.get(left)
        option_id = option_lookup.get(right)
        if item_id and option_id:
            matches.append({"item_id": item_id, "option_id": option_id})

    return matches


def _diagram_parts_lookup(question_content: Dict[str, Any]) -> Dict[str, str]:
    parts = question_content.get("parts_to_label", [])
    lookup: Dict[str, str] = {}
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_id = str(part.get("id", "")).strip()
            description = str(part.get("description", "")).strip()
            if part_id:
                lookup[part_id] = description

    if not lookup:
        lookup = {
            "1": "left image",
            "2": "center image",
            "3": "right image",
        }
    return lookup


def _extract_diagram_selected_parts(
    raw_text: str,
    question_content: Dict[str, Any],
) -> List[str]:
    normalized = _normalize_text(raw_text)
    ids: List[str] = []

    canonical_hits = re.findall(r"selected_parts\s*:\s*([0-9, ]+)", normalized)
    if canonical_hits:
        for token in canonical_hits[-1].split(","):
            token_clean = token.strip()
            if token_clean in {"1", "2", "3"}:
                ids.append(token_clean)

    pair_patterns = [
        r"middle\s*(and|&)\s*right",
        r"right\s*(and|&)\s*middle",
        r"\u0627\u0644\u0648\u0633\u0637(?:\u0649)?\s*\u0648\s*\u0627\u0644\u064a\u0645\u064a\u0646(?:\u0649)?",
        r"\u0627\u0644\u064a\u0645\u064a\u0646(?:\u0649)?\s*\u0648\s*\u0627\u0644\u0648\u0633\u0637(?:\u0649)?",
    ]
    if not ids and any(re.search(p, normalized) for p in pair_patterns):
        ids.extend(["2", "3"])

    if "2" not in ids and any(
        k in normalized
        for k in [
            "middle",
            "center",
            "\u0648\u0633\u0637",
            "\u0627\u0648\u0633\u0637",
            "\u0627\u0644\u0627\u0648\u0633\u0637",
            "\u0627\u0644\u0648\u0633\u0637\u0649",
            "\u0634\u0645\u0639\u0647",
            "candle",
        ]
    ):
        ids.append("2")
    if "3" not in ids and any(
        k in normalized
        for k in [
            "right",
            "\u064a\u0645\u064a\u0646",
            "\u0627\u064a\u0645\u0646",
            "\u0627\u0644\u0627\u064a\u0645\u0646",
            "\u0627\u0644\u064a\u0645\u0646\u0649",
            "\u0645\u063a\u0646\u0627\u0637\u064a\u0633",
            "magnet",
        ]
    ):
        ids.append("3")
    if "1" not in ids and any(
        k in normalized
        for k in [
            "left",
            "\u064a\u0633\u0627\u0631",
            "\u0627\u0644\u064a\u0633\u0631\u0649",
            "\u0645\u0627\u0621 \u0628\u0627\u0631\u062f",
            "cold water",
        ]
    ):
        ids.append("1")

    unique_ids: List[str] = []
    for value in ids:
        if value not in unique_ids:
            unique_ids.append(value)

    parts_lookup = _diagram_parts_lookup(question_content)
    return [value for value in unique_ids if value in parts_lookup]


def _build_structured_answer(
    question_type: str,
    question_content: Dict[str, Any],
    raw_text: str,
) -> Dict[str, Any]:
    answer: Dict[str, Any] = {"raw_text": raw_text or "[UNK]"}
    normalized_type = str(question_type).upper().strip()

    if normalized_type == "RELATING":
        answer["matches"] = _extract_relating_matches(raw_text, question_content)

    if normalized_type == "DIAGRAM":
        selected_ids = _extract_diagram_selected_parts(raw_text, question_content)
        if selected_ids:
            lookup = _diagram_parts_lookup(question_content)
            answer["selected_parts"] = selected_ids
            answer["selected_part_labels"] = [
                lookup[part_id] for part_id in selected_ids if part_id in lookup
            ]
            answer["raw_text"] = f"selected_parts:{','.join(selected_ids)}"

    return answer


def _extract_answer_text(image_path: Path) -> str:
    import ocr_gemini
    from extract_correction_content import extract_correction_content

    ocr_text = (ocr_gemini.run_ocr(str(image_path)) or "").strip()
    if ocr_text:
        return ocr_text

    try:
        extracted = extract_correction_content(str(image_path))
        if not extracted:
            return "[UNK]"
        options = (
            extracted.get("content", {})
            .get("correct_answer", {})
            .get("correct answer ", [])
        )
        if not isinstance(options, list):
            return "[UNK]"

        fallback_text = "\n".join(
            str(item.get("text", ""))
            for item in options
            if isinstance(item, dict)
        ).strip()
        return fallback_text if fallback_text else "[UNK]"
    except Exception:
        return "[UNK]"


def build_unified_content(
    image_path: Path,
    manual_flag: Optional[bool],
) -> Dict[str, Any]:
    from question_classifier import QuestionClassifier

    classifier = QuestionClassifier()
    classification = classifier.classify_question(image_path)
    question_block = classifier.extract_question_content(
        image_path,
        classification["question_type"],
    )

    question_type = classification.get("question_type", "UNKNOWN")
    confidence = float(classification.get("confidence", 0.0))
    question_content = question_block.get("content", {})

    mode_info = _infer_is_submission(image_path, manual_flag)
    is_submission = bool(mode_info["is_submission"])
    flag = "submission" if is_submission else "correction"

    raw_text = _extract_answer_text(image_path)
    answer = _build_structured_answer(question_type, question_content, raw_text)

    content_block: Dict[str, Any] = {
        "content": question_content,
        "notes": ["1 point"],
        "confidence": confidence,
    }
    if is_submission:
        content_block["student_submission"] = answer
        content_block["student_answer"] = answer
    else:
        content_block["correct_answer"] = answer

    item = {
        "question_type": question_type,
        "confidence": confidence,
        "content": content_block,
        "meta_data": {
            "image_path": str(image_path),
            "image_name": image_path.name,
        },
    }

    return {
        "flag": flag,
        "is_submission_flag": is_submission,
        "decision_source": mode_info.get("decision_source"),
        "handwriting_detection": mode_info.get("handwriting"),
        "exam_content": [item],
    }


def _parse_manual_flag(value: str) -> Optional[bool]:
    normalized = (value or "auto").strip().lower()
    if normalized == "auto":
        return None
    if normalized in {"1", "true", "yes", "submission", "sub"}:
        return True
    if normalized in {"0", "false", "no", "correction", "corr"}:
        return False
    raise ValueError("--is_submission_flag must be one of: auto|true|false")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified exam image analyzer for submission/correction"
    )
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument(
        "--is_submission_flag",
        default="auto",
        help="auto|true|false (default: auto)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Default: "
            "agents_module/content_correction.py/output_json/"
            "<image_name>_unified.json"
        ),
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    manual_flag = _parse_manual_flag(args.is_submission_flag)
    result = build_unified_content(image_path, manual_flag)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(
            "agents_module/content_correction.py/output_json"
        ) / f"{image_path.stem}_unified.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON: {output_path}")


if __name__ == "__main__":
    main()

