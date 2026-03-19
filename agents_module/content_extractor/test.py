import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents_module.content_extractor.extractor import (  # noqa: E402
    OpenAISectionExtractor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenAI vision extraction on section images."
    )
    parser.add_argument(
        "--folder",
        default="data/Sections/svt",
        help="Folder containing section images.",
    )
    parser.add_argument(
        "--output",
        default="agents_module/content_extractor/test_output_sc.json",
        help="Path to save JSON output.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional OpenAI model override.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    extractor = OpenAISectionExtractor(model=args.model)
    results = extractor.extract_folder(args.folder)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved output to: {output_path}")

    for item in results:
        section_number = item.get("section_number")
        question = item.get("question")
        student_answer = item.get("student_answer")
        confidence = item.get("confidence")

        print(f"\nSECTION {section_number}")
        print(f"Question: {question}")
        print(f"Student Answer: {student_answer}")
        print(f"Confidence: {confidence}")


if __name__ == "__main__":
    main()
