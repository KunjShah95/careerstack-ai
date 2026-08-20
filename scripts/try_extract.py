"""Manual eyeball CLI for text_extract.py / layout.py / llm_parse.py. Not a
test.

Usage:
    python scripts/try_extract.py path/to/resume.pdf
    python scripts/try_extract.py path/to/resume.pdf --parse
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.extraction.layout import layout_signals
from app.services.extraction.llm_parse import parse_resume
from app.services.extraction.text_extract import extract_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a resume PDF or DOCX")
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Also run parse_resume() against the extracted text and pretty-print it",
    )
    args = parser.parse_args()

    path = Path(args.path)
    file_bytes = path.read_bytes()

    result = extract_text(file_bytes, path.name)

    print("=== extract_text ===")
    for key, value in result.items():
        if key not in ("text", "naive_text"):
            print(f"{key}: {value}")

    if path.suffix.lower() == ".pdf":
        signals = layout_signals(file_bytes)
        print("\n=== layout_signals ===")
        for key, value in signals.items():
            print(f"{key}: {value}")

    print("\n=== first 400 chars of text ===")
    print(result["text"][:400])

    if args.parse:
        print("\n=== parse_resume ===")
        resume = parse_resume(result["text"])
        print(json.dumps(resume.model_dump(), indent=2))


if __name__ == "__main__":
    main()
