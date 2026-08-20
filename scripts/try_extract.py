"""Manual eyeball CLI for text_extract.py / layout.py. Not a test.

Usage:
    python scripts/try_extract.py path/to/resume.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.extraction.layout import layout_signals
from app.services.extraction.text_extract import extract_text


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/try_extract.py <path_to_resume>")
        sys.exit(1)

    path = Path(sys.argv[1])
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


if __name__ == "__main__":
    main()
