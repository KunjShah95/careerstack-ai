"""Layout signal detection for PDF resumes.

Structural signals only (images, tables, columns, header/footer text, fonts).
These feed format_compliance scoring later; no scoring happens here.
"""

import io

import fitz  # PyMuPDF
import pdfplumber

from app.services.extraction.text_extract import detect_multicolumn

HEADER_FOOTER_MARGIN = 0.06  # top/bottom 6% of page height

# A real table needs at least this many rows/columns; a single divider
# rule under a section heading extracts as a 1x1 or 1xN "table" and isn't
# one.
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2

# A running header/footer needs to repeat on at least this many pages.
MIN_HEADER_FOOTER_REPEATS = 2


def _page_has_real_table(page: "pdfplumber.page.Page") -> bool:
    for table in page.find_tables():
        rows = table.extract()
        n_cols = max((len(row) for row in rows), default=0)
        if len(rows) >= MIN_TABLE_ROWS and n_cols >= MIN_TABLE_COLS:
            return True
    return False


def _normalize_band_text(text: str) -> str:
    return " ".join(text.split()).lower()


def _has_repeated_band_text(doc: "fitz.Document") -> bool:
    """True if the same-looking text sits in the top or bottom band on two
    or more pages -- a real running header/footer, not just content that
    happens to fall near a page edge.
    """
    top_page_counts: dict[str, int] = {}
    bottom_page_counts: dict[str, int] = {}

    for page in doc:
        page_height = page.rect.height
        top_bound = page_height * HEADER_FOOTER_MARGIN
        bottom_bound = page_height * (1 - HEADER_FOOTER_MARGIN)

        top_on_page = set()
        bottom_on_page = set()

        for block in page.get_text("blocks"):
            text = block[4].strip()
            if not text:
                continue
            normalized = _normalize_band_text(text)
            y0, y1 = block[1], block[3]
            if y0 <= top_bound:
                top_on_page.add(normalized)
            if y1 >= bottom_bound:
                bottom_on_page.add(normalized)

        for normalized in top_on_page:
            top_page_counts[normalized] = top_page_counts.get(normalized, 0) + 1
        for normalized in bottom_on_page:
            bottom_page_counts[normalized] = bottom_page_counts.get(normalized, 0) + 1

    return any(count >= MIN_HEADER_FOOTER_REPEATS for count in top_page_counts.values()) or any(
        count >= MIN_HEADER_FOOTER_REPEATS for count in bottom_page_counts.values()
    )


def layout_signals(file_bytes: bytes) -> dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        page_count = doc.page_count
        has_images = False
        font_families: set[str] = set()
        is_multicolumn = False

        for index, page in enumerate(doc):
            if page.get_images():
                has_images = True

            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            font_families.add(span.get("font", ""))

            if index == 0:
                is_multicolumn = detect_multicolumn(page)

        # A single-page PDF has no structural header/footer -- the
        # name/contact block at the top of a one-page resume is normal
        # content, not a repeating page header. Only count top/bottom-band
        # text as a header/footer once it actually repeats across pages.
        text_in_header_footer = page_count > 1 and _has_repeated_band_text(doc)
    finally:
        doc.close()

    has_tables = False
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            if _page_has_real_table(page):
                has_tables = True
                break

    return {
        "has_images": has_images,
        "has_tables": has_tables,
        "is_multicolumn": is_multicolumn,
        "page_count": page_count,
        "font_families": sorted(font_families),
        "text_in_header_footer": text_in_header_footer,
    }
