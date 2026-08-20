"""Plain text extraction from resume files (PDF, DOCX).

Pure extraction only: pulls raw text out of a file and cleans it up. No
parsing, no LLM calls, no scoring.
"""

import os
import re
import tempfile
from pathlib import Path

import docx2txt
import fitz  # PyMuPDF

NEEDS_OCR_CHAR_THRESHOLD = 200

# Gutter detection for multi-column pages. We scan this middle band of the
# page width looking for a vertical strip no text block crosses.
MULTICOLUMN_SCAN_START_FRACTION = 0.25
MULTICOLUMN_SCAN_END_FRACTION = 0.80
MULTICOLUMN_SCAN_STEP_PX = 5
MULTICOLUMN_MIN_GUTTER_PX = 20

BULLET_CHARS = "•▪◦‣●·"
_BULLET_RE = re.compile(f"[{re.escape(BULLET_CHARS)}]")
_NULL_BYTE_RE = re.compile("\x00")
_SPACE_TAB_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


class ExtractionError(Exception):
    """Raised when text cannot be extracted from the given file."""


def _group_into_bands(text_blocks: list) -> list[list]:
    """Group blocks into bands of vertically-overlapping blocks.

    Blocks land in the same band when their y-ranges overlap, directly or
    through a chain of overlaps. A full-width header/contact line has no
    y-overlap with the two-column body beneath it, so it lands in its own
    band and can't hide a real gutter that only exists lower on the page.
    """
    sorted_blocks = sorted(text_blocks, key=lambda b: b[1])
    bands: list[list] = []
    band_y1 = None

    for block in sorted_blocks:
        y0, y1 = block[1], block[3]
        if bands and y0 <= band_y1:
            bands[-1].append(block)
            band_y1 = max(band_y1, y1)
        else:
            bands.append([block])
            band_y1 = y1

    return bands


def _band_gutter(blocks_in_band: list, page_width: float) -> tuple[float, float] | None:
    """Find the gutter for one band of vertically-overlapping blocks.

    Scans candidate x positions across the middle of the page looking for a
    vertical strip that no block's x-range spans. The widest such gap is
    the candidate gutter; it counts as a real gutter only if it's at least
    MULTICOLUMN_MIN_GUTTER_PX wide and there is text entirely to its left
    and entirely to its right. Returns (gutter_start, gutter_end), or None
    if the band has no gutter.
    """
    x_ranges = [(b[0], b[2]) for b in blocks_in_band]

    scan_start = page_width * MULTICOLUMN_SCAN_START_FRACTION
    scan_end = page_width * MULTICOLUMN_SCAN_END_FRACTION

    positions = []
    x = scan_start
    while x <= scan_end:
        positions.append(x)
        x += MULTICOLUMN_SCAN_STEP_PX
    if not positions:
        return None

    is_clear = [not any(x0 <= x <= x1 for x0, x1 in x_ranges) for x in positions]

    best_start_idx = None
    best_width = 0.0
    best_end_idx = None
    run_start_idx = None
    for i, clear in enumerate(is_clear):
        if clear:
            if run_start_idx is None:
                run_start_idx = i
            run_width = positions[i] - positions[run_start_idx]
            if run_width > best_width:
                best_width = run_width
                best_start_idx = run_start_idx
                best_end_idx = i
        else:
            run_start_idx = None

    if best_start_idx is None or best_width < MULTICOLUMN_MIN_GUTTER_PX:
        return None

    gutter_start = positions[best_start_idx]
    gutter_end = positions[best_end_idx]

    has_left_text = any(x1 <= gutter_start for x0, x1 in x_ranges)
    has_right_text = any(x0 >= gutter_end for x0, x1 in x_ranges)

    if not (has_left_text and has_right_text):
        return None

    return gutter_start, gutter_end


def detect_multicolumn(page: "fitz.Page") -> bool:
    """Vertical-gutter multi-column detection for a single PyMuPDF page.

    Blocks are grouped into horizontal bands by y-overlap first (see
    _group_into_bands), then each band is scanned independently for a
    gutter (see _band_gutter). Banding keeps a full-width header or
    section line from blocking gutter detection for the whole page.
    """
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[4].strip()]
    if not text_blocks:
        return False

    page_width = page.rect.width
    bands = _group_into_bands(text_blocks)

    return any(_band_gutter(band, page_width) is not None for band in bands)


def _dominant_gutter_x(bands: list[list], page_width: float) -> float | None:
    """Median gutter midpoint across bands that have one.

    Median rather than mean so one band with a noisy/off gutter position
    doesn't drag the whole-page split away from where most bands agree it
    is.
    """
    midpoints = []
    for band in bands:
        gutter = _band_gutter(band, page_width)
        if gutter is not None:
            midpoints.append((gutter[0] + gutter[1]) / 2)

    if not midpoints:
        return None

    midpoints.sort()
    mid = len(midpoints) // 2
    if len(midpoints) % 2:
        return midpoints[mid]
    return (midpoints[mid - 1] + midpoints[mid]) / 2


def _column_ordered_text(page: "fitz.Page") -> str:
    """Reading-order text for a page already known to be multi-column.

    Finds the dominant gutter x-position from the page's banded rows, then
    partitions every text block on the page around that single x: blocks
    entirely right of the gutter (x0 > gutter) go in the right group,
    everything else -- blocks entirely left of it, plus any block that
    straddles it (typically a full-width name/contact header) -- goes in
    the left group. Each group is sorted top-to-bottom, then the left
    column is emitted in full before the right column. Straddling blocks
    sort to the top of the left group on their own low y, which is where a
    full-width header belongs anyway.
    """
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[4].strip()]
    if not text_blocks:
        return ""

    page_width = page.rect.width
    bands = _group_into_bands(text_blocks)
    gutter_x = _dominant_gutter_x(bands, page_width)

    if gutter_x is None:
        # Bands disagree on where the gutter is -- shouldn't happen once
        # the caller has confirmed detect_multicolumn(page), but fall back
        # to natural block order rather than guessing.
        return page.get_text(sort=False)

    left, right = [], []
    for block in text_blocks:
        if block[0] > gutter_x:
            right.append(block)
        else:
            left.append(block)

    left.sort(key=lambda b: b[1])
    right.sort(key=lambda b: b[1])

    return "\n".join(block[4] for block in left + right)


def clean_text(text: str) -> str:
    """Strip null bytes, collapse whitespace, normalise bullet glyphs."""
    text = _NULL_BYTE_RE.sub("", text)
    text = _BULLET_RE.sub("-", text)
    text = _SPACE_TAB_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _extract_pdf(file_bytes: bytes) -> tuple[str, str, int]:
    """Returns (text, naive_text, page_count).

    text is column-ordered reading order for multi-column pages, natural
    block order otherwise. naive_text is what page.get_text(sort=...) gives
    on its own -- y-then-x sort interleaves columns on a multi-column page,
    which is exactly the failure mode text is built to avoid. naive_text is
    never used for scoring or downstream parsing; it exists only so the UI
    can show the user what a naive parser sees as evidence for why a fix is
    needed.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        page_count = doc.page_count
        multicolumn = detect_multicolumn(doc[0]) if page_count > 0 else False

        text_parts = []
        naive_parts = []
        for page in doc:
            naive_parts.append(page.get_text(sort=multicolumn))
            if multicolumn:
                text_parts.append(_column_ordered_text(page))
            else:
                text_parts.append(page.get_text(sort=False))

        text = "\n".join(text_parts)
        naive_text = "\n".join(naive_parts)
    finally:
        doc.close()
    return text, naive_text, page_count


def _extract_docx(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(file_bytes)
        return docx2txt.process(tmp_path) or ""
    finally:
        os.unlink(tmp_path)


def extract_text(file_bytes: bytes, filename: str) -> dict:
    """Extract and clean text from a resume file.

    Dispatches on the file extension: .pdf via PyMuPDF, .docx/.doc via
    docx2txt. Raises ExtractionError for anything else.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        raw_text, raw_naive_text, page_count = _extract_pdf(file_bytes)
        method = "pymupdf"
    elif ext in (".docx", ".doc"):
        raw_text = _extract_docx(file_bytes, filename)
        raw_naive_text = raw_text  # no column layout to reorder in a docx
        page_count = None
        method = "docx2txt"
    else:
        raise ExtractionError(f"Unsupported file type: {ext or filename}")

    text = clean_text(raw_text)
    naive_text = clean_text(raw_naive_text)
    char_count = len(text)

    return {
        "text": text,
        "naive_text": naive_text,
        "method": method,
        "page_count": page_count,
        "char_count": char_count,
        "needs_ocr": char_count < NEEDS_OCR_CHAR_THRESHOLD,
    }
