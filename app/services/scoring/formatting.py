"""Format compliance scoring: deterministic checks only, no network, cannot
fail in a demo. This is why it's built first among the scoring components
(see CLAUDE.md's build order).

Nine checks, each with its own point value, totalling 100 when every check
applies. A tenth check (no_header_footer) was cut: text_in_header_footer
always returns False on a single-page resume (see layout.py), so it could
never fail in practice -- its 8 points were redistributed, +4 each to
no_tables and single_column, which are the two checks that keep doing real
work against real bad fixtures (demo_before.pdf has a genuine ruled skills
grid and a genuine two-column layout).

reasonable_length is not applicable for DOCX input, since text_extract.py
doesn't compute a page count for DOCX -- see _check_reasonable_length.
When that happens its 8 points are dropped from points_possible rather
than scored as a failure, so a DOCX resume is scored out of 92, not
unfairly docked for a property that was never actually measured.
"""

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Deliberately loose: finds any digit-heavy run (allowing +, spaces,
# dashes, dots, parens as separators) and validates it by counting the
# digits, rather than trying to hand-write one regex for every phone
# format. 10 digits covers a plain Indian mobile number or a US-style
# "(123) 456-7890"; 11-13 covers a country code like +91 attached to it.
_PHONE_CANDIDATE_RE = re.compile(r"[+(]?\d[\d\-.\s()]{6,}\d\)?")
_PHONE_MIN_DIGITS = 10
_PHONE_MAX_DIGITS = 13

TEXT_EXTRACTABLE_MIN_CHARS = 300

CORE_SECTIONS = {"experience", "education", "skills"}

_YEAR_TOKEN_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_PRESENT_CURRENT_RE = re.compile(r"\b(?:present|current)\b", re.IGNORECASE)
MIN_YEAR_TOKENS = 2

REASONABLE_LENGTH_MIN_PAGES = 1
REASONABLE_LENGTH_MAX_PAGES = 2

# Broader than text_extract.py's own bullet-normalisation set -- this is a
# defensive re-check for anything that made it through clean_text() (or
# text that bypassed it) still carrying a decorative bullet glyph.
_EXOTIC_BULLET_CHARS = "•▪◦‣●·○■□▶➤➔❖✦∙"
_EXOTIC_BULLET_RE = re.compile(f"[{re.escape(_EXOTIC_BULLET_CHARS)}]")


def _has_phone(text: str) -> bool:
    for match in _PHONE_CANDIDATE_RE.finditer(text):
        digit_count = len(re.sub(r"\D", "", match.group(0)))
        if _PHONE_MIN_DIGITS <= digit_count <= _PHONE_MAX_DIGITS:
            return True
    return False


def _has_dates(text: str) -> bool:
    if len(_YEAR_TOKEN_RE.findall(text)) >= MIN_YEAR_TOKENS:
        return True
    return bool(_PRESENT_CURRENT_RE.search(text))


def _check_has_email(resume_text: str, layout: dict, sections_found: list[str]) -> bool:
    return bool(EMAIL_RE.search(resume_text))


def _check_has_phone(resume_text: str, layout: dict, sections_found: list[str]) -> bool:
    return _has_phone(resume_text)


def _check_text_extractable(resume_text: str, layout: dict, sections_found: list[str]) -> bool:
    return len(resume_text.strip()) > TEXT_EXTRACTABLE_MIN_CHARS


def _check_has_core_sections(resume_text: str, layout: dict, sections_found: list[str]) -> bool:
    return CORE_SECTIONS.issubset({s.lower() for s in sections_found})


def _check_no_tables(resume_text: str, layout: dict, sections_found: list[str]) -> bool:
    return not layout.get("has_tables", False)


def _check_single_column(resume_text: str, layout: dict, sections_found: list[str]) -> bool:
    return not layout.get("is_multicolumn", False)


def _check_reasonable_length(resume_text: str, layout: dict, sections_found: list[str]) -> bool | None:
    """None means not applicable -- currently true for all DOCX input,
    since text_extract.py doesn't compute a page count for DOCX. Scoring
    that as a fail would penalise every Word resume for a property we
    never actually measured; format_score() excludes a None result from
    points_possible entirely instead of counting it against the resume.
    """
    page_count = layout.get("page_count")
    if page_count is None:
        return None
    return REASONABLE_LENGTH_MIN_PAGES <= page_count <= REASONABLE_LENGTH_MAX_PAGES


def _check_has_dates(resume_text: str, layout: dict, sections_found: list[str]) -> bool:
    return _has_dates(resume_text)


def _check_standard_bullets(resume_text: str, layout: dict, sections_found: list[str]) -> bool:
    return not _EXOTIC_BULLET_RE.search(resume_text)


# (check name, points, check function, message shown when the check fails)
CHECKS: list[tuple[str, int, object, str]] = [
    (
        "has_email",
        8,
        _check_has_email,
        "No email address found — recruiters and ATS software both need a way to contact you.",
    ),
    (
        "has_phone",
        6,
        _check_has_phone,
        "No phone number found — add one in a standard format, e.g. +91 98765 43210.",
    ),
    (
        "text_extractable",
        15,
        _check_text_extractable,
        "Very little text could be extracted from this file — it may be a scanned image "
        "rather than real text, which most ATS software cannot read at all.",
    ),
    (
        "has_core_sections",
        15,
        _check_has_core_sections,
        "Missing a standard experience, education, or skills section heading — ATS "
        "section parsers look for these headings by name to structure the resume.",
    ),
    (
        "no_tables",
        16,
        _check_no_tables,
        "Tables detected — ATS parsers frequently misread table cells as scrambled or "
        "out-of-order text, or drop their contents entirely.",
    ),
    (
        "single_column",
        16,
        _check_single_column,
        "Two-column layout — parsers may interleave the columns, scrambling reading order.",
    ),
    (
        "reasonable_length",
        8,
        _check_reasonable_length,
        "Resume is not 1-2 pages — most ATS workflows and recruiters expect a concise "
        "resume in that range.",
    ),
    (
        "has_dates",
        8,
        _check_has_dates,
        "Few or no dates found — ATS software relies on employment dates to compute "
        "years of experience, and missing dates undercount it.",
    ),
    (
        "standard_bullets",
        8,
        _check_standard_bullets,
        "Non-standard bullet glyphs found — decorative bullet characters can render as "
        "garbled symbols or get stripped out by some parsers.",
    ),
]


def format_score(resume_text: str, layout: dict, sections_found: list[str]) -> dict:
    """Run all format compliance checks and score the result out of 100.

    layout is the dict returned by layout_signals(); sections_found is
    ParsedResume.sections_found. Every check is pure Python -- no network,
    no LLM, so this can never fail to produce a score.
    """
    points_earned = 0
    points_possible = 0
    issues = []
    skipped = []

    for name, points, check_fn, message in CHECKS:
        result = check_fn(resume_text, layout, sections_found)

        if result is None:
            # Not applicable -- excluded from points_possible entirely,
            # not scored as a failure. See _check_reasonable_length.
            skipped.append({"check": name, "status": "not_applicable"})
            continue

        points_possible += points
        if result:
            points_earned += points
        else:
            issues.append({"check": name, "penalty": points, "message": message})

    issues.sort(key=lambda issue: issue["penalty"], reverse=True)

    score = round(points_earned / points_possible, 4) if points_possible else 0.0

    return {
        "score": score,
        "points_earned": points_earned,
        "points_possible": points_possible,
        "issues": issues,
        "skipped": skipped,
    }
