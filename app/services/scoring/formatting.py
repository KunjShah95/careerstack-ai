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

# Ordered (not a set) since the failure message names sections in this
# order, and display names for the message text.
CORE_SECTIONS = ["experience", "education", "skills"]
CORE_SECTION_DISPLAY_NAMES = {"experience": "Experience", "education": "Education", "skills": "Skills"}

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


def _experience_satisfied(found: set[str]) -> bool:
    """A student or fresher with no employment history legitimately has no
    Experience section -- a Projects section is accepted as a substitute.
    Education and skills get no such allowance; see _missing_core_sections.
    """
    return "experience" in found or "projects" in found


def _missing_core_sections(sections_found: list[str]) -> list[str]:
    """Which of experience/education/skills are absent, in that order --
    "experience" here already accounts for the projects substitute.
    """
    found = {s.lower() for s in sections_found}
    missing = []
    if not _experience_satisfied(found):
        missing.append("experience")
    for section in ("education", "skills"):
        if section not in found:
            missing.append(section)
    return missing


def _core_sections_passed_via_projects(sections_found: list[str]) -> bool:
    found = {s.lower() for s in sections_found}
    return "experience" not in found and "projects" in found


def _check_has_core_sections(resume_text: str, layout: dict, sections_found: list[str]) -> bool:
    return not _missing_core_sections(sections_found)


def _join_with_or(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} or {names[1]}"
    return f"{', '.join(names[:-1])}, or {names[-1]}"


def _has_core_sections_message(resume_text: str, layout: dict, sections_found: list[str]) -> str:
    """Names exactly which heading(s) are missing, correctly pluralised --
    not a static "one of these three" message regardless of how many are
    actually absent.
    """
    missing = _missing_core_sections(sections_found)
    names = [CORE_SECTION_DISPLAY_NAMES[section] for section in missing]
    joined = _join_with_or(names)
    heading_word = "heading" if len(names) == 1 else "headings"
    pronoun = "this" if len(names) == 1 else "these"
    return (
        f"No {joined} section {heading_word} found — ATS parsers look for "
        f"{pronoun} {heading_word} by name to structure the resume."
    )


def _has_core_sections_note(resume_text: str, layout: dict, sections_found: list[str]) -> str | None:
    """Surfaced (via format_score's "notes" list) when the check passed by
    accepting Projects in place of Experience, so that substitution is
    visible rather than silently folded into a plain pass.
    """
    if _core_sections_passed_via_projects(sections_found):
        return (
            "No Experience section found, but a Projects section was accepted in its "
            "place (no employment history) — worth knowing, since not every ATS makes "
            "the same allowance."
        )
    return None


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
        # A callable message, unlike every other check's static string --
        # format_score() calls it with the same (resume_text, layout,
        # sections_found) args to get a message naming exactly what's
        # missing, since that varies per resume.
        "has_core_sections",
        15,
        _check_has_core_sections,
        _has_core_sections_message,
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


# Optional per-check function returning a note when a check passed via a
# documented substitution (rather than the literal thing being present),
# so that stays visible instead of reading as a plain, unqualified pass.
# Only has_core_sections uses this today (the projects-for-experience
# allowance).
_PASS_NOTE_FNS = {"has_core_sections": _has_core_sections_note}


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
    notes = []

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
            note_fn = _PASS_NOTE_FNS.get(name)
            if note_fn:
                note = note_fn(resume_text, layout, sections_found)
                if note:
                    notes.append({"check": name, "note": note})
        else:
            resolved_message = message(resume_text, layout, sections_found) if callable(message) else message
            issues.append({"check": name, "penalty": points, "message": resolved_message})

    issues.sort(key=lambda issue: issue["penalty"], reverse=True)

    score = round(points_earned / points_possible, 4) if points_possible else 0.0

    return {
        "score": score,
        "points_earned": points_earned,
        "points_possible": points_possible,
        "issues": issues,
        "skipped": skipped,
        "notes": notes,
    }
