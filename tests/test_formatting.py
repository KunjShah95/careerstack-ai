"""Tests for the deterministic format compliance checks in formatting.py.

These are pure functions, so they're cheap to test directly against known
inputs -- no network, no LLM, no fixture-parsing round trip required except
where a test specifically wants real extracted layout signals.
"""

from pathlib import Path

from app.services.extraction.layout import layout_signals
from app.services.extraction.text_extract import extract_text
from app.services.scoring.formatting import format_score

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

CLEAN_RESUME_TEXT = """Aarav Mehta
Backend Developer
aarav.mehta.dev@gmail.com | +91 98250 41172 | Ahmedabad, Gujarat

SUMMARY
Backend developer with experience building REST APIs in Python and Django.

SKILLS
Python, Django, PostgreSQL, Docker, AWS, Git

EXPERIENCE
Backend Development Intern
Zenlabs Technologies, Ahmedabad
Jan 2025 - Present
- Built and documented REST API endpoints in Django REST Framework.
- Optimised slow PostgreSQL queries with composite indexes.
- Containerised the local development environment with Docker.

EDUCATION
B.E. Computer Engineering
Gujarat Technological University
2022 - 2026
"""

CLEAN_RESUME_LAYOUT = {
    "has_images": False,
    "has_tables": False,
    "is_multicolumn": False,
    "page_count": 1,
    "font_families": ["Helvetica"],
    "text_in_header_footer": False,
}

CLEAN_RESUME_SECTIONS = ["summary", "skills", "experience", "education"]


def test_clean_resume_scores_perfectly():
    result = format_score(CLEAN_RESUME_TEXT, CLEAN_RESUME_LAYOUT, CLEAN_RESUME_SECTIONS)

    assert result["issues"] == []
    assert result["points_earned"] == result["points_possible"] == 100
    assert result["score"] == 1.0


def test_tables_and_two_column_resume_loses_exactly_thirty_two_points():
    """demo_before.pdf has a genuine ruled skills grid (survives the
    >=2 rows AND >=2 columns table filter) and a genuine two-column
    layout, so no_tables (16) and single_column (16) should both fail --
    and, with everything else about the fixture already fine (real email,
    real +91 phone, plenty of text, several dates), nothing else should.
    """
    file_bytes = (FIXTURES_DIR / "demo_before.pdf").read_bytes()
    extracted = extract_text(file_bytes, "demo_before.pdf")
    layout = layout_signals(file_bytes)

    assert layout["has_tables"] is True
    assert layout["is_multicolumn"] is True

    # Supplied directly rather than via the LLM parser: this test is about
    # formatting.py's own scoring logic, not resume section-heading
    # phrasing, and the fixture's informal headings ("WHAT I KNOW",
    # "WHERE I WORKED") are a separate, unrelated failure mode.
    sections_found = ["experience", "education", "skills"]

    result = format_score(extracted["text"], layout, sections_found)

    assert result["points_possible"] == 100
    assert result["points_possible"] - result["points_earned"] == 32

    failed_checks = {issue["check"] for issue in result["issues"]}
    assert failed_checks == {"no_tables", "single_column"}
    for issue in result["issues"]:
        assert issue["penalty"] == 16

    penalties = [issue["penalty"] for issue in result["issues"]]
    assert penalties == sorted(penalties, reverse=True)


def test_empty_text_fails_text_extractable():
    result = format_score("", CLEAN_RESUME_LAYOUT, [])

    failed_checks = {issue["check"] for issue in result["issues"]}
    assert "text_extractable" in failed_checks

    text_extractable_issue = next(
        issue for issue in result["issues"] if issue["check"] == "text_extractable"
    )
    assert text_extractable_issue["penalty"] == 15
    assert result["score"] < 1.0


def test_docx_input_skips_reasonable_length_instead_of_failing_it():
    """page_count is None for DOCX (text_extract.py never computes one for
    DOCX), which reasonable_length has no way to check. It must be
    excluded from scoring rather than penalised -- dropped from
    points_possible and reported as skipped, not as a failed issue.
    """
    docx_layout = {**CLEAN_RESUME_LAYOUT, "page_count": None}

    result = format_score(CLEAN_RESUME_TEXT, docx_layout, CLEAN_RESUME_SECTIONS)

    assert result["points_possible"] == 92
    assert result["points_earned"] == 92
    assert result["score"] == 1.0

    failed_checks = {issue["check"] for issue in result["issues"]}
    assert "reasonable_length" not in failed_checks

    assert result["skipped"] == [{"check": "reasonable_length", "status": "not_applicable"}]
