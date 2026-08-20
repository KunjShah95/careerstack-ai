"""Tests for the pure-Python date arithmetic in llm_parse.py.

The LLM never computes these numbers -- see compute_experience_months.
"""

from datetime import date

from app.models.resume import Experience, ParsedResume
from app.services.extraction.llm_parse import (
    _merge_intervals,
    compute_experience_months,
)


def _resume_with(*experiences: Experience) -> ParsedResume:
    return ParsedResume(experience=list(experiences))


def test_month_precision_is_inclusive_of_both_endpoints():
    resume = _resume_with(
        Experience(title="A", start_date="Jan 2025", end_date="Jun 2025")
    )
    assert compute_experience_months(resume) == 6


def test_single_month_role_counts_as_one_month():
    resume = _resume_with(
        Experience(title="A", start_date="Jan 2025", end_date="Jan 2025")
    )
    assert compute_experience_months(resume) == 1


def test_year_boundary_role_is_thirteen_months():
    resume = _resume_with(
        Experience(title="A", start_date="May 2014", end_date="May 2015")
    )
    assert compute_experience_months(resume) == 13


def test_nested_interval_is_not_double_counted():
    """B sits entirely inside A's date range (a second, concurrent role);
    it must not add its own months on top of A's.

    A alone: 2020-01 -> 2021-01 inclusive = 13 months.
    C alone: 2021-06 -> 2022-06 inclusive = 13 months.
    B is fully nested inside A and contributes nothing extra.
    A naive (double-counted) sum would be 13 + 4 + 13 = 30.
    """
    resume = _resume_with(
        Experience(title="A", start_date="2020-01", end_date="2021-01"),
        Experience(title="B", start_date="2020-06", end_date="2020-09"),
        Experience(title="C", start_date="2021-06", end_date="2022-06"),
    )
    assert compute_experience_months(resume) == 26


def test_merge_intervals_absorbs_a_nested_interval():
    a = (date(2020, 1, 1), date(2021, 1, 31))
    b = (date(2020, 6, 1), date(2020, 9, 30))  # nested inside a
    c = (date(2021, 6, 1), date(2022, 6, 30))  # separate

    assert _merge_intervals([a, b, c]) == [a, c]
