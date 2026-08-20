"""Tests for experience alignment scoring in experience.py."""

from app.services.scoring.experience import _degree_rank, experience_score


def test_aarav_mehta_hits_the_years_floor_with_a_plain_language_explanation():
    """6 months against a 3.5-year market median is intended to hit the
    0.2 floor, not score 0 -- and the dict must explain why in plain
    language, not just hand back a bare number.
    """
    result = experience_score(
        resume_months=6, required_years_min=3.5, resume_education=["B.E. Computer Engineering"]
    )

    assert result["resume_years"] == 0.5
    assert result["years_component"] == 0.2
    assert result["education_component"] == 1.0
    assert result["score"] == 0.44  # 0.7*0.2 + 0.3*1.0
    assert result["years_explanation"] == "0.5 years vs 3.5 typically required."
    assert "Bachelor" in result["education_explanation"]


def test_years_meeting_or_exceeding_requirement_scores_full_marks():
    result = experience_score(resume_months=60, required_years_min=3.5, resume_education=["B.Tech"])
    assert result["years_component"] == 1.0
    assert "meets or exceeds" in result["years_explanation"]


def test_no_market_minimum_scores_years_as_full_marks():
    for required in (None, 0, 0.0):
        result = experience_score(resume_months=6, required_years_min=required, resume_education=[])
        assert result["years_component"] == 1.0
        assert "no market minimum" in result["years_explanation"]


def test_years_component_never_drops_below_the_floor():
    """A near-zero months count against a real market median should still
    floor at 0.2, not approach 0.
    """
    result = experience_score(resume_months=1, required_years_min=10.0, resume_education=[])
    assert result["years_component"] == 0.2


def test_degree_ranking():
    assert _degree_rank("PhD in Computer Science") == 5
    assert _degree_rank("Doctorate in Physics") == 5
    assert _degree_rank("M.Tech in AI") == 4
    assert _degree_rank("MBA") == 4
    assert _degree_rank("B.E. Computer Engineering") == 3
    assert _degree_rank("Bachelor of Science") == 3
    assert _degree_rank("Diploma in Mechanical Engineering") == 2
    assert _degree_rank("High School") == 1
    assert _degree_rank("Certificate in Photography") is None


def test_high_school_diploma_ranks_as_high_school_not_diploma():
    """"High School Diploma" contains both "high school" and "diploma" --
    it must resolve to the high-school rank (1), not the diploma rank (2),
    since that's what the credential actually is.
    """
    assert _degree_rank("High School Diploma") == 1


def test_education_component_one_level_below_baseline():
    # Diploma (2) is one level below the Bachelor's baseline (3) -> 0.6
    result = experience_score(
        resume_months=60, required_years_min=None, resume_education=["Diploma in IT"]
    )
    assert result["education_component"] == 0.6
    assert "one level below" in result["education_explanation"]


def test_education_component_well_below_baseline():
    # High school (1) is two levels below the Bachelor's baseline (3) -> 0.3
    result = experience_score(
        resume_months=60, required_years_min=None, resume_education=["High School"]
    )
    assert result["education_component"] == 0.3
    assert "well below" in result["education_explanation"]


def test_education_component_no_recognisable_degree():
    result = experience_score(resume_months=60, required_years_min=None, resume_education=[])
    assert result["education_component"] == 0.3
    assert result["highest_degree_rank"] is None
    assert "No recognisable degree" in result["education_explanation"]


def test_highest_degree_wins_when_multiple_are_listed():
    result = experience_score(
        resume_months=60,
        required_years_min=None,
        resume_education=["Bachelor of Science", "M.Tech in AI"],
    )
    assert result["highest_degree_rank"] == 4
    assert result["education_component"] == 1.0
