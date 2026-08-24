"""Tests for the combined ATS score and action plan in ats.py.

The ParsedResume and role_profile below are hand-constructed rather than
produced via parse_resume() (Groq) or mine_role_profile() (Adzuna): the
point of these tests is ats.py's own arithmetic, not the LLM or the live
job market, and a hand-built fixture keeps that arithmetic deterministic
and network-free. resume_text and layout are the real, locally-extracted
demo_before.pdf output (extract_text/layout_signals are pure local PDF
processing -- no network, no LLM), so format_score and semantic_score are
still exercised against genuine data.
"""

from pathlib import Path

from app.models.resume import ContactInfo, Education, Experience, ParsedResume
from app.services.extraction.layout import layout_signals
from app.services.extraction.text_extract import extract_text
from app.services.scoring.ats import (
    WEIGHTS,
    _CONTACT_CHECKS,
    _FORMAT_PARSING_CHECKS,
    _SECTION_STRUCTURE_CHECKS,
    _format_group,
    build_action_plan,
    compute_ats_score,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _demo_before_resume(skills: list[str]) -> ParsedResume:
    """A hand-built ParsedResume matching demo_before.pdf's real content
    (Aarav Mehta, confirmed by inspecting extract_text's output for this
    fixture in an earlier step) -- confirmed to mention neither Django,
    REST API, nor Docker anywhere in its text.
    """
    return ParsedResume(
        contact=ContactInfo(
            name="Aarav Mehta",
            email="aarav.mehta.dev@gmail.com",
            phone="+91 98250 41172",
            location="Ahmedabad, Gujarat",
        ),
        summary="Final year computer engineering student interested in server side development.",
        skills=skills,
        experience=[
            Experience(
                title="Software Development Intern",
                company="Zenlabs Technologies",
                location="Ahmedabad",
                start_date="2025-01",
                end_date="2025-06",
                is_current=False,
                bullets=[
                    "Worked on backend APIs for the internal dashboard.",
                    "Helped the team with database related work and queries.",
                ],
            )
        ],
        education=[
            Education(
                degree="B.E. Computer Engineering",
                field="Computer Engineering",
                institution="Gujarat Technological University",
                start_date="2022",
                end_date="2026",
            )
        ],
        projects=[],
        certifications=[],
        sections_found=["experience", "education", "skills"],
        total_experience_months=6,
    )


ROLE_PROFILE = {
    "role": "backend developer",
    "location": "India",
    "postings_sampled": 40,
    "sampled_at": "2026-08-01T00:00:00+00:00",
    "skill_frequencies": {
        "python": {"frequency": 0.55, "count": 22},  # matched from the start
        "mysql": {"frequency": 0.20, "count": 8},  # matched from the start
        "django": {"frequency": 0.45, "count": 18},  # missing -- highest freq among missing
        "rest api": {"frequency": 0.30, "count": 12},  # missing
        "docker": {"frequency": 0.15, "count": 6},  # missing
    },
    "requirement_sentences": [
        "Hands-on experience building applications with Python",
        "Working knowledge of MySQL in a production environment",
        "Proficiency in Django for day-to-day development work",
        "Demonstrated ability to use REST API to solve real engineering problems",
        "Comfortable working with Docker as part of a development team",
    ],
    "median_experience_years": 3.5,
    "source_ids": [],
}


def _score_demo_before(skills: list[str]) -> dict:
    file_bytes = (FIXTURES_DIR / "demo_before.pdf").read_bytes()
    resume_text = extract_text(file_bytes, "demo_before.pdf")["text"]
    layout = layout_signals(file_bytes)
    resume = _demo_before_resume(skills)
    return compute_ats_score(resume, resume_text, layout, ROLE_PROFILE)


def test_weights_sum_to_one():
    assert round(sum(WEIGHTS.values()), 10) == 1.0


def test_compute_ats_score_structure_and_band():
    result = _score_demo_before(["Python", "MySQL", "HTML", "CSS"])

    assert 0.0 <= result["overall_score"] <= 100.0
    assert result["band"] in {"Strong match", "Competitive", "Needs work", "Poor match"}
    assert set(result["subscores"].keys()) == {"keyword", "semantic", "format", "experience"}
    assert result["role_profile_meta"]["role"] == "backend developer"
    assert result["role_profile_meta"]["postings_sampled"] == 40


def test_action_plan_top_missing_skill_gain_matches_actual_rescored_delta():
    """The number an examiner is most likely to probe: if you follow the
    top missing-skill recommendation, the actual overall_score delta after
    rescoring must match its predicted estimated_gain. If they don't
    match, the gain formula in ats.py is wrong and needs fixing -- not
    this test.
    """
    before_skills = ["Python", "MySQL", "HTML", "CSS"]
    before_result = _score_demo_before(before_skills)
    plan = build_action_plan(before_result)

    assert len(plan["items"]) > 0
    top_item = plan["items"][0]
    assert top_item["type"] == "missing_skill"
    assert "django" in top_item["action"].lower()  # highest-frequency missing skill

    predicted_gain = top_item["estimated_gain"]

    after_skills = before_skills + ["Django"]
    after_result = _score_demo_before(after_skills)

    actual_delta = after_result["overall_score"] - before_result["overall_score"]

    assert abs(actual_delta - predicted_gain) < 1.0

    # only the keyword component should have moved
    assert after_result["subscores"]["format"] == before_result["subscores"]["format"]
    assert after_result["subscores"]["semantic"] == before_result["subscores"]["semantic"]
    assert after_result["subscores"]["experience"] == before_result["subscores"]["experience"]
    assert after_result["subscores"]["keyword"] > before_result["subscores"]["keyword"]


def test_action_plan_capped_at_six_items_sorted_by_gain_descending():
    result = _score_demo_before(["Python", "MySQL", "HTML", "CSS"])
    plan = build_action_plan(result)

    assert len(plan["items"]) <= 6
    gains = [item["estimated_gain"] for item in plan["items"]]
    assert gains == sorted(gains, reverse=True)
    assert [item["rank"] for item in plan["items"]] == list(range(1, len(plan["items"]) + 1))


def test_projected_score_is_capped_at_one_hundred():
    result = _score_demo_before(["Python", "MySQL", "HTML", "CSS"])
    # Force an unrealistically high overall_score to exercise the cap.
    result["overall_score"] = 99.9
    plan = build_action_plan(result)
    assert plan["projected_score_under_our_model"] <= 100.0


def test_dimensions_have_six_named_entries_with_expected_shape():
    result = _score_demo_before(["Python", "MySQL", "HTML", "CSS"])
    dimensions = result["dimensions"]

    assert [d["name"] for d in dimensions] == [
        "Format & ATS Parsing",
        "Section Structure",
        "Contact & Personal Details",
        "Skills & Keywords",
        "Evidence & Relevance",
        "Experience & Education",
    ]
    for dimension in dimensions:
        assert set(dimension.keys()) == {"name", "score", "status", "note"}
        assert 0.0 <= dimension["score"] <= 100.0
        assert dimension["status"] in {"good", "fair", "weak"}
        assert isinstance(dimension["note"], str) and dimension["note"]


def test_format_dimensions_partition_points_earned_exactly():
    """The three format-decomposed dimensions must never silently drift
    from format_detail's own points_earned -- if a check ever gets added,
    renamed, or miscategorised between the three groups, this catches it
    rather than letting the dashboard quietly show numbers that don't add
    up to the parent format score.
    """
    result = _score_demo_before(["Python", "MySQL", "HTML", "CSS"])
    format_detail = result["format_detail"]

    parsing = _format_group(format_detail, _FORMAT_PARSING_CHECKS)
    structure = _format_group(format_detail, _SECTION_STRUCTURE_CHECKS)
    contact = _format_group(format_detail, _CONTACT_CHECKS)

    total_earned = parsing["earned"] + structure["earned"] + contact["earned"]
    total_possible = parsing["possible"] + structure["possible"] + contact["possible"]

    assert total_earned == format_detail["points_earned"]
    assert total_possible == format_detail["points_possible"]
