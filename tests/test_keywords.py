"""Tests for frequency-weighted keyword coverage in keywords.py.

Run with `-s` to see the tier summary print in
test_keyword_score_matched_via_all_three_tiers -- pytest captures stdout on
a pass by default, so it's silent otherwise.
"""

from app.services.scoring.keywords import keyword_score, normalize_skill, skill_present


def test_normalize_skill_resolves_aliases_and_case():
    assert normalize_skill("React.js") == "react"
    assert normalize_skill("REACTJS") == "react"
    assert normalize_skill("JS") == "javascript"


def test_skill_only_in_a_project_bullet_matches_via_resume_text_tier():
    """Jenkins is mentioned in a project bullet but never listed under
    Skills -- tier 1 (exact match against the skills list) must miss it,
    and tier 2 (a scan of the full resume text) must catch it.
    """
    resume_text = """
    PROJECTS
    Inventory Dashboard
    Automated deployments with Jenkins as part of a CI/CD pipeline running
    on Docker containers.
    """
    resume_skills = {"Python", "Docker", "SQL"}  # Jenkins is NOT listed here

    assert skill_present("jenkins", resume_text, resume_skills) is True

    result = keyword_score({"jenkins": {"frequency": 0.2, "count": 8}}, resume_text, resume_skills)
    assert result["matched"][0]["matched_via"] == "resume_text"
    assert result["matched"][0]["count"] == 8


def test_weighted_coverage_hand_computed_example():
    """Frequencies deliberately don't sum to 1.0, so this can't accidentally
    pass from a denominator bug that silently assumes they do.

    matched: python (0.5) + sql (0.4) = 0.9
    missing: aws (0.3)
    score = 0.9 / (0.5 + 0.4 + 0.3) = 0.9 / 1.2 = 0.75
    """
    skill_frequencies = {
        "python": {"frequency": 0.5, "count": 20},
        "sql": {"frequency": 0.4, "count": 16},
        "aws": {"frequency": 0.3, "count": 12},
    }
    resume_text = "Experienced backend engineer skilled in Python and SQL."
    resume_skills = {"Python", "SQL"}

    result = keyword_score(skill_frequencies, resume_text, resume_skills)

    assert result["score"] == 0.75
    assert [item["skill"] for item in result["matched"]] == ["python", "sql"]
    assert [item["skill"] for item in result["missing"]] == ["aws"]
    # count carried through on both lists, per the role profile's raw mentions
    assert result["matched"][0]["count"] == 20
    assert result["missing"][0]["count"] == 12


def test_keyword_score_matched_via_all_three_tiers():
    """One skill per tier, plus one genuinely missing skill, so a run of
    this test can be eyeballed (with -s) to confirm tier 2 and tier 3
    actually fire rather than everything landing on tier 1.
    """
    resume_text = """
    SKILLS
    Python, SQL, Kubernets

    EXPERIENCE
    Automated deployments with Jenkins as part of a CI/CD pipeline.
    """
    resume_skills = {"Python", "SQL", "Kubernets"}  # "Kubernets" is a typo

    skill_frequencies = {
        "python": {"frequency": 0.6, "count": 24},  # tier 1: skills_list
        "jenkins": {"frequency": 0.3, "count": 12},  # tier 2: resume_text only
        "kubernetes": {"frequency": 0.2, "count": 8},  # tier 3: fuzzy (typo)
        "rust": {"frequency": 0.1, "count": 4},  # missing entirely
    }

    result = keyword_score(skill_frequencies, resume_text, resume_skills)

    tier_summary: dict[str, int] = {}
    for item in result["matched"]:
        tier_summary[item["matched_via"]] = tier_summary.get(item["matched_via"], 0) + 1
    print(f"\ntier summary: {tier_summary}")
    for item in result["matched"]:
        print(f"  matched  {item['skill']:<12} via {item['matched_via']}")
    for item in result["missing"]:
        print(f"  missing  {item['skill']}")

    matched_via = {item["skill"]: item["matched_via"] for item in result["matched"]}
    assert matched_via["python"] == "skills_list"
    assert matched_via["jenkins"] == "resume_text"
    assert matched_via["kubernetes"] == "fuzzy"
    assert "rust" not in matched_via
    assert [item["skill"] for item in result["missing"]] == ["rust"]

    # the actual point of this test: tiers 2 and 3 must both have fired
    assert tier_summary.get("resume_text", 0) >= 1
    assert tier_summary.get("fuzzy", 0) >= 1
