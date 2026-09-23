"""Tests for skills_vocab.json's technical/professional categorisation and
miner.py's skill_category() lookup -- see ats.py's by_category breakdown,
which depends on every canonical skill having exactly one category.
"""

from app.services.roleprofile.miner import SKILL_CATEGORIES, skill_category

EXPECTED_PROFESSIONAL = {
    "adaptability",
    "agile",
    "analytical skills",
    "attention to detail",
    "communication skills",
    "conflict resolution",
    "critical thinking",
    "cross-functional collaboration",
    "decision making",
    "kanban",
    "leadership",
    "mentoring",
    "problem solving",
    "project management",
    "scrum",
    "stakeholder management",
    "team collaboration",
    "time management",
}


def test_every_vocab_skill_has_exactly_one_of_the_two_categories():
    assert set(SKILL_CATEGORIES.values()) == {"technical", "professional"}


def test_professional_skills_match_the_expected_set():
    professional = {skill for skill, category in SKILL_CATEGORIES.items() if category == "professional"}
    assert professional == EXPECTED_PROFESSIONAL


def test_tools_and_methodologies_are_classified_as_the_user_specified():
    """Explicit judgement calls from the spec: agile/scrum are
    methodologies, not technologies, so they're professional; git and jira
    are tools, so they're technical.
    """
    assert skill_category("agile") == "professional"
    assert skill_category("scrum") == "professional"
    assert skill_category("git") == "technical"
    assert skill_category("jira") == "technical"


def test_unknown_skill_defaults_to_technical():
    assert skill_category("some-made-up-skill-xyz") == "technical"
