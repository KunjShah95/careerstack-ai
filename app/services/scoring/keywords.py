"""Frequency-weighted keyword coverage: how well a resume covers the skills
a role profile says the market actually wants, weighted by how often each
skill shows up in real postings for that role.

Deterministic matching only -- no LLM call decides whether a skill is
"present". See CLAUDE.md's one rule: the LLM never produces the score.
"""

import json
import re
import string
from pathlib import Path

from rapidfuzz import fuzz

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_SKILL_ALIASES: dict[str, str] = json.loads(
    (_DATA_DIR / "skill_aliases.json").read_text(encoding="utf-8")
)

# Kept in skill names because they're meaningful there ("c++", "c#",
# "node.js", "ci/cd"); every other punctuation character is noise.
_ALLOWED_PUNCTUATION = "+#./-"
_PUNCTUATION_TO_STRIP = "".join(ch for ch in string.punctuation if ch not in _ALLOWED_PUNCTUATION)
_STRIP_PUNCTUATION_RE = re.compile(f"[{re.escape(_PUNCTUATION_TO_STRIP)}]")
_WHITESPACE_RE = re.compile(r"\s+")

FUZZY_MATCH_THRESHOLD = 90


def normalize_skill(skill: str) -> str:
    """Canonical form of a skill string: lowercase, stripped, punctuation
    removed (except + # . / -), whitespace collapsed, then mapped through
    skill_aliases if it's a known variant spelling ("reactjs" -> "react").
    """
    lowered = skill.strip().lower()
    stripped_punctuation = _STRIP_PUNCTUATION_RE.sub("", lowered)
    collapsed = _WHITESPACE_RE.sub(" ", stripped_punctuation).strip()
    return _SKILL_ALIASES.get(collapsed, collapsed)


def _boundary_pattern(term: str) -> re.Pattern:
    """Word-boundary regex for one (already-normalised) skill string.

    Mirrors miner.py's _boundary_pattern, including the same left-side "."
    exclusion (so scanning for "javascript" doesn't fire on "react.js") --
    kept as a separate copy rather than a shared import since the two
    modules belong to different services and this is a small, stable
    piece of logic. Keep the two in sync if this ever needs another fix.
    """
    escaped = re.escape(term)
    return re.compile(r"(?<![A-Za-z0-9.])" + escaped + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _find_match_tier(
    skill: str, resume_text: str, normalized_resume_skills: set[str]
) -> str | None:
    """Which tier matched `skill` against the resume, cheapest first:

    1. skills_list -- exact match in the normalised resume skill set.
    2. resume_text -- word-boundary regex over the full resume text; catches
       a skill only mentioned inside a project or experience bullet, never
       listed under Skills.
    3. fuzzy -- rapidfuzz ratio >= FUZZY_MATCH_THRESHOLD against any resume
       skill; catches typos ("Djnago").

    Returns the tier name, or None if no tier matched.
    """
    normalized_skill = normalize_skill(skill)

    if normalized_skill in normalized_resume_skills:
        return "skills_list"

    if _boundary_pattern(normalized_skill).search(resume_text):
        return "resume_text"

    for resume_skill in normalized_resume_skills:
        if fuzz.ratio(normalized_skill, resume_skill) >= FUZZY_MATCH_THRESHOLD:
            return "fuzzy"

    return None


def skill_present(skill: str, resume_text: str, resume_skills: set[str]) -> bool:
    """Whether `skill` is present anywhere in the resume: its skills list,
    its body text, or a close-enough fuzzy match. See keyword_score for
    which of the three tiers actually fired.
    """
    normalized_resume_skills = {normalize_skill(s) for s in resume_skills}
    return _find_match_tier(skill, resume_text, normalized_resume_skills) is not None


def keyword_score(
    skill_frequencies: dict[str, dict], resume_text: str, resume_skills: set[str]
) -> dict:
    """Frequency-weighted keyword coverage against a role profile.

    skill_frequencies is role_profile["skill_frequencies"]: canonical skill
    -> {"frequency": float, "count": int} (see miner.py -- count is the raw
    number of postings that mentioned it, out of postings_sampled).

        score = sum(freq_i * matched_i) / sum(freq_i)

    Returns {"score", "matched": [...], "missing": [...]}, both lists
    carrying "count" alongside "frequency" (so the UI can show "3 of 40
    postings" rather than just a percentage) and sorted by frequency
    descending. matched entries also carry "matched_via".
    """
    normalized_resume_skills = {normalize_skill(s) for s in resume_skills}

    matched = []
    missing = []
    weighted_sum = 0.0
    total_frequency = 0.0

    for skill, stats in skill_frequencies.items():
        frequency = stats["frequency"]
        count = stats["count"]
        total_frequency += frequency

        tier = _find_match_tier(skill, resume_text, normalized_resume_skills)
        if tier is not None:
            weighted_sum += frequency
            matched.append(
                {
                    "skill": skill,
                    "frequency": frequency,
                    "count": count,
                    "matched_via": tier,
                }
            )
        else:
            missing.append({"skill": skill, "frequency": frequency, "count": count})

    matched.sort(key=lambda item: item["frequency"], reverse=True)
    missing.sort(key=lambda item: item["frequency"], reverse=True)

    score = round(weighted_sum / total_frequency, 4) if total_frequency else 0.0

    return {"score": score, "matched": matched, "missing": missing}
