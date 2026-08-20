"""Experience alignment: years of experience and education level against
what a role's market typically expects.

Pure Python, no LLM, no network. resume_months comes from
compute_experience_months (llm_parse.py) -- date arithmetic the LLM never
does. required_years_min comes from a role profile's
median_experience_years (see miner.py).
"""

import re

YEARS_WEIGHT = 0.7
EDUCATION_WEIGHT = 0.3
YEARS_FLOOR = 0.2  # a fresher isn't zeroed out entirely against a mid-level market median

# Ranked highest to lowest. Checked in this order so a resume mentioning
# multiple credentials (e.g. a Master's built on a Bachelor's) is scored
# on its highest one.
_DEGREE_RANK_KEYWORDS: list[tuple[int, tuple[str, ...]]] = [
    (5, ("phd", "doctorate")),
    (4, ("master", "mtech", "msc", "mca", "mba")),
    (3, ("bachelor", "btech", "be", "bsc", "bca")),
    (2, ("diploma",)),
]
HIGH_SCHOOL_RANK = 1
BASELINE_DEGREE_RANK = 3  # bachelor's assumed baseline requirement for these roles

_DEGREE_LABELS = {
    5: "PhD/Doctorate",
    4: "Master's",
    3: "Bachelor's",
    2: "Diploma",
    1: "High school",
}

_PERIOD_RE = re.compile(r"\.")


def _normalize_degree_text(degree: str) -> str:
    # Strip periods so "B.E." / "M.Tech." match the same as "BE" / "MTech".
    return _PERIOD_RE.sub("", degree.lower())


def _degree_rank(degree: str) -> int | None:
    """Rank for one degree string, or None if nothing recognisable is in
    it. Word-boundary matched, so this is only as safe as resume_education
    strings actually being short, structured degree names rather than free
    text -- a stray "be" inside an unrelated sentence would false-positive
    as Bachelor's, same class of risk as any short-abbreviation match.
    """
    normalized = _normalize_degree_text(degree)

    # Checked first: "High School Diploma" contains both "high school" and
    # "diploma". "high school" is the more specific phrase when both
    # appear together, and should rank as high-school-level (1), not
    # diploma-level (2).
    if re.search(r"\bhigh school\b", normalized):
        return HIGH_SCHOOL_RANK

    for rank, keywords in _DEGREE_RANK_KEYWORDS:
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", normalized):
                return rank

    return None


def _highest_degree_rank(resume_education: list[str]) -> int | None:
    ranks = [rank for degree in resume_education if degree for rank in [_degree_rank(degree)] if rank is not None]
    return max(ranks) if ranks else None


def _years_component(resume_years: float, required_years_min: float | None) -> float:
    if not required_years_min:  # None or 0
        return 1.0
    if resume_years >= required_years_min:
        return 1.0
    return max(YEARS_FLOOR, resume_years / required_years_min)


def _education_component(highest_degree_rank: int | None) -> float:
    if highest_degree_rank is not None and highest_degree_rank >= BASELINE_DEGREE_RANK:
        return 1.0
    if highest_degree_rank is not None and highest_degree_rank == BASELINE_DEGREE_RANK - 1:
        return 0.6
    return 0.3


def _years_explanation(resume_years: float, required_years_min: float | None) -> str:
    if not required_years_min:
        return f"{resume_years} years of experience (no market minimum available for this role)."
    if resume_years >= required_years_min:
        return f"{resume_years} years vs {required_years_min} typically required — meets or exceeds the market bar."
    return f"{resume_years} years vs {required_years_min} typically required."


def _education_explanation(highest_degree_rank: int | None) -> str:
    need_label = _DEGREE_LABELS[BASELINE_DEGREE_RANK]
    if highest_degree_rank is None:
        return f"No recognisable degree found on the resume — assumed below the {need_label} baseline."

    have_label = _DEGREE_LABELS[highest_degree_rank]
    if highest_degree_rank >= BASELINE_DEGREE_RANK:
        return f"{have_label} meets the assumed {need_label} baseline for this role."
    if highest_degree_rank == BASELINE_DEGREE_RANK - 1:
        return f"{have_label} is one level below the assumed {need_label} baseline."
    return f"{have_label} is well below the assumed {need_label} baseline."


def experience_score(
    resume_months: int,
    required_years_min: float | None,
    resume_education: list[str],
) -> dict:
    """Experience alignment: 0.7 * years_component + 0.3 * education_component.

    years_component floors at YEARS_FLOOR rather than hitting 0, so a
    fresher scored against a mid-level market median (e.g. Aarav Mehta's 6
    months of experience against a 3.5-year backend-developer median in
    India) doesn't get zeroed out entirely -- that floor is intended
    behaviour, not a bug, and years_explanation spells out why in plain
    language rather than leaving a bare 20 on screen.

    resume_education is a list of degree strings (e.g. ["B.E. Computer
    Engineering"]), not full Education records -- only the degree name is
    ranked.
    """
    resume_years = round(resume_months / 12, 2)
    highest_degree_rank = _highest_degree_rank(resume_education)

    years_component = _years_component(resume_years, required_years_min)
    education_component = _education_component(highest_degree_rank)

    score = round(YEARS_WEIGHT * years_component + EDUCATION_WEIGHT * education_component, 4)

    return {
        "score": score,
        "years_component": round(years_component, 4),
        "education_component": round(education_component, 4),
        "resume_years": resume_years,
        "required_years_min": required_years_min,
        "highest_degree_rank": highest_degree_rank,
        "years_explanation": _years_explanation(resume_years, required_years_min),
        "education_explanation": _education_explanation(highest_degree_rank),
    }
