"""Combine the four scoring components into one ATS score and a ranked
action plan.

WEIGHTS is defined once, here. Nothing else may hardcode these numbers --
every other scoring module returns its own 0-1 score and stays ignorant of
how it gets combined with the others.
"""

from app.models.resume import ParsedResume
from app.services.scoring.experience import experience_score
from app.services.scoring.formatting import format_score
from app.services.scoring.keywords import keyword_score
from app.services.scoring.semantic import semantic_score

WEIGHTS = {"keyword": 0.40, "semantic": 0.25, "format": 0.20, "experience": 0.15}

BAND_THRESHOLDS = [(80, "Strong match"), (60, "Competitive"), (40, "Needs work")]
DEFAULT_BAND = "Poor match"

TOP_ACTION_ITEMS = 6
WEAK_EVIDENCE_FLAT_GAIN = 2.0


def _band(overall_score: float) -> str:
    for threshold, label in BAND_THRESHOLDS:
        if overall_score >= threshold:
            return label
    return DEFAULT_BAND


def compute_ats_score(
    resume: ParsedResume, resume_text: str, layout: dict, role_profile: dict
) -> dict:
    """Run all four scoring components and combine them with WEIGHTS.

    format_detail["score"] is already earned/possible, not
    points_earned/100 -- format_score's points_possible varies (e.g. 92
    for a DOCX resume, since page count -- and therefore
    reasonable_length -- can't be checked for DOCX), so its "score" field
    is used directly rather than re-deriving a fraction here.
    """
    skill_frequencies = role_profile.get("skill_frequencies", {})
    # Already frequency-ordered (miner.py sorts skill_frequencies
    # descending before returning it).
    top_skills = list(skill_frequencies.keys())

    keyword_detail = keyword_score(skill_frequencies, resume_text, set(resume.skills))
    semantic_detail = semantic_score(
        resume_text, role_profile.get("requirement_sentences", []), top_skills=top_skills
    )
    format_detail = format_score(resume_text, layout, resume.sections_found)
    experience_detail = experience_score(
        resume.total_experience_months,
        role_profile.get("median_experience_years"),
        [education.degree for education in resume.education if education.degree],
    )

    component_scores = {
        "keyword": keyword_detail["score"],
        "semantic": semantic_detail["score"],
        "format": format_detail["score"],
        "experience": experience_detail["score"],
    }

    overall_score = round(
        100 * sum(WEIGHTS[component] * component_scores[component] for component in WEIGHTS), 2
    )
    subscores = {component: round(value * 100, 2) for component, value in component_scores.items()}

    return {
        "overall_score": overall_score,
        "band": _band(overall_score),
        "weights": dict(WEIGHTS),
        "subscores": subscores,
        "keyword_detail": keyword_detail,
        "semantic_detail": semantic_detail,
        "format_detail": format_detail,
        "experience_detail": experience_detail,
        "role_profile_meta": {
            "postings_sampled": role_profile.get("postings_sampled"),
            "sampled_at": role_profile.get("sampled_at"),
            "role": role_profile.get("role"),
            "location": role_profile.get("location"),
        },
    }


def _missing_skill_actions(ats_result: dict) -> list[dict]:
    """One action per missing skill. estimated_gain is exact, not a rough
    estimate: keyword_score's own formula is
    sum(freq_i * matched_i) / sum(freq_i), so flipping one missing skill
    to matched raises the keyword component by exactly
    freq_i / sum(freq_i), and the overall score by WEIGHTS["keyword"]
    times that, times 100. See test_ats.py for a rescoring check that this
    is actually true, not just algebraically plausible.
    """
    keyword_detail = ats_result["keyword_detail"]
    total_frequency = sum(item["frequency"] for item in keyword_detail["matched"]) + sum(
        item["frequency"] for item in keyword_detail["missing"]
    )
    if total_frequency <= 0:
        return []

    actions = []
    for item in keyword_detail["missing"]:
        gain = 100 * WEIGHTS["keyword"] * item["frequency"] / total_frequency
        actions.append(
            {
                "action": (
                    f'Add "{item["skill"]}" to your resume -- it appears in '
                    f'{item["count"]} of the sampled postings for this role.'
                ),
                "type": "missing_skill",
                "estimated_gain": round(gain, 4),
                # Assumes the skill is genuinely there but unlisted, not
                # something to learn from scratch -- usually true for a
                # resume-completeness fix, so this defaults to low effort.
                "effort": "low",
            }
        )
    return actions


def _format_issue_actions(ats_result: dict) -> list[dict]:
    format_detail = ats_result["format_detail"]
    points_possible = format_detail["points_possible"]
    if not points_possible:
        return []

    actions = []
    for issue in format_detail["issues"]:
        gain = 100 * WEIGHTS["format"] * issue["penalty"] / points_possible
        actions.append(
            {
                "action": issue["message"],
                "type": "format_issue",
                "estimated_gain": round(gain, 4),
                # Structural resume edits (remove a table, fix bullets,
                # add an email) -- quick fixes, so low effort by default.
                "effort": "low",
            }
        )
    return actions


def _weak_evidence_actions(ats_result: dict) -> list[dict]:
    semantic_detail = ats_result["semantic_detail"]
    actions = []
    for item in semantic_detail["weakest_requirements"]:
        actions.append(
            {
                "action": f'Add stronger evidence for: "{item["requirement"]}"',
                "type": "weak_evidence",
                "estimated_gain": WEAK_EVIDENCE_FLAT_GAIN,
                "effort": "medium",
            }
        )
    return actions


def build_action_plan(ats_result: dict) -> dict:
    """Rank findings into a to-do list of at most TOP_ACTION_ITEMS items,
    highest estimated_gain first.

    projected_score_under_our_model is overall_score plus the sum of the
    gains actually shown in the plan, capped at 100. The name is
    deliberate: this is arithmetic on our own weighted formula, not a
    prediction about how a real ATS would respond to these changes.
    """
    candidates = (
        _missing_skill_actions(ats_result)
        + _format_issue_actions(ats_result)
        + _weak_evidence_actions(ats_result)
    )
    candidates.sort(key=lambda item: item["estimated_gain"], reverse=True)

    top_candidates = candidates[:TOP_ACTION_ITEMS]
    items = [{"rank": rank, **item} for rank, item in enumerate(top_candidates, start=1)]

    total_gain = sum(item["estimated_gain"] for item in top_candidates)
    projected_score = min(100.0, ats_result["overall_score"] + total_gain)

    return {
        "items": items,
        "projected_score_under_our_model": round(projected_score, 2),
    }
