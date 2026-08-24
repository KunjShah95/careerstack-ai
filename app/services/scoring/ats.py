"""Combine the four scoring components into one ATS score and a ranked
action plan.

WEIGHTS is defined once, here. Nothing else may hardcode these numbers --
every other scoring module returns its own 0-1 score and stays ignorant of
how it gets combined with the others.
"""

from app.models.resume import ParsedResume
from app.services.scoring.experience import experience_score
from app.services.scoring.formatting import CHECKS as FORMAT_CHECKS
from app.services.scoring.formatting import format_score
from app.services.scoring.keywords import keyword_score
from app.services.scoring.semantic import semantic_score

WEIGHTS = {"keyword": 0.40, "semantic": 0.25, "format": 0.20, "experience": 0.15}

BAND_THRESHOLDS = [(80, "Strong match"), (60, "Competitive"), (40, "Needs work")]
DEFAULT_BAND = "Poor match"

TOP_ACTION_ITEMS = 6
WEAK_EVIDENCE_FLAT_GAIN = 2.0

# Points per formatting.py check, read from its own CHECKS table rather
# than re-hardcoded here -- one source of truth for what each check is
# worth. Used to decompose format_detail back into named sub-scores below.
_FORMAT_CHECK_POINTS = {name: points for name, points, _, _ in FORMAT_CHECKS}

# The three format-decomposed dimensions must partition formatting.py's
# nine checks exactly -- see test_ats.py's assertion that their earned
# points sum back to format_detail["points_earned"].
_FORMAT_PARSING_CHECKS = ["text_extractable", "no_tables", "single_column", "reasonable_length", "standard_bullets"]
_SECTION_STRUCTURE_CHECKS = ["has_core_sections", "has_dates"]
_CONTACT_CHECKS = ["has_email", "has_phone"]

DIMENSION_STATUS_THRESHOLDS = [(80, "good"), (60, "fair")]
DEFAULT_DIMENSION_STATUS = "weak"


def _band(overall_score: float) -> str:
    for threshold, label in BAND_THRESHOLDS:
        if overall_score >= threshold:
            return label
    return DEFAULT_BAND


def _dimension_status(score: float) -> str:
    for threshold, label in DIMENSION_STATUS_THRESHOLDS:
        if score >= threshold:
            return label
    return DEFAULT_DIMENSION_STATUS


def _format_group(format_detail: dict, check_names: list[str]) -> dict:
    """Score, and pass/fail sets, for one named group of format_detail's
    checks. A check in format_detail["skipped"] (not applicable -- e.g.
    reasonable_length for a DOCX with no page count) is excluded from both
    earned and possible here, the same way format_score() excludes it from
    the parent score -- so the three groups' points still add up exactly.
    """
    failed = {issue["check"] for issue in format_detail["issues"]}
    not_applicable = {item["check"] for item in format_detail["skipped"]}

    earned = 0
    possible = 0
    passed, failed_in_group = set(), set()

    for name in check_names:
        if name in not_applicable:
            continue
        points = _FORMAT_CHECK_POINTS[name]
        possible += points
        if name in failed:
            failed_in_group.add(name)
        else:
            earned += points
            passed.add(name)

    score = round(100 * earned / possible, 2) if possible else 0.0
    return {"score": score, "earned": earned, "possible": possible, "passed": passed, "failed": failed_in_group}


def _format_parsing_note(failed: set) -> str:
    if not failed:
        return "Clean single-column layout, fully machine readable."
    if "no_tables" in failed and "single_column" in failed:
        return "Two-column layout with tables; parsers may scramble reading order."
    if "text_extractable" in failed:
        return "Very little text could be extracted — this may read as blank to a parser."
    if "no_tables" in failed:
        return "Tables detected — cell contents may be scrambled or dropped by a parser."
    if "single_column" in failed:
        return "Two-column layout — a parser may interleave the columns while reading."
    if "reasonable_length" in failed:
        return "Resume length falls outside the typical 1-2 page range."
    if "standard_bullets" in failed:
        return "Non-standard bullet glyphs found — some may render as garbled symbols."
    return "Some formatting checks failed — see the format section for details."


def _section_structure_note(failed: set) -> str:
    if not failed:
        return "Core sections and employment dates are both present and labelled."
    if "has_core_sections" in failed and "has_dates" in failed:
        return "Missing standard section headings and clear employment dates."
    if "has_core_sections" in failed:
        return "Missing a standard experience, education, or skills heading."
    if "has_dates" in failed:
        return "Few or no employment dates found — experience duration may be undercounted."
    return "Some structure checks failed."


def _contact_note(failed: set) -> str:
    if not failed:
        return "Email and phone both present and detectable."
    if "has_email" in failed and "has_phone" in failed:
        return "No email or phone number could be detected."
    if "has_email" in failed:
        return "No email address could be detected."
    if "has_phone" in failed:
        return "No phone number could be detected."
    return "Some contact details are missing."


def _keyword_note(score: float) -> str:
    if score >= 60:
        return "Covers most in-demand skills for this role."
    if score >= 40:
        return "Covers some in-demand skills, with real gaps against sampled postings."
    return "Missing several skills common in sampled postings."


def _semantic_note(score: float) -> str:
    if score >= 60:
        return "Your resume's wording closely matches what sampled postings ask for."
    if score >= 40:
        return "Some requirements are supported, but the evidence is thin in places."
    return "Resume wording rarely matches the specific requirements sampled postings ask for."


def _experience_note(score: float) -> str:
    if score >= 80:
        return "Experience and education comfortably meet what this market expects."
    if score >= 60:
        return "Experience and education are broadly in line with market expectations."
    if score >= 40:
        return "Experience or education fall short of what this market typically expects."
    return "Experience is well below what this market typically expects for this role."


def _build_dimensions(subscores: dict, format_detail: dict) -> list[dict]:
    """Six named, human-readable dimensions -- no new scoring. Three
    decompose format_detail's existing checks into named groups; three are
    the existing keyword/semantic/experience subscores under display names.
    Never a third total: each dimension stands alone, out of 100.
    """
    parsing = _format_group(format_detail, _FORMAT_PARSING_CHECKS)
    structure = _format_group(format_detail, _SECTION_STRUCTURE_CHECKS)
    contact = _format_group(format_detail, _CONTACT_CHECKS)

    rows = [
        ("Format & ATS Parsing", parsing["score"], _format_parsing_note(parsing["failed"])),
        ("Section Structure", structure["score"], _section_structure_note(structure["failed"])),
        ("Contact & Personal Details", contact["score"], _contact_note(contact["failed"])),
        ("Skills & Keywords", subscores["keyword"], _keyword_note(subscores["keyword"])),
        ("Evidence & Relevance", subscores["semantic"], _semantic_note(subscores["semantic"])),
        ("Experience & Education", subscores["experience"], _experience_note(subscores["experience"])),
    ]

    return [
        {"name": name, "score": score, "status": _dimension_status(score), "note": note}
        for name, score, note in rows
    ]


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
        "dimensions": _build_dimensions(subscores, format_detail),
        "keyword_detail": keyword_detail,
        "semantic_detail": semantic_detail,
        "format_detail": format_detail,
        "experience_detail": experience_detail,
        "role_profile_meta": {
            "postings_sampled": role_profile.get("postings_sampled"),
            "sampled_at": role_profile.get("sampled_at"),
            "role": role_profile.get("role"),
            "location": role_profile.get("location"),
            "sparse_profile": role_profile.get("sparse_profile"),
            # None when cache.py's get_or_mine didn't need to fall back to
            # a broader location; otherwise {"requested_location",
            # "used_location", "requested_postings_sampled"}.
            "location_fallback": role_profile.get("location_fallback"),
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
                # Structured fields behind the "action" sentence, for a UI
                # that wants to show the evidence rather than re-parse it
                # out of prose (postings_sampled itself lives on
                # role_profile_meta, not duplicated here).
                "detail": {"skill": item["skill"], "count": item["count"], "frequency": item["frequency"]},
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
                "detail": {"check": issue["check"], "message": issue["message"], "penalty": issue["penalty"]},
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
                "detail": {"requirement": item["requirement"], "evidence_strength": item["evidence_strength"]},
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
