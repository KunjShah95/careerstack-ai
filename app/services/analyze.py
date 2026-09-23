"""End-to-end resume analysis pipeline: extract -> layout -> parse ->
mine/cache role profile -> score -> action plan -> save.

This is the one place that wires the whole pipeline together. app/main.py
calls run_analysis() and only run_analysis() -- see CLAUDE.md's "routers
validate input, call a service, return a response; zero logic in main.py"
rule.
"""

import logging
import uuid
from datetime import datetime, timezone

from app.services.extraction.layout import get_layout_signals
from app.services.extraction.parse_cache import get_or_parse, resume_file_key
from app.services.extraction.text_extract import extract_text
from app.services.roleprofile.cache import get_most_recent_cached_profile, get_or_mine
from app.services.scoring.ats import build_action_plan, compute_ats_score
from app.store import list_keys, load_json, save_json

logger = logging.getLogger(__name__)

ANALYSES_COLLECTION = "analyses"

NEEDS_OCR_MESSAGE = (
    "We couldn't find any selectable text in this file. Most ATS systems "
    "can't read it either — export a text-based PDF and try again."
)


class NeedsOcrError(Exception):
    """Raised when the uploaded file has no selectable text. main.py maps
    this to a 422.
    """


class RoleProfileUnavailableError(Exception):
    """Raised when role profile data can't be obtained for the requested
    role/location, AND no cached profile exists for any other role or
    location either -- there is genuinely nothing to fall back to.
    main.py maps this to a 503.
    """


def _degraded_message(requested_role: str, requested_location: str, fallback_profile: dict) -> str:
    return (
        f"Live market data for '{requested_role}' in '{requested_location}' wasn't "
        f"available, so this analysis uses the most recently cached market data instead "
        f"('{fallback_profile.get('role')}' in '{fallback_profile.get('location')}', "
        f"sampled {fallback_profile.get('sampled_at')}). Scores may not reflect your "
        f"specific role or location."
    )


async def run_analysis(file_bytes: bytes, filename: str, role: str, location: str, user_id: str) -> dict:
    """Run the full pipeline and return the saved analysis dict.

    user_id is stamped onto the saved record -- analyses belong to the
    user who created them (see main.py's protected /api/analyze and
    /api/analyze/{id}, and list_user_analyses below). Required, not
    optional: every analysis has an owner now.

    Can raise: ExtractionError (text_extract.py / llm_parse.py -- bad or
    unparseable file), NeedsOcrError (no selectable text), RuntimeError
    (get_or_parse, only in DEMO_MODE with no cached parse for this exact
    file), RoleProfileUnavailableError (role profile mining failed and no
    cached profile exists for any role/location to fall back to either).
    main.py is responsible for turning each of these into a plain-language
    HTTP response.
    """
    extracted = extract_text(file_bytes, filename)

    if extracted["needs_ocr"]:
        raise NeedsOcrError(NEEDS_OCR_MESSAGE)

    resume_text = extracted["text"]
    layout = get_layout_signals(file_bytes, filename)

    # Both of these are slow (LLM call, and a cold role-profile mine can
    # take 15-25s) and both run synchronously/blocking here -- no
    # background tasks for this step, per the build order. Acceptable for
    # a demo-scale tool; a production version would want these off the
    # request thread.
    resume = get_or_parse(file_bytes, resume_text)

    degraded_message = None
    try:
        role_profile = await get_or_mine(role, location)
    except Exception as exc:
        # Deliberately broad: "if role profile mining fails entirely" is a
        # live-demo safety net, not a specific-exception-type contract --
        # a RuntimeError (DEMO_MODE with nothing cached), a live Adzuna
        # failure, anything. Never let this be the reason a demo 500s.
        logger.warning(
            "Role profile unavailable for role=%r location=%r (%s) -- "
            "trying the most recent cached profile for any role instead",
            role,
            location,
            exc,
        )
        fallback_profile = get_most_recent_cached_profile()
        if fallback_profile is None:
            raise RoleProfileUnavailableError(
                "No role market data is cached yet, for this role/location or any "
                "other. Run scripts/prep_demo.py, or mine a profile with DEMO_MODE "
                "off, before analysing a resume."
            ) from exc

        role_profile = fallback_profile
        degraded_message = _degraded_message(role, location, fallback_profile)

    ats_result = compute_ats_score(resume, resume_text, layout, role_profile)
    action_plan = build_action_plan(ats_result)

    analysis_id = uuid.uuid4().hex
    analysis = {
        "analysis_id": analysis_id,
        "user_id": user_id,
        "filename": filename,
        "role": role,
        "location": location,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extraction_meta": {
            "method": extracted["method"],
            "page_count": extracted["page_count"],
            "char_count": extracted["char_count"],
        },
        # The parse cache's own key for this file. An analysis record
        # stores scoring output, not the resume itself, so without this
        # there's no way back from an analysis_id to the resume's skills --
        # which discovery.py needs to score individual postings. Records
        # written before this field existed simply don't have it; callers
        # degrade rather than fail (see discovery.py's _resume_for_analysis).
        "resume_file_key": resume_file_key(file_bytes),
        "degraded": degraded_message is not None,
        "degraded_message": degraded_message,
        **ats_result,
        # build_action_plan returns "items", not "actions" -- kept as-is
        # here rather than renamed, so the API response matches what the
        # function actually produces.
        "action_plan": action_plan,
    }

    save_json(ANALYSES_COLLECTION, analysis_id, analysis)
    return analysis


SUMMARY_FIELDS = ("analysis_id", "filename", "role", "location", "overall_score", "band", "created_at")


def list_user_analyses(user_id: str) -> list[dict]:
    """Summary rows for every analysis this user has run, newest first.

    Deliberately projection-only (SUMMARY_FIELDS, not the full record) --
    a history list should stay cheap to render; the full payload is still
    available one at a time via GET /api/analyze/{id}.

    Also carries "parse_score" alongside SUMMARY_FIELDS -- not in the
    originally-specified field list, but the history UI shows both the
    ATS Parse Score and the Role Fit Score per row ("the two scores as
    small pills"), and overall_score alone can't produce that. No new
    computation: format_detail.score is already stored on every analysis
    record, just rescaled from its 0-1 fraction to the 0-100 scale
    overall_score/subscores already use.
    """
    summaries = []
    for key in list_keys(ANALYSES_COLLECTION):
        record = load_json(ANALYSES_COLLECTION, key)
        if not record or record.get("user_id") != user_id:
            continue

        summary = {field: record.get(field) for field in SUMMARY_FIELDS}
        format_score_fraction = (record.get("format_detail") or {}).get("score")
        summary["parse_score"] = (
            round(format_score_fraction * 100, 2) if isinstance(format_score_fraction, (int, float)) else None
        )
        summaries.append(summary)

    summaries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return summaries
