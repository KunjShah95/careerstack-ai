"""LLM-based structured extraction from raw resume text.

The LLM only extracts what's already in the text -- names, dates, skills as
written, bullet text verbatim. It never computes a number. Total experience
months is pure Python date arithmetic (compute_experience_months), never
asked of the model.
"""

import calendar
import json
import re
from datetime import date, datetime

from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
from groq import APIError, Groq
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from app.config import settings
from app.models.resume import Experience, ParsedResume
from app.services.extraction.text_extract import ExtractionError

MAX_INPUT_CHARS = 20000
MAX_RETRIES = 3

_PRESENT_TOKENS = {"present", "current", "now"}
_YEAR_ONLY_RE = re.compile(r"^\d{4}$")

_RESUME_SCHEMA_JSON = json.dumps(ParsedResume.model_json_schema(), indent=2)

SYSTEM_PROMPT = f"""You are a strict resume-parsing engine. Given raw resume \
text, extract structured data and output it as a single JSON object matching \
this JSON Schema exactly:

{_RESUME_SCHEMA_JSON}

Rules you must follow exactly:
- Output ONLY a JSON object. No markdown code fences, no commentary, no \
explanation before or after it.
- Never invent information. Use null for any missing scalar field and [] for \
any missing list field.
- Copy skills exactly as written in the resume. Do not expand abbreviations, \
do not normalise casing, do not infer skills that are not explicitly named.
- Keep experience bullets verbatim -- copy the original wording, do not \
summarise or rewrite them.
- sections_found must list only the section headings that actually appear in \
the resume text, lowercased, drawn only from this set: summary, skills, \
experience, education, projects, certifications, awards.
- Always set total_experience_months to 0. It is computed separately and \
your value is discarded.
"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class _InvalidLLMResponse(Exception):
    """Internal: raised to trigger a tenacity retry with a correction turn."""


def _extract_json_text(raw: str) -> str:
    """Strip markdown fences if present, else fall back to regex-extracting
    the outermost {...} span if the response doesn't start with one.
    """
    text = raw.strip()

    fence_match = _FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()

    if text.startswith("{"):
        return text

    object_match = _OBJECT_RE.search(text)
    if object_match:
        return object_match.group(0)

    return text


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type((_InvalidLLMResponse, APIError)),
    reraise=True,
)
def _call_and_validate(client: Groq, messages: list[dict]) -> ParsedResume:
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=messages,
    )
    raw = response.choices[0].message.content or ""
    json_text = _extract_json_text(raw)

    try:
        data = json.loads(json_text)
        return ParsedResume.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        # Give the model a chance to fix its own output: show it what it
        # said, and what was wrong with it, then ask again.
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"That response was invalid: {exc}. Reply again with "
                    "ONLY a corrected JSON object matching the schema above "
                    "-- no markdown fences, no commentary."
                ),
            }
        )
        raise _InvalidLLMResponse(str(exc)) from exc


def parse_resume(text: str) -> ParsedResume:
    """Extract a ParsedResume from raw resume text via the LLM.

    Truncates input to MAX_INPUT_CHARS. Retries up to MAX_RETRIES times on
    invalid JSON, feeding the error back to the model each time. Sets
    total_experience_months from compute_experience_months, not the model.
    """
    truncated = text[:MAX_INPUT_CHARS]
    client = Groq(api_key=settings.groq_api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": truncated},
    ]

    try:
        resume = _call_and_validate(client, messages)
    except _InvalidLLMResponse as exc:
        raise ExtractionError(
            f"Could not parse a valid resume structure from the model after "
            f"{MAX_RETRIES} attempts: {exc}"
        ) from exc
    except APIError as exc:
        raise ExtractionError(f"Resume parsing service is unavailable: {exc}") from exc

    resume.total_experience_months = compute_experience_months(resume)
    return resume


def _is_present(value: str) -> bool:
    return value.strip().lower() in _PRESENT_TOKENS


def _parse_start_date(value: str | None) -> date | None:
    """Parse a resume date string as the START of a period.

    "Present"/"Current"/"Now" (any case) map to today. Month/year precision
    ("2025-01" or "2025") defaults to day 1, which is correct for a start
    date. Returns None if the value is empty or dateutil can't parse it.
    """
    if not value or not value.strip():
        return None

    stripped = value.strip()
    if _is_present(stripped):
        return date.today()

    try:
        parsed = date_parser.parse(stripped, default=datetime(1900, 1, 1))
    except (ValueError, OverflowError, TypeError):
        return None

    return parsed.date()


def _parse_end_date(value: str | None) -> date | None:
    """Parse a resume date string as the END of a period.

    "Present"/"Current"/"Now" (any case) map to today's actual date,
    unsnapped -- today already has a real day, there's no period to be
    inclusive of. A month- or year-precision date snaps to the LAST day of
    that period instead of the first, so the interval counts as inclusive
    of the end month: "Jan 2025" to "Jun 2025" is 6 months, not 5. Returns
    None if the value is empty or dateutil can't parse it.
    """
    if not value or not value.strip():
        return None

    stripped = value.strip()
    if _is_present(stripped):
        return date.today()

    try:
        parsed = date_parser.parse(stripped, default=datetime(1900, 1, 1))
    except (ValueError, OverflowError, TypeError):
        return None

    # A bare "YYYY" defaults to January under our parse default, but as an
    # end date it means the role ran through the whole year -- December.
    month = 12 if _YEAR_ONLY_RE.match(stripped) else parsed.month
    last_day = calendar.monthrange(parsed.year, month)[1]
    return date(parsed.year, month, last_day)


def _experience_interval(experience: Experience) -> tuple[date, date] | None:
    """(start, end) date range for one job, or None if it can't be dated.

    A missing end_date is treated as ongoing (today) only when is_current
    is true; otherwise the role is left out of the total rather than
    guessed at.
    """
    start = _parse_start_date(experience.start_date)
    if start is None:
        return None

    end = _parse_end_date(experience.end_date)
    if end is None:
        if not experience.is_current:
            return None
        end = date.today()

    if end < start:
        return None

    return start, end


def _merge_intervals(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Merge overlapping or touching date intervals.

    Concurrent or back-to-back roles must not be double counted.
    """
    if not intervals:
        return []

    ordered = sorted(intervals, key=lambda interval: interval[0])
    merged = [ordered[0]]

    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def compute_experience_months(resume: ParsedResume) -> int:
    """Total months of experience from merged, non-overlapping employment
    intervals. Pure Python date arithmetic -- never delegated to the LLM.
    """
    intervals = []
    for experience in resume.experience:
        interval = _experience_interval(experience)
        if interval is not None:
            intervals.append(interval)

    total_months = 0
    for start, end in _merge_intervals(intervals):
        delta = relativedelta(end, start)
        months = delta.years * 12 + delta.months
        if delta.days > 0:
            months += 1  # a partial month still counts as a month
        total_months += months

    return total_months
