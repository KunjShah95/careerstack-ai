"""JSearch (RapidAPI) adapter.

The one source that returns full job descriptions rather than a truncated
snippet, which is why it outranks the others in dedupe (see
base.SOURCE_PRIORITY). That matters directly: discovery's per-posting
keyword score is computed over the description text, and Adzuna's ~500
character cut-off is what forced the coverage-confidence damping in the
first place.

Two guards keep this inside a 200 call/month free tier:

1. One call per role/location, never per generated query variant.
   num_pages=2 buys ~20 results for the cost of a single call.
2. A local monthly counter (see quota.py). At MONTHLY_CALL_BUDGET the
   source stops itself, keeping a reserve back for the demo -- RapidAPI's
   free plan returns no usable remaining-quota header, so this is tracked
   on our side or not at all.

Without settings.rapidapi_key this returns [] for every call and warns
once, making no HTTP request. It never raises: discovery must still run on
the other sources.
"""

import logging
from datetime import datetime, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.services.jobs import quota
from app.services.jobs.base import raw_posting

logger = logging.getLogger(__name__)

SOURCE_NAME = "jsearch"
SOURCE_LABEL = "via JSearch"

BASE_URL = "https://jsearch.p.rapidapi.com/search"
API_HOST = "jsearch.p.rapidapi.com"
REQUEST_TIMEOUT_SECONDS = 30.0

COUNTRY = "in"
DATE_POSTED = "month"
LANGUAGE = "en"
NUM_PAGES = "2"  # ~20 results, still one call against quota

# The free tier is 200/month. Stopping at 180 reserves 20 calls so a demo
# can't be the thing that discovers the quota ran out.
MONTHLY_CALL_LIMIT = 200
MONTHLY_CALL_RESERVE = 20
MONTHLY_CALL_BUDGET = MONTHLY_CALL_LIMIT - MONTHLY_CALL_RESERVE

# Warn once per process, not once per call: a missing key is a
# configuration fact, and repeating it on every discovery run would bury
# the logs that matter.
_warned_missing_key = False


def _warn_missing_key_once() -> None:
    global _warned_missing_key
    if not _warned_missing_key:
        logger.warning(
            "RAPIDAPI_KEY is not set -- the JSearch source is disabled and will "
            "return no postings. Discovery still runs on the other sources. "
            "Set RAPIDAPI_KEY in .env to enable it."
        )
        _warned_missing_key = True


def is_configured() -> bool:
    return bool(settings.rapidapi_key)


def calls_used_this_month() -> int:
    return quota.usage(SOURCE_NAME)


# Statuses where retrying cannot help and only burns quota. 403 in
# particular is RapidAPI's "you are not subscribed to this API" -- a
# permanent account-level fact, not a transient blip, and retrying it
# three times costs three calls against a 200/month tier for nothing.
_PERMANENT_STATUSES = frozenset({400, 401, 403, 404})


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in _PERMANENT_STATUSES
    return isinstance(exc, httpx.HTTPError)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=_is_retryable,
    reraise=True,
)
async def _get(params: dict, headers: dict) -> httpx.Response:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(BASE_URL, params=params, headers=headers)
        response.raise_for_status()
        return response


def _salary(posting: dict) -> tuple[float | None, float | None, str | None]:
    minimum = posting.get("job_min_salary")
    maximum = posting.get("job_max_salary")
    if minimum is None and maximum is None:
        return None, None, None
    return minimum, maximum, posting.get("job_salary_currency")


def _location(posting: dict) -> str | None:
    """"{city}, {state}" when both are given, else whatever is present.

    Falls back to job_country so a posting with no city still carries a
    location rather than None -- discovery's location_fit reads this, and
    an empty string would score it as "somewhere else" rather than unknown.
    """
    city = posting.get("job_city")
    state = posting.get("job_state")
    if city and state:
        return f"{city}, {state}"
    return city or state or posting.get("job_country")


def _posted_at(posting: dict) -> str | None:
    """job_posted_at_datetime_utc normalised to an ISO string with an
    explicit UTC offset, so discovery's recency parser doesn't have to
    guess at a naive timestamp.
    """
    raw = posting.get("job_posted_at_datetime_utc")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        # Hand it back unchanged rather than dropping it; recency_score
        # already treats an unparseable date as "unknown, score the floor".
        return str(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _employment_type(posting: dict) -> str | None:
    value = posting.get("job_employment_type")
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item) or None
    return value or None


def _normalise(posting: dict) -> dict:
    salary_min, salary_max, currency = _salary(posting)
    return raw_posting(
        source=SOURCE_NAME,
        source_label=SOURCE_LABEL,
        external_id=posting.get("job_id"),
        title=posting.get("job_title"),
        company=posting.get("employer_name"),
        location=_location(posting),
        # The whole reason this source ranks first -- not a snippet.
        description=posting.get("job_description"),
        category=_employment_type(posting),
        employment_type=_employment_type(posting),
        is_remote=bool(posting.get("job_is_remote")),
        url=posting.get("job_apply_link"),
        created=_posted_at(posting),
        salary_min=salary_min,
        salary_max=salary_max,
        salary_currency=currency,
    )


async def fetch(role: str, location: str, limit: int = 50) -> list[dict]:
    """One JSearch call for this role/location.

    Returns [] -- never raises -- when the key is missing, the monthly
    budget is spent, or the request fails after retries.
    """
    if not is_configured():
        _warn_missing_key_once()
        return []

    if not quota.has_budget(SOURCE_NAME, MONTHLY_CALL_BUDGET):
        return []

    params = {
        "query": f"{role} in {location}",
        "country": COUNTRY,
        "page": "1",
        "num_pages": NUM_PAGES,
        "date_posted": DATE_POSTED,
        "language": LANGUAGE,
        # Deliberately no work_from_home: remote is a client-side filter on
        # the discovery page, so narrowing it here would remove postings the
        # user can still choose to see.
    }
    headers = {
        "X-RapidAPI-Key": settings.rapidapi_key,
        "X-RapidAPI-Host": API_HOST,
    }

    # Counted before the request: a call that times out still consumed
    # quota upstream, so counting only successes would drift optimistic.
    used = quota.record_call(SOURCE_NAME)
    logger.info("JSearch call %d/%d this month (role=%r, location=%r)",
                used, MONTHLY_CALL_BUDGET, role, location)

    try:
        response = await _get(params, headers)
    except httpx.HTTPError as exc:
        # Includes 429 (quota exhausted) and 403 (key not subscribed to
        # this API on RapidAPI) -- both expected failure modes that must
        # never take a run down.
        logger.warning("JSearch request failed for role=%r location=%r: %s", role, location, exc)
        return []

    try:
        payload = response.json()
    except ValueError as exc:
        logger.warning("JSearch returned unparseable JSON for role=%r: %s", role, exc)
        return []

    results = payload.get("data") or []
    return [_normalise(posting) for posting in results[:limit]]
