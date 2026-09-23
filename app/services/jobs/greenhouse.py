"""Greenhouse job board adapter.

No API key, no quota, and it's the companies' own ATS -- so for the boards
in app/data/greenhouse_companies.json this is authoritative data rather
than an aggregator's copy. It only covers those companies, which is why it
sits below JSearch but above Adzuna in dedupe priority.

Descriptions come back as HTML-escaped markup (`&lt;div&gt;...`), not
plain text, and need unescaping and tag-stripping before the skill
extractor sees them -- otherwise every posting reads as a wall of markup.
Measured on a real board: 5913 characters of raw markup reduce to 4103
characters of usable text, still an order of magnitude more than Adzuna's
~500 character snippet.

Each board is one HTTP call, fanned out concurrently under a semaphore. A
company that 404s or times out contributes [] and the run continues.
"""

import asyncio
import html
import json
import logging
import re
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.services.jobs.base import raw_posting

logger = logging.getLogger(__name__)

SOURCE_NAME = "greenhouse"

BASE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
REQUEST_TIMEOUT_SECONDS = 20.0
CONCURRENCY = 8

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_COMPANIES_FILE = _DATA_DIR / "greenhouse_companies.json"

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# City spellings that differ between what a user types and what boards
# publish. Greenhouse locations are free text written by each employer, so
# "Bangalore" (what people search) and "Bengaluru" (what boards say) never
# match on substring alone. Only genuine aliases for the same place --
# this is not a general synonym list.
_CITY_ALIASES = {
    "bangalore": ["bangalore", "bengaluru"],
    "bengaluru": ["bangalore", "bengaluru"],
    "bombay": ["bombay", "mumbai"],
    "mumbai": ["bombay", "mumbai"],
    "delhi": ["delhi", "new delhi", "ncr"],
    "new delhi": ["delhi", "new delhi", "ncr"],
    "gurgaon": ["gurgaon", "gurugram"],
    "gurugram": ["gurgaon", "gurugram"],
    "calcutta": ["calcutta", "kolkata"],
    "kolkata": ["calcutta", "kolkata"],
    "madras": ["madras", "chennai"],
    "chennai": ["madras", "chennai"],
    "poona": ["poona", "pune"],
    "pune": ["poona", "pune"],
    "trivandrum": ["trivandrum", "thiruvananthapuram"],
    "noida": ["noida", "greater noida"],
}

# Treated as "anywhere in the country", so a country-wide search keeps
# every India-located posting instead of matching the literal word.
_COUNTRY_WIDE = {"india", "in", "anywhere", ""}


def load_company_tokens() -> list[str]:
    """Board tokens from the curated file.

    Every token there was verified to return HTTP 200 -- see that file's
    _comment for why the originally-specified list could not be used as
    given (14 of its 15 entries 404'd).
    """
    try:
        payload = json.loads(_COMPANIES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", _COMPANIES_FILE.name, exc)
        return []
    return [entry["token"] for entry in payload.get("companies", []) if entry.get("token")]


def clean_description(content: str | None) -> str:
    """HTML-escaped Greenhouse markup -> plain text.

    Unescape first, then strip tags: doing it the other way round leaves
    `&lt;div&gt;` untouched, since it isn't a tag until it's unescaped.
    """
    if not content:
        return ""
    unescaped = html.unescape(content)
    without_tags = _TAG_RE.sub(" ", unescaped)
    return _WHITESPACE_RE.sub(" ", without_tags).strip()


def _location_terms(location: str) -> list[str]:
    normalised = (location or "").strip().lower()
    return _CITY_ALIASES.get(normalised, [normalised])


def matches_location(job_location: str | None, wanted: str) -> bool:
    """Case-insensitive city match against a board's free-text location.

    A country-wide search ("India") matches everything these boards
    return that is India-located; a city search matches that city or any
    known alias of it.
    """
    wanted_normalised = (wanted or "").strip().lower()
    haystack = (job_location or "").lower()

    if wanted_normalised in _COUNTRY_WIDE:
        return "india" in haystack

    return any(term and term in haystack for term in _location_terms(wanted_normalised))


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def _get_board(client: httpx.AsyncClient, token: str) -> list[dict]:
    response = await client.get(
        BASE_URL.format(token=token), params={"content": "true"}, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json().get("jobs", [])


def _normalise(job: dict, company_fallback: str) -> dict:
    location = (job.get("location") or {}).get("name")
    return raw_posting(
        source=SOURCE_NAME,
        external_id=str(job.get("id")) if job.get("id") is not None else None,
        title=job.get("title"),
        company=job.get("company_name") or company_fallback,
        location=location,
        description=clean_description(job.get("content")),
        category=", ".join(
            department.get("name", "") for department in (job.get("departments") or [])
        )
        or None,
        # Links back to the board itself, as the terms require.
        url=job.get("absolute_url"),
        created=job.get("first_published") or job.get("updated_at"),
    )


async def _fetch_one(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, token: str, location: str
) -> list[dict]:
    async with semaphore:
        try:
            jobs = await _get_board(client, token)
        except (httpx.HTTPError, ValueError) as exc:
            # A renamed or removed board 404s here. Log and contribute
            # nothing rather than failing the whole fan-out.
            logger.warning("Greenhouse board %r unavailable: %s", token, exc)
            return []

    return [
        _normalise(job, token)
        for job in jobs
        if matches_location((job.get("location") or {}).get("name"), location)
    ]


async def fetch(role: str, location: str, limit: int = 200) -> list[dict]:
    """Every posting across the curated boards that matches `location`.

    `role` is deliberately unused for filtering: these boards are small
    enough to take whole, and discovery ranks by fit anyway -- pre-filtering
    on a title string here would drop a "Software Engineer II" posting from
    a "backend developer" search for no good reason.
    """
    tokens = load_company_tokens()
    if not tokens:
        logger.warning("No Greenhouse company tokens configured -- source disabled")
        return []

    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_fetch_one(client, semaphore, token, location) for token in tokens],
            return_exceptions=True,
        )

    collected: list[dict] = []
    for token, result in zip(tokens, results):
        if isinstance(result, BaseException):
            # _fetch_one already swallows the expected failures; this is
            # the belt-and-braces case for anything unforeseen.
            logger.warning("Greenhouse board %r raised unexpectedly: %s", token, result)
            continue
        collected.extend(result)

    return collected[:limit]
