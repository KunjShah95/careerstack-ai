"""Adzuna job search API client.

Thin, retried wrapper around Adzuna's job search endpoint. Every failure
after retries returns [] and logs rather than raising -- one dead job
source must not kill a role-profile mining run.
"""

import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.adzuna.com/v1/api/jobs"
MAX_DAYS_OLD = 30
REQUEST_TIMEOUT_SECONDS = 15.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def _get(url: str, params: dict) -> httpx.Response:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response


async def search(query: str, location: str, limit: int, country: str = "in") -> list[dict]:
    """Search Adzuna for job postings.

    Returns a list of dicts with id, title, company, location, description,
    category, url, created, salary_min, salary_max. Adzuna truncates
    `description` to a short snippet (~250 chars) on the free tier rather
    than the full posting text -- that's an API limitation, not a bug here.
    `title` and `category` are never truncated, which is why miner.py scans
    them too instead of relying on description alone.
    """
    url = f"{BASE_URL}/{country}/search/1"
    params = {
        "app_id": settings.adzuna_app_id,
        "app_key": settings.adzuna_app_key,
        "results_per_page": limit,
        "what": query,
        "where": location,
        "max_days_old": MAX_DAYS_OLD,
        "content-type": "application/json",
    }

    try:
        response = await _get(url, params)
    except httpx.HTTPError as exc:
        logger.warning("Adzuna search failed for query=%r location=%r: %s", query, location, exc)
        return []

    try:
        payload = response.json()
    except ValueError as exc:
        logger.warning(
            "Adzuna returned unparseable JSON for query=%r location=%r: %s", query, location, exc
        )
        return []

    results = []
    for job in payload.get("results", []):
        company = job.get("company") or {}
        job_location = job.get("location") or {}
        category = job.get("category") or {}
        results.append(
            {
                "id": job.get("id"),
                "title": job.get("title"),
                "company": company.get("display_name"),
                "location": job_location.get("display_name"),
                "description": job.get("description"),
                "category": category.get("label"),
                "url": job.get("redirect_url"),
                "created": job.get("created"),
                "salary_min": job.get("salary_min"),
                "salary_max": job.get("salary_max"),
            }
        )
    return results
