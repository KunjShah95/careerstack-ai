"""Adzuna adapter.

Deliberately a thin wrapper over the existing roleprofile/adzuna.py client
rather than a rewrite: that module is still imported directly by miner.py
for role-profile mining, and it already carries the tenacity retry, the
returns-[]-on-failure contract and the payload flattening. Duplicating it
here would leave two copies to keep in sync.

What this adds is the common shape (base.raw_posting) and the paging loop
discovery needs.
"""

import logging

from app.services.jobs.base import raw_posting
from app.services.roleprofile.adzuna import search as adzuna_search

logger = logging.getLogger(__name__)

SOURCE_NAME = "adzuna"

RESULTS_PER_PAGE = 50
MAX_PAGES = 3

# Adzuna's India endpoint reports salaries in INR; the API returns no
# currency field, so this is the country endpoint's currency, not a value
# read off any individual posting.
CURRENCY = "INR"


def _normalise(posting: dict) -> dict:
    has_salary = posting.get("salary_min") is not None or posting.get("salary_max") is not None
    return raw_posting(
        source=SOURCE_NAME,
        external_id=str(posting.get("id")) if posting.get("id") is not None else None,
        title=posting.get("title"),
        company=posting.get("company"),
        location=posting.get("location"),
        description=posting.get("description"),
        category=posting.get("category"),
        url=posting.get("url"),
        created=posting.get("created"),
        salary_min=posting.get("salary_min"),
        salary_max=posting.get("salary_max"),
        salary_currency=CURRENCY if has_salary else None,
    )


async def fetch(role: str, location: str, limit: int = RESULTS_PER_PAGE * MAX_PAGES) -> list[dict]:
    """Search Adzuna for one query, paging until `limit` or an empty page.

    An empty page ends paging: it's either a genuine end-of-results or a
    failure the client already logged and swallowed, and neither warrants
    another request.
    """
    collected: list[dict] = []

    for page in range(1, MAX_PAGES + 1):
        postings = await adzuna_search(
            role, location, limit=RESULTS_PER_PAGE, country="in", page=page
        )
        if not postings:
            break

        collected.extend(_normalise(posting) for posting in postings)
        if len(collected) >= limit:
            break

    return collected[:limit]
