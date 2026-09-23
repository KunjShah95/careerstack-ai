"""The adapter interface every job source implements.

A source takes a role, a location and a result budget, and returns raw
posting dicts in one common shape. Normalising *here* rather than in
discovery.py is the point of the interface: discovery stays unaware of
which provider a posting came from, so adding a fourth source later
touches this package and nothing else.

Two rules hold for every adapter, and the pipeline depends on both:

1. Never raise. A dead or rate-limited source returns [] and logs. One
   source failing must not cost the user their other results -- the same
   rule CLAUDE.md already states for job sources.
2. Always stamp `source` with the adapter's own SOURCE_NAME, since dedupe
   priority in discovery.py is decided by that field.
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

# Dedupe priority when the same posting arrives from several sources,
# best first. JSearch wins because it returns full descriptions rather
# than a truncated snippet, which is what the per-posting keyword score is
# computed over. Greenhouse is authoritative for its own companies (it IS
# their ATS) but only covers those companies. Adzuna is the broad fallback.
SOURCE_PRIORITY = ["jsearch", "greenhouse", "adzuna"]


def source_rank(source: str | None) -> int:
    """Lower is better. An unknown source sorts last rather than raising,
    so a future adapter that forgets to register here degrades instead of
    breaking dedupe.
    """
    try:
        return SOURCE_PRIORITY.index(source or "")
    except ValueError:
        return len(SOURCE_PRIORITY)


class JobSource(Protocol):
    """Structural interface -- adapters are plain modules, not classes, so
    this documents and type-checks the shape without forcing inheritance.
    """

    SOURCE_NAME: str

    async def fetch(self, role: str, location: str, limit: int) -> list[dict]:
        ...


# Display credit per source. Each provider's terms require attribution, so
# the label travels with the posting rather than being inferred in the UI.
SOURCE_LABELS = {
    "jsearch": "via JSearch",
    "greenhouse": "via Greenhouse",
    "adzuna": "via Adzuna",
}


def default_source_label(source: str | None) -> str:
    return SOURCE_LABELS.get(source or "", f"via {source}" if source else "via an external board")


def raw_posting(
    *,
    source: str,
    external_id: str | None,
    title: str | None,
    company: str | None,
    location: str | None,
    description: str | None,
    url: str | None,
    created: str | None,
    salary_min: float | None = None,
    salary_max: float | None = None,
    salary_currency: str | None = None,
    category: str | None = None,
    source_label: str | None = None,
    is_remote: bool | None = None,
    employment_type: str | None = None,
) -> dict:
    """The common raw-posting shape, built in one place.

    Keyword-only on purpose: these are mostly-optional string fields, and
    positional calls across three adapters would be a silent field-swap
    waiting to happen.

    Field names match what discovery.normalise_posting() already reads
    from Adzuna payloads, so existing callers need no changes.

    is_remote is a tri-state: True/False when the provider states it
    (JSearch does, via job_is_remote), None when it doesn't. None means
    "unknown", and discovery falls back to sniffing the text for it --
    passing False there would assert something the provider never said.
    """
    return {
        "id": external_id,
        "source": source,
        "source_label": source_label or default_source_label(source),
        "title": title,
        "company": company,
        "location": location,
        "description": description or "",
        "category": category,
        "employment_type": employment_type,
        "is_remote": is_remote,
        "url": url,
        "created": created,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": salary_currency,
    }
