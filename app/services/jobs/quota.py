"""Monthly call counter for quota-limited job sources.

JSearch's free tier is 200 calls a month and RapidAPI returns no usable
remaining-quota header on this plan, so consumption is tracked locally:
every call is counted, the count resets on the 1st, and the source stops
itself before the tier runs out.

Persisted through app/store.py like everything else -- one JSON document
per source per month, so last month's usage stays readable rather than
being overwritten.
"""

import logging
from datetime import datetime, timezone

from app.store import load_json, save_json

logger = logging.getLogger(__name__)

COLLECTION = "api_quota"


def current_period(now: datetime | None = None) -> str:
    """The period key, "YYYY-MM". Counting per calendar month is what makes
    the reset automatic: on the 1st the key changes and the lookup simply
    misses, so there's no scheduled job to run and nothing to forget.
    """
    now = now or datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def _key(source: str, period: str) -> str:
    return f"{source}-{period}"


def usage(source: str, now: datetime | None = None) -> int:
    """Calls recorded for this source in the current period."""
    record = load_json(COLLECTION, _key(source, current_period(now)))
    return int((record or {}).get("calls", 0))


def record_call(source: str, now: datetime | None = None) -> int:
    """Count one call and return the new total.

    Recorded before the HTTP request rather than after: a request that
    times out or 500s still consumed quota on the provider's side, so
    counting only successes would drift optimistic and overrun the tier.
    """
    period = current_period(now)
    key = _key(source, period)
    record = load_json(COLLECTION, key) or {"source": source, "period": period, "calls": 0}
    record["calls"] = int(record.get("calls", 0)) + 1
    record["last_call_at"] = (now or datetime.now(timezone.utc)).isoformat()
    save_json(COLLECTION, key, record)
    return record["calls"]


def has_budget(source: str, limit: int, now: datetime | None = None) -> bool:
    """Whether this source may make another call under `limit`.

    `limit` is deliberately below the provider's real ceiling -- see
    jsearch.MONTHLY_CALL_BUDGET for why a reserve is held back.
    """
    used = usage(source, now)
    if used >= limit:
        logger.warning(
            "%s has used %d calls this month (self-imposed limit %d) -- skipping it "
            "to preserve the remaining quota. Resets on the 1st.",
            source,
            used,
            limit,
        )
        return False
    return True
