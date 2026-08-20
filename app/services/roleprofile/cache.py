"""Disk cache for mined role profiles.

Adzuna's free tier is ~1000 calls/month; a mining run costs 3 calls (one
per query variant), so re-mining on every request would burn through it
fast. Cached profiles live under data/role_profiles/ via app/store.py.
"""

import hashlib
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.services.roleprofile.miner import mine_role_profile
from app.store import load_json, save_json

COLLECTION = "role_profiles"
CACHE_TTL_DAYS = 7


def _cache_key(role: str, location: str) -> str:
    raw = f"{role.lower()}|{location.lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _is_fresh(profile: dict) -> bool:
    sampled_at = profile.get("sampled_at")
    if not sampled_at:
        return False

    try:
        sampled_time = datetime.fromisoformat(sampled_at)
    except ValueError:
        return False

    if sampled_time.tzinfo is None:
        sampled_time = sampled_time.replace(tzinfo=timezone.utc)

    return datetime.now(timezone.utc) - sampled_time < timedelta(days=CACHE_TTL_DAYS)


async def get_or_mine(role: str, location: str) -> dict:
    """Cached role profile for (role, location); mines a fresh one only
    when the cache is missing or older than CACHE_TTL_DAYS.

    In DEMO_MODE, this never touches the network: it returns the cached
    profile if one exists, or raises a clear error if it doesn't. That's
    what lets the demo run with zero live network dependency -- cache the
    profiles you need beforehand, with DEMO_MODE off.
    """
    key = _cache_key(role, location)
    cached = load_json(COLLECTION, key)

    if settings.demo_mode:
        if cached is not None:
            return cached
        raise RuntimeError(
            f"DEMO_MODE is on and no cached role profile exists for "
            f"role={role!r}, location={location!r}. Mine and cache it with "
            f"DEMO_MODE off before relying on it in demo mode."
        )

    if cached is not None and _is_fresh(cached):
        return cached

    profile = await mine_role_profile(role, location)
    save_json(COLLECTION, key, profile)
    return profile
