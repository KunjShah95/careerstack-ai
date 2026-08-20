"""Disk cache for mined role profiles.

Adzuna's free tier is ~1000 calls/month; a mining run costs 3 calls (one
per query variant), so re-mining on every request would burn through it
fast. Cached profiles live under data/role_profiles/ via app/store.py.
"""

import hashlib
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.services.roleprofile.miner import mine_role_profile
from app.store import list_keys, load_json, save_json

COLLECTION = "role_profiles"
CACHE_TTL_DAYS = 7

# A city-level location can turn up too few postings to mine anything
# meaningful. Below this, get_or_mine retries with LOCATION_FALLBACK_TARGET
# (half of miner.py's default target_postings=40) instead of returning a
# profile with too little data behind it.
LOCATION_FALLBACK_MIN_POSTINGS = 20
LOCATION_FALLBACK_TARGET = "India"


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


async def _get_cached_or_mine(role: str, location: str) -> dict:
    """Cached role profile for one exact (role, location) pair; mines a
    fresh one only when the cache is missing or older than CACHE_TTL_DAYS.

    In DEMO_MODE, this never touches the network: it returns the cached
    profile if one exists, or raises RuntimeError if it doesn't. That's
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


async def get_or_mine(role: str, location: str) -> dict:
    """Cached role profile for (role, location), with a location fallback
    on top of _get_cached_or_mine.

    If the requested location comes back sparse (fewer than
    LOCATION_FALLBACK_MIN_POSTINGS postings) and isn't already
    LOCATION_FALLBACK_TARGET itself, this also fetches a profile for
    LOCATION_FALLBACK_TARGET and returns that instead, annotated with
    "location_fallback" so the caller can say e.g. "only 17 postings found
    in Ahmedabad, showing India instead":

        {"requested_location": "Ahmedabad", "used_location": "India",
         "requested_postings_sampled": 17}

    "location_fallback" is always present on the returned dict -- None
    when no fallback happened. It's added here, not stored in the cached
    JSON itself, since it describes this particular lookup, not a
    property of the mined data.

    The fallback lookup goes through DEMO_MODE the same way the primary
    lookup does; if DEMO_MODE is on and India isn't cached either, the
    fallback is silently skipped and the original sparse profile is
    returned rather than raising over a fallback that was never critical
    to begin with.
    """
    profile = await _get_cached_or_mine(role, location)

    is_fallback_target = location.strip().lower() == LOCATION_FALLBACK_TARGET.lower()
    if not is_fallback_target and profile.get("postings_sampled", 0) < LOCATION_FALLBACK_MIN_POSTINGS:
        try:
            fallback_profile = await _get_cached_or_mine(role, LOCATION_FALLBACK_TARGET)
        except RuntimeError:
            pass  # DEMO_MODE and India isn't cached either -- fall through
        else:
            return {
                **fallback_profile,
                "location_fallback": {
                    "requested_location": location,
                    "used_location": LOCATION_FALLBACK_TARGET,
                    "requested_postings_sampled": profile.get("postings_sampled", 0),
                },
            }

    return {**profile, "location_fallback": None}


def get_most_recent_cached_profile() -> dict | None:
    """The freshest cached role profile across every role/location ever
    mined, or None if nothing has ever been cached.

    Last-resort fallback for when get_or_mine can't produce a profile for
    the specifically-requested role/location at all -- see
    app/services/analyze.py's degraded-analysis handling. Not used by
    get_or_mine itself; get_or_mine's job is one exact (role, location),
    this is "give me anything at all".
    """
    profiles = [load_json(COLLECTION, key) for key in list_keys(COLLECTION)]
    profiles = [profile for profile in profiles if profile]
    if not profiles:
        return None

    # ISO 8601 strings with a consistent format sort chronologically as
    # plain strings -- no need to parse them into datetimes just to
    # compare.
    return max(profiles, key=lambda profile: profile.get("sampled_at") or "")
