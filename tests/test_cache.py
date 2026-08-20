"""Tests for the role-profile cache and location fallback in cache.py.

Profiles are seeded directly into the disk cache via save_json rather than
going through mine_role_profile, so these tests never touch the network --
same reasoning as test_ats.py's hand-built fixtures.

Plain asyncio.run() rather than async test functions, to avoid adding a
pytest-asyncio dependency for four tests.
"""

import asyncio
from datetime import datetime, timezone

import pytest

import app.services.roleprofile.cache as cache_module
from app.config import settings
from app.services.roleprofile.cache import _cache_key, get_or_mine
from app.store import save_json

TEST_COLLECTION = "role_profiles"


def _seed_profile(role: str, location: str, postings_sampled: int) -> None:
    profile = {
        "role": role,
        "location": location,
        "postings_sampled": postings_sampled,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "skill_frequencies": {},
        "requirement_sentences": [],
        "median_experience_years": 0.0,
        "source_ids": [],
    }
    save_json(TEST_COLLECTION, _cache_key(role, location), profile)


@pytest.fixture(autouse=True)
def _demo_mode_off():
    original = settings.demo_mode
    settings.demo_mode = False
    yield
    settings.demo_mode = original


def test_no_fallback_when_postings_are_sufficient():
    _seed_profile("test role a", "Ahmedabad", 40)

    result = asyncio.run(get_or_mine("test role a", "Ahmedabad"))

    assert result["location_fallback"] is None
    assert result["postings_sampled"] == 40


def test_fallback_triggers_when_sparse():
    _seed_profile("test role b", "Ahmedabad", 17)
    _seed_profile("test role b", "India", 40)

    result = asyncio.run(get_or_mine("test role b", "Ahmedabad"))

    assert result["postings_sampled"] == 40  # the India profile's data
    assert result["location_fallback"] == {
        "requested_location": "Ahmedabad",
        "used_location": "India",
        "requested_postings_sampled": 17,
    }


def test_no_fallback_when_already_the_target_location():
    """Sparse results for India itself must not trigger a fallback to
    India -- that would be a no-op at best, infinite recursion at worst.
    """
    _seed_profile("test role c", "India", 5)

    result = asyncio.run(get_or_mine("test role c", "India"))

    assert result["location_fallback"] is None
    assert result["postings_sampled"] == 5


def test_demo_mode_skips_fallback_gracefully_when_target_not_cached():
    """DEMO_MODE never hits the network -- if the fallback location isn't
    cached either, this must return the original sparse profile rather
    than raising.
    """
    _seed_profile("test role d", "Ahmedabad", 17)
    settings.demo_mode = True

    result = asyncio.run(get_or_mine("test role d", "Ahmedabad"))

    assert result["location_fallback"] is None
    assert result["postings_sampled"] == 17


def test_demo_mode_still_falls_back_when_target_is_cached():
    _seed_profile("test role e", "Ahmedabad", 17)
    _seed_profile("test role e", "India", 40)
    settings.demo_mode = True

    result = asyncio.run(get_or_mine("test role e", "Ahmedabad"))

    assert result["postings_sampled"] == 40
    assert result["location_fallback"]["used_location"] == "India"


def test_demo_mode_never_calls_mine_role_profile_on_a_cache_hit(monkeypatch):
    """Airtight check of the DEMO_MODE contract: patch mine_role_profile
    to fail loudly if it's ever called, then confirm a cache hit under
    DEMO_MODE never reaches it -- not just that the right data came back,
    but that the network path was never even attempted.
    """

    async def _network_was_hit(*args, **kwargs):
        raise AssertionError("DEMO_MODE must never call mine_role_profile")

    monkeypatch.setattr(cache_module, "mine_role_profile", _network_was_hit)

    _seed_profile("test role f", "India", 40)
    settings.demo_mode = True

    result = asyncio.run(get_or_mine("test role f", "India"))
    assert result["postings_sampled"] == 40


def test_demo_mode_never_calls_mine_role_profile_and_raises_on_a_cache_miss(monkeypatch):
    """Same airtight check for the miss case: DEMO_MODE with nothing
    cached must raise RuntimeError, not silently fall through to mining.
    """

    async def _network_was_hit(*args, **kwargs):
        raise AssertionError("DEMO_MODE must never call mine_role_profile")

    monkeypatch.setattr(cache_module, "mine_role_profile", _network_was_hit)

    settings.demo_mode = True

    with pytest.raises(RuntimeError, match="DEMO_MODE is on"):
        asyncio.run(get_or_mine("test role g", "Nowhereville"))
