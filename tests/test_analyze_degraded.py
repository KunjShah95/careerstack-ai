"""Tests for the degraded-analysis fallback in analyze.py: if role profile
mining fails entirely, fall back to the most recent cached profile for any
role rather than 500ing.

Uses a real fixture (blank-page detection aside, extract_text/layout are
local and network-free) but stubs out get_or_parse and get_or_mine so no
Groq or Adzuna calls happen -- this is testing analyze.py's own fallback
wiring, not those two external dependencies.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

import app.services.analyze as analyze_module
from app.models.resume import ContactInfo, ParsedResume
from app.services.analyze import RoleProfileUnavailableError, run_analysis
from app.services.roleprofile.cache import _cache_key
from app.store import save_json

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _stub_resume() -> ParsedResume:
    return ParsedResume(
        contact=ContactInfo(name="Someone"),
        skills=["Python"],
        sections_found=["skills"],
        total_experience_months=12,
    )


def _seed_profile(role: str, location: str, postings_sampled: int) -> dict:
    profile = {
        "role": role,
        "location": location,
        "postings_sampled": postings_sampled,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "skill_frequencies": {"python": {"frequency": 0.5, "count": postings_sampled // 2}},
        "requirement_sentences": [],
        "median_experience_years": 3.0,
        "source_ids": [],
    }
    save_json("role_profiles", _cache_key(role, location), profile)
    return profile


def test_falls_back_to_most_recent_cached_profile_when_mining_fails(monkeypatch):
    monkeypatch.setattr(analyze_module, "get_or_parse", lambda file_bytes, text: _stub_resume())

    async def _mining_failed(role, location):
        raise RuntimeError("simulated mining failure")

    monkeypatch.setattr(analyze_module, "get_or_mine", _mining_failed)

    fallback = _seed_profile("some other role", "Delhi", 40)

    file_bytes = (FIXTURES_DIR / "demo_after.pdf").read_bytes()
    result = asyncio.run(run_analysis(file_bytes, "demo_after.pdf", "backend developer", "Ahmedabad"))

    assert result["degraded"] is True
    assert "some other role" in result["degraded_message"]
    assert "Delhi" in result["degraded_message"]
    assert result["role_profile_meta"]["role"] == fallback["role"]
    assert result["role_profile_meta"]["location"] == fallback["location"]
    # the analysis still completed and produced real scores, not an error
    assert isinstance(result["overall_score"], float)


def test_no_degradation_when_mining_succeeds_normally(monkeypatch):
    monkeypatch.setattr(analyze_module, "get_or_parse", lambda file_bytes, text: _stub_resume())

    profile = _seed_profile("backend developer", "Ahmedabad", 40)

    async def _mining_succeeded(role, location):
        return {**profile, "location_fallback": None}

    monkeypatch.setattr(analyze_module, "get_or_mine", _mining_succeeded)

    file_bytes = (FIXTURES_DIR / "demo_after.pdf").read_bytes()
    result = asyncio.run(run_analysis(file_bytes, "demo_after.pdf", "backend developer", "Ahmedabad"))

    assert result["degraded"] is False
    assert result["degraded_message"] is None


def test_raises_role_profile_unavailable_when_nothing_is_cached_anywhere(monkeypatch):
    """Mining fails AND there's no cached profile for any role/location --
    genuinely nothing to fall back to. Must raise a specific, catchable
    error (main.py maps this to a 503), never a raw exception that
    escapes as an unhandled 500.
    """
    monkeypatch.setattr(analyze_module, "get_or_parse", lambda file_bytes, text: _stub_resume())

    async def _mining_failed(role, location):
        raise RuntimeError("simulated mining failure")

    monkeypatch.setattr(analyze_module, "get_or_mine", _mining_failed)
    monkeypatch.setattr(analyze_module, "get_most_recent_cached_profile", lambda: None)

    file_bytes = (FIXTURES_DIR / "demo_after.pdf").read_bytes()

    with pytest.raises(RoleProfileUnavailableError):
        asyncio.run(run_analysis(file_bytes, "demo_after.pdf", "backend developer", "Ahmedabad"))
