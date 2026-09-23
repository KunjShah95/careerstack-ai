"""Tests for list_user_analyses, the backing function for GET /api/analyses.

Network-free: get_or_parse/get_or_mine stubbed the same way test_ats.py
and test_analyze_degraded.py already do.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import app.services.analyze as analyze_module
from app.models.resume import ContactInfo, ParsedResume
from app.services.analyze import list_user_analyses, run_analysis

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _stub_resume() -> ParsedResume:
    return ParsedResume(
        contact=ContactInfo(name="Someone"),
        skills=["Python"],
        sections_found=["skills"],
        total_experience_months=12,
    )


def test_list_user_analyses_returns_summaries_newest_first_with_both_scores(monkeypatch):
    monkeypatch.setattr(analyze_module, "get_or_parse", lambda file_bytes, text: _stub_resume())

    async def _mining_succeeded(role, location):
        return {
            "role": "backend developer",
            "location": "India",
            "postings_sampled": 40,
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "skill_frequencies": {"python": {"frequency": 0.5, "count": 20}},
            "requirement_sentences": [],
            "median_experience_years": 3.0,
            "source_ids": [],
            "location_fallback": None,
        }

    monkeypatch.setattr(analyze_module, "get_or_mine", _mining_succeeded)

    user_id = f"history-test-user-{uuid.uuid4().hex}"
    file_bytes = (FIXTURES_DIR / "demo_after.pdf").read_bytes()

    first = asyncio.run(run_analysis(file_bytes, "first.pdf", "backend developer", "India", user_id))
    second = asyncio.run(run_analysis(file_bytes, "second.pdf", "backend developer", "India", user_id))

    summaries = list_user_analyses(user_id)

    assert [s["analysis_id"] for s in summaries] == [second["analysis_id"], first["analysis_id"]]
    for summary in summaries:
        assert set(summary.keys()) == {
            "analysis_id",
            "filename",
            "role",
            "location",
            "overall_score",
            "band",
            "created_at",
            "parse_score",
        }
        assert isinstance(summary["overall_score"], float)
        assert isinstance(summary["parse_score"], float)


def test_list_user_analyses_only_returns_that_users_analyses(monkeypatch):
    monkeypatch.setattr(analyze_module, "get_or_parse", lambda file_bytes, text: _stub_resume())

    async def _mining_succeeded(role, location):
        return {
            "role": "backend developer",
            "location": "India",
            "postings_sampled": 40,
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "skill_frequencies": {},
            "requirement_sentences": [],
            "median_experience_years": 3.0,
            "source_ids": [],
            "location_fallback": None,
        }

    monkeypatch.setattr(analyze_module, "get_or_mine", _mining_succeeded)

    user_a = f"user-a-{uuid.uuid4().hex}"
    user_b = f"user-b-{uuid.uuid4().hex}"
    file_bytes = (FIXTURES_DIR / "demo_after.pdf").read_bytes()

    asyncio.run(run_analysis(file_bytes, "a.pdf", "backend developer", "India", user_a))
    asyncio.run(run_analysis(file_bytes, "b.pdf", "backend developer", "India", user_b))

    assert [s["filename"] for s in list_user_analyses(user_a)] == ["a.pdf"]
    assert [s["filename"] for s in list_user_analyses(user_b)] == ["b.pdf"]
