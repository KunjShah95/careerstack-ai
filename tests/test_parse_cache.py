"""Tests for the parsed-resume cache in parse_cache.py -- same pattern as
test_cache.py's coverage of roleprofile/cache.py, including an airtight
check that DEMO_MODE never calls Groq.
"""

import uuid

import pytest

import app.services.extraction.parse_cache as parse_cache_module
from app.config import settings
from app.models.resume import ContactInfo, ParsedResume
from app.services.extraction.parse_cache import get_or_parse, resume_file_key
from app.store import save_json


def _unique_bytes() -> bytes:
    # store.py writes to the real ./data/ cache, which persists across
    # test runs -- a literal byte string would get permanently cached to
    # disk the first time a "cache miss" test runs, silently turning it
    # into a "cache hit" test on every run after that. A fresh uuid per
    # call guarantees a genuine miss every time.
    return f"resume content {uuid.uuid4()}".encode()

COLLECTION = "parsed_resumes"


@pytest.fixture(autouse=True)
def _demo_mode_off():
    original = settings.demo_mode
    settings.demo_mode = False
    yield
    settings.demo_mode = original


def _stub_resume(name: str) -> ParsedResume:
    return ParsedResume(contact=ContactInfo(name=name), skills=["Python"], sections_found=["skills"])


def test_identical_bytes_hit_the_cache_regardless_of_filename(monkeypatch):
    calls = []

    def _fake_parse(text: str) -> ParsedResume:
        calls.append(text)
        return _stub_resume("Someone")

    monkeypatch.setattr(parse_cache_module, "parse_resume", _fake_parse)

    file_bytes = _unique_bytes()

    first = get_or_parse(file_bytes, "resume text v1")
    second = get_or_parse(file_bytes, "resume text v1, but this arg is ignored on a cache hit")

    assert len(calls) == 1  # parse_resume only called once
    assert first.contact.name == second.contact.name == "Someone"


def test_cache_key_is_based_on_file_bytes_not_filename():
    """The function doesn't take a filename at all -- this just confirms
    the key genuinely comes from content, not something incidental.
    """
    assert resume_file_key(b"resume A") != resume_file_key(b"resume B")
    assert resume_file_key(b"same bytes") == resume_file_key(b"same bytes")


def test_cache_miss_calls_parse_resume_and_saves_the_result(monkeypatch):
    def _fake_parse(text: str) -> ParsedResume:
        return _stub_resume("Freshly Parsed")

    monkeypatch.setattr(parse_cache_module, "parse_resume", _fake_parse)

    file_bytes = _unique_bytes()
    result = get_or_parse(file_bytes, "some resume text")

    assert result.contact.name == "Freshly Parsed"

    from app.store import load_json

    cached = load_json(COLLECTION, resume_file_key(file_bytes))
    assert cached is not None
    assert cached["contact"]["name"] == "Freshly Parsed"


def test_demo_mode_never_calls_parse_resume_on_a_cache_hit(monkeypatch):
    def _network_was_hit(text: str):
        raise AssertionError("DEMO_MODE must never call parse_resume")

    monkeypatch.setattr(parse_cache_module, "parse_resume", _network_was_hit)

    file_bytes = b"a resume that is already cached"
    save_json(COLLECTION, resume_file_key(file_bytes), _stub_resume("Cached Person").model_dump())

    settings.demo_mode = True
    result = get_or_parse(file_bytes, "text is irrelevant on a cache hit")

    assert result.contact.name == "Cached Person"


def test_demo_mode_never_calls_parse_resume_and_raises_on_a_cache_miss(monkeypatch):
    def _network_was_hit(text: str):
        raise AssertionError("DEMO_MODE must never call parse_resume")

    monkeypatch.setattr(parse_cache_module, "parse_resume", _network_was_hit)

    settings.demo_mode = True

    with pytest.raises(RuntimeError, match="No cached parse for this file"):
        get_or_parse(_unique_bytes(), "some text")
