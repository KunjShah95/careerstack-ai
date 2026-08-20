"""Disk cache for LLM-parsed resumes, same pattern as
roleprofile/cache.py but keyed on the file's own content.

A resume's parse never goes stale -- the file doesn't change -- so unlike
role_profiles this cache has no TTL. Keying on sha1 of the raw file bytes
(not the filename) means identical uploads hit the cache regardless of
what the file happens to be named.
"""

import hashlib

from app.config import settings
from app.models.resume import ParsedResume
from app.services.extraction.llm_parse import parse_resume
from app.store import load_json, save_json

COLLECTION = "parsed_resumes"


def _cache_key(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()


def get_or_parse(file_bytes: bytes, resume_text: str) -> ParsedResume:
    """Cached ParsedResume for this exact file's bytes; calls parse_resume()
    (Groq) only on a cache miss.

    resume_text is the already-extracted text to parse on a miss -- kept
    as a separate argument rather than re-extracted here, since the caller
    (run_analysis) already has it and text_extract.py's extraction isn't
    this function's job.

    In DEMO_MODE, this never touches Groq: it returns the cached parse if
    one exists, or raises a clear RuntimeError if it doesn't. Pre-warm the
    cache with scripts/prep_demo.py, DEMO_MODE off, before relying on this
    in demo mode.
    """
    key = _cache_key(file_bytes)
    cached = load_json(COLLECTION, key)

    if cached is not None:
        return ParsedResume.model_validate(cached)

    if settings.demo_mode:
        raise RuntimeError("No cached parse for this file. Run prep_demo.py with DEMO_MODE off first.")

    resume = parse_resume(resume_text)
    save_json(COLLECTION, key, resume.model_dump())
    return resume
