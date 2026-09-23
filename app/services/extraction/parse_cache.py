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


def resume_file_key(file_bytes: bytes) -> str:
    """Public cache key for a resume file's bytes.

    Exposed (rather than kept private) because analyze.py stamps this onto
    the saved analysis record, which is what lets discovery.py recover the
    full ParsedResume from an analysis_id alone -- an analysis stores only
    scoring output, not the resume itself. See load_cached_parse.
    """
    return hashlib.sha1(file_bytes).hexdigest()


def load_cached_parse(key: str) -> ParsedResume | None:
    """The cached ParsedResume for a resume_file_key, or None if this key
    was never cached (or predates the key being recorded on analyses).

    Never parses on a miss -- a miss here means the caller has to degrade,
    not silently spend a Groq call on data it only needs opportunistically.
    """
    cached = load_json(COLLECTION, key)
    return ParsedResume.model_validate(cached) if cached is not None else None


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
    key = resume_file_key(file_bytes)
    cached = load_json(COLLECTION, key)

    if cached is not None:
        return ParsedResume.model_validate(cached)

    if settings.demo_mode:
        raise RuntimeError("No cached parse for this file. Run prep_demo.py with DEMO_MODE off first.")

    resume = parse_resume(resume_text)
    save_json(COLLECTION, key, resume.model_dump())
    return resume
