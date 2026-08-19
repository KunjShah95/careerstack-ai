# CareerStack AI — Resume Analysis Module

## What this is

An ATS resume analysis engine. A user uploads a resume (PDF/DOCX) and types a
target job role. The system scores the resume against what that role actually
demands in the live job market, and returns a prioritised list of fixes.

Academic project (Software Group Project, 7th sem CSE). Demo deadline: 22 Aug.
Scope is deliberately narrow — see "Out of scope" below and do not exceed it.

## The one rule that matters

**The LLM never produces the score.**

- LLM does: extract structured data from resume text, extract skills from job
  descriptions, write human-readable explanations of already-computed numbers.
- Deterministic Python does: all scoring, all arithmetic, all weighting.

Reason: the score must be reproducible across runs and explainable in a viva.
If you ever find yourself prompting a model for a number, stop and write a
function instead.

## Scoring formula

```
ATS_SCORE = 0.40 * keyword_coverage
          + 0.25 * semantic_fit
          + 0.20 * format_compliance
          + 0.15 * experience_alignment
```

Each term in [0, 1], multiplied by 100 at the end. Weights live in one constant
named WEIGHTS in app/services/scoring/ats.py. Never hardcode them elsewhere.

### keyword_coverage
Frequency-weighted, not binary. The requirement set is mined from ~40 real job
postings for the target role, so each skill carries its market frequency as its
weight:

```
score = sum(frequency_i * matched_i) / sum(frequency_i)
```

Skills below 0.15 frequency are dropped as noise. Skill matching is three-tier:
exact match on normalised skill list, then word-boundary regex over full resume
text, then rapidfuzz ratio >= 90.

### semantic_fit
Per-requirement, NOT whole-document. For each requirement sentence in the role
profile, find the best-matching line anywhere in the resume, then average those
maxima. Whole-document cosine similarity destroys the signal — do not do it.

Calibrate raw MiniLM cosine from the 0.30-0.85 band onto 0-1.

### format_compliance
Ten deterministic checks with individual point values (email present, text
extractable, single column, no tables, no header/footer text, core sections
present, reasonable length, dates parseable, standard bullets, phone present).
Each failure returns its own penalty so the UI can show "fixing this recovers
N points".

### experience_alignment
0.7 * years_component + 0.3 * education_component. Years come from merged,
non-overlapping employment intervals computed in Python — never ask the LLM to
do date arithmetic.

## Tech stack

- Python 3.12, FastAPI, uvicorn
- PyMuPDF (text + layout), pdfplumber (table detection), docx2txt
- sentence-transformers all-MiniLM-L6-v2, CPU only, local
- groq SDK (llama-3.3-70b-versatile), temperature=0
- rapidfuzz, httpx, tenacity, pydantic v2
- Adzuna API for mining role profiles (country code "in" for India)

Deliberately NOT used: LangChain, LangGraph, MongoDB, Docker, React, Firebase.
Do not introduce them.

## Storage

JSON files under ./data/. No database. Cache role profiles keyed on
sha1(role + location) with a 7-day TTL. Cache every Adzuna response to disk —
the free tier is ~1000 calls/month and blowing it kills the demo.

Add a DEMO_MODE env flag that reads only from cache and never hits the network.

## Layout

```
app/
  main.py                       FastAPI app, routes only, zero logic
  config.py                     pydantic-settings from .env
  models/                       Pydantic schemas
  services/
    extraction/                 text_extract.py, layout.py, llm_parse.py
    scoring/                    ats.py, keywords.py, semantic.py,
                                formatting.py, experience.py
    roleprofile/                miner.py, adzuna.py, cache.py
  data/skill_aliases.json
tests/
  fixtures/                     sample resumes (real PDFs)
static/index.html               single-page UI, plain fetch(), no build step
data/                           runtime cache, gitignored
```

Routers validate input, call a service, return a response. Every algorithm
lives in services/. This is non-negotiable — it is what makes the code testable.

## Conventions

- Type hints on every function signature
- Pure functions in services/ wherever possible; pass data in, return data out
- No global mutable state except the lru_cached embedding model
- Every external call wrapped in tenacity retry, and every job-source failure
  returns [] rather than raising — one dead source must not kill a run
- Round every number that reaches the user
- Errors surface as plain-language messages, never raw exceptions

## Commands

```
uvicorn app.main:app --reload      run dev server on :8000
pytest -q                          run tests
pytest tests/test_scoring.py -v    scoring tests only
```

## Build order

Do these strictly in sequence. Do not start a step before the previous one runs
correctly on real input.

1. text_extract.py — prove it on 5 real resume PDFs, print to terminal
2. llm_parse.py — structured extraction, validated against ParsedResume
3. adzuna.py + miner.py — mine a role profile, print the frequency table
4. formatting.py — pure Python, no network, cannot fail in a demo. Build first
   among the scoring components.
5. keywords.py — frequency-weighted coverage
6. semantic.py — per-requirement embeddings
7. experience.py
8. ats.py — combine
9. FastAPI routes
10. static/index.html

## Testing

Write tests for scoring components as they are built, not afterwards. Scoring
functions are pure, so they are cheap to test and the tests catch silent
regressions in the numbers.

## Out of scope — do not build

Bullet rewriting, job discovery UI, application tracker, authentication,
OCR, LangGraph orchestration, MongoDB, React frontend, Docker.

If a change would require any of these, say so and stop rather than adding it.

## Demo safety

Two sample resumes live in tests/fixtures/: one deliberately bad (two-column,
contact details in the header, thin skills) and one fixed version of the same
resume. The demo is: score the bad one (~43), walk through the detected issues,
score the fixed one (~70). Both must have cached analyses on disk before the
demo so nothing depends on a live network call.
