# CareerStack AI — Resume Analysis

An ATS resume analysis engine. Upload a resume (PDF or DOCX), name a target
role and city, and it scores the resume against what that role actually
demands in the live job market — then lists the fixes worth making, in
priority order, and finds matching jobs.

Academic project (Software Group Project, 7th sem CSE).

---

## The one thing worth knowing about the design

**The LLM never produces the score.**

An LLM extracts structured data from the resume text and writes plain-English
explanations of numbers that have already been computed. Every number — every
weight, every subscore, every percentage — comes from deterministic Python.

That is a deliberate constraint, not an oversight. It means the score is
reproducible across runs and every figure can be traced to a line of code, so
the system can be defended in a viva rather than hand-waved at.

```
ATS_SCORE = 0.40 * keyword_coverage      # frequency-weighted against ~40 real postings
          + 0.25 * semantic_fit          # per-requirement MiniLM similarity
          + 0.20 * format_compliance     # 9 deterministic parseability checks
          + 0.15 * experience_alignment  # merged employment intervals + education
```

---

## Requirements

- **Python 3.11 or newer.** The code uses `X | Y` union syntax (3.10+) and is
  developed and tested on 3.11. Nothing requires 3.12.
- About 2 GB of disk space — `sentence-transformers` pulls a PyTorch build,
  and the MiniLM model (~90 MB) downloads on first use.
- No database. Everything persists as JSON files under `./data/`.

---

## Quick start

```bash
git clone https://github.com/krishpatel719/careerstack-ai.git
cd careerstack-ai

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
cp .env.example .env           # copy .env.example .env   on Windows cmd
```

Now generate a JWT secret — **the app will sign tokens with an empty string
if you skip this**, which is worse than failing outright, because nothing
visibly breaks:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Paste the output into `.env` as `JWT_SECRET=...`, then:

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>.

> Deleting the `JWT_SECRET` line entirely makes the app refuse to start with a
> clear pydantic error. Leaving it *blank* does not — it starts and signs
> every token with `""`. Set it.

---

## Do I need API keys?

**To run the test suite: no.** All 181 tests pass on a fresh clone with no
keys configured — they are network-free by design.

```bash
pytest -q
```

**To analyse a real resume: yes, two.** Both have free tiers.

| Key | What it unlocks | Without it |
|---|---|---|
| `GROQ_API_KEY` | Resume parsing (the structured extraction step) | Analysis fails — this one is required |
| `ADZUNA_APP_ID` + `ADZUNA_APP_KEY` | Role profiles and the main job source | No market data to score against |
| `RAPIDAPI_KEY` | The JSearch job source (optional) | Job discovery runs on the other two sources |

- **Groq** — free at [console.groq.com](https://console.groq.com). Generous limits.
- **Adzuna** — free at [developer.adzuna.com](https://developer.adzuna.com).
  **~1000 calls/month**, and this project caches aggressively because of it
  (role profiles for 7 days, job-source results for 6 hours).
- **RapidAPI / JSearch** — optional, see the caveat below.

### The JSearch subscription trap

A RapidAPI key alone is **not** enough. You must also subscribe to the JSearch
API specifically, at
[rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch)
→ *Subscribe to Test* → free Basic plan (200 requests/month).

An unsubscribed key returns `403 {"message": "You are not subscribed to this
API."}` — **byte-identical to the response for a completely invalid key**. It
reads like a bad key when it is not. Verified by sending a deliberately
invalid key and diffing the responses.

The app handles this gracefully: JSearch logs one warning, returns nothing,
and discovery carries on with Adzuna and Greenhouse. Check your consumption
with `python scripts/check_quota.py`.

---

## Using it

1. **Register** an account (or log in) — analyses belong to the user who
   created them.
2. **Upload** a resume, name a target role and city.
3. **Read the dashboard** — two scores kept deliberately distinct:
   - *ATS Parse Score* — can an applicant tracking system read your document?
   - *Role Fit Score* — how well do you match this role's market demand?
4. **Work the action plan** — each fix shows the points it recovers.
5. **Find matching jobs** — live postings ranked against your resume.

### Try it with the included samples

`tests/fixtures/` holds a deliberately bad resume and a fixed version of the
same one:

| File | Role | Location | Expected |
|---|---|---|---|
| `demo_before.pdf` | backend developer | **India** | ~45 (Needs work) |
| `demo_after.pdf` | backend developer | **India** | ~72 (Competitive) |

Use **India**, not a city. Those figures hold against the India role profile.
Ahmedabad is a genuinely sparser market and gives ~38 → ~55 for the same two
files. Both are correct — India is the one these numbers describe.

---

## Offline demo mode

`DEMO_MODE=true` in `.env` makes the app serve **only** from cache and never
touch the network, so a dead wifi connection or an exhausted API quota cannot
break a live presentation.

```bash
# 1. With DEMO_MODE=false, warm every cache the demo needs:
python scripts/prep_demo.py

# 2. Set DEMO_MODE=true in .env, restart the server.
```

`prep_demo.py` caches resume parses, mines the role profile, runs both demo
analyses, and warms three job-discovery runs. It creates a demo account:
`demo@careerstack.ai` / `demo12345`.

In demo mode, anything **not** cached fails with a message naming the fix
rather than silently going online. That strictness is the point.

---

## Project layout

```
app/
  main.py                    FastAPI app — routes only, zero logic
  config.py                  pydantic-settings, reads .env
  models/                    Pydantic schemas (resume, job, user)
  services/
    extraction/              PDF/DOCX text, layout signals, LLM parsing
    scoring/                 ats, keywords, semantic, formatting, experience
    roleprofile/             Adzuna client, profile mining, caching
    jobs/                    Job source adapters (Adzuna, JSearch, Greenhouse)
    discovery.py             Job discovery pipeline
    auth.py                  JWT + bcrypt
  data/                      Skill vocabulary and aliases (source, not cache)
tests/
  fixtures/                  Real sample resumes
static/index.html            Entire frontend — one file, no build step
data/                        Runtime cache (gitignored)
```

Routers validate input, call a service, return a response. Every algorithm
lives in `services/`. That separation is what makes the scoring testable.

The frontend is a single self-contained HTML file with vanilla JavaScript —
no build step, no CDN, no npm. Open it and it works.

---

## Commands

```bash
uvicorn app.main:app --reload       # dev server on :8000
pytest -q                           # all 181 tests (no network, no keys needed)
pytest tests/test_discovery.py -v   # one module

python scripts/prep_demo.py         # warm every cache for an offline demo
python scripts/check_quota.py       # JSearch calls used this month
python scripts/verify_greenhouse.py # check the job boards still resolve
```

---

## Notes for anyone extending this

**Run the tests before trusting cached data.** `tests/conftest.py` points
storage at a temp directory for the whole session. Without it the suite writes
real cache files into `./data/` — that once seeded a role profile containing a
single skill, which silently inflated a demo score from 45.4 to 81.7 with
nothing failing to flag it. Do not remove that fixture.

**Job-source coverage varies sharply by city.** The Greenhouse boards skew
towards Bangalore and Mumbai; for tier-2 cities Adzuna carries nearly
everything. `scripts/verify_greenhouse.py` reports per-board India counts.

**Per-posting match scores are preliminary, and the UI says so.** Adzuna
truncates descriptions to ~500 characters, so a posting's keyword coverage is
computed over whatever skills fall inside that snippet. Postings that name too
few skills are flagged `LOW CONFIDENCE`, and those naming none are flagged
`NOT SCORED ON SKILLS` rather than shown as a misleading 0%.

`CLAUDE.md` documents the measurements behind each non-obvious scoring
decision, including several where the obvious approach was tried first and
found wrong on real data.

---

## Tech stack

Python 3.11 · FastAPI · uvicorn · Pydantic v2 · PyMuPDF · pdfplumber ·
docx2txt · sentence-transformers (all-MiniLM-L6-v2, CPU) · Groq ·
rapidfuzz · httpx · tenacity · pyjwt · bcrypt

Deliberately not used: LangChain, MongoDB, Docker, React, Firebase.
Storage is JSON files; the frontend is one HTML file.
