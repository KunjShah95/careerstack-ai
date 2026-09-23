"""Job discovery: fetch live postings for a saved analysis's role, rank
them against that resume, and persist the run.

Composition, not a new pipeline. The Adzuna client (roleprofile/adzuna.py),
the skill vocabulary and extractor (roleprofile/miner.py) and the
frequency-weighted keyword scorer (scoring/keywords.py) all already exist
and are reused as-is; nothing here re-implements them.

The one rule still holds: no LLM decides anything here. Query generation,
dedupe and ranking are all deterministic Python, so a run is reproducible
and explainable in a viva.
"""

import asyncio
import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone

from rapidfuzz import fuzz

from app.config import settings
from app.models.job import Job, JobMeta
from app.models.resume import ParsedResume
from app.services.extraction.parse_cache import load_cached_parse
from app.services.jobs import adzuna_source, greenhouse, jsearch
from app.services.jobs.base import default_source_label, source_rank
from app.services.roleprofile.miner import (
    SKILL_CATEGORIES,
    extract_skills_from_text,
    skill_category,
)
from app.services.scoring.keywords import keyword_score, normalize_skill
from app.store import list_keys, load_json, save_json

logger = logging.getLogger(__name__)

COLLECTION = "discovery_runs"
ANALYSES_COLLECTION = "analyses"

RESULTS_PER_PAGE = 50
MAX_PAGES = 3
TARGET_UNIQUE_JOBS = 50

# Below this many unique postings, a city-level search is too thin to
# browse -- widen to the whole country. Mirrors roleprofile/cache.py's
# LOCATION_FALLBACK_MIN_POSTINGS, which makes the same call for mining.
WIDEN_MIN_UNIQUE = 20
WIDEN_TARGET = "India"

# rapidfuzz threshold for pass-2 dedupe, applied to titles only and only
# once the company already matches exactly. See dedupe() for why the
# company is an exact gate rather than part of the fuzzed string.
NEAR_DUPLICATE_THRESHOLD = 88

# Corporate suffixes dropped before fingerprinting, so "Acme Pvt Ltd" and
# "Acme Technologies" collide on "acme".
_CORPORATE_SUFFIXES = [
    "private limited",
    "pvt ltd",
    "pvt",
    "ltd",
    "limited",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "technologies",
    "technology",
    "tech",
    "solutions",
    "services",
    "systems",
    "software",
    "labs",
    "consulting",
    "consultancy",
    "india",
    "global",
    "group",
    "co",
]
_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# Title abbreviations expanded before comparison. rapidfuzz alone can't
# close these: "sr software engineer" vs "senior software engineer" scores
# 82 on token_sort_ratio (below NEAR_DUPLICATE_THRESHOLD), because the
# abbreviation shares so few characters with the word it stands for.
# Raising the threshold to catch it would merge genuinely different jobs --
# two unrelated companies one token apart already score 90. So the fix is
# to normalise the abbreviation away first and leave the threshold strict.
_TITLE_ABBREVIATIONS = {
    "sr": "senior",
    "snr": "senior",
    "jr": "junior",
    "jnr": "junior",
    "sde": "software engineer",
    "swe": "software engineer",
    "dev": "developer",
    "engr": "engineer",
    "mgr": "manager",
    "admin": "administrator",
    "assoc": "associate",
    "ii": "2",
    "iii": "3",
}

_REMOTE_RE = re.compile(r"\b(remote|work from home|wfh|telecommute)\b", re.IGNORECASE)

# Recency bands, checked in order: (max age in days, score).
RECENCY_BANDS = [(3, 1.0), (7, 0.85), (14, 0.65), (30, 0.40)]
RECENCY_FLOOR = 0.15

LOCATION_FIT_REMOTE = 1.0
LOCATION_FIT_CITY = 1.0
LOCATION_FIT_STATE = 0.5
LOCATION_FIT_OTHER = 0.2

RANK_WEIGHTS = {"keyword": 0.70, "recency": 0.20, "location": 0.10}

# Below this many skills detected in a posting, its keyword coverage is
# not a trustworthy number. Measured, not guessed: on a live 65-posting
# run against Adzuna's ~497-char snippets, 65% of postings yielded <=1
# detectable skill, and every posting scoring 100% coverage did so off a
# single detected skill. Coverage over a 1-skill sample is a coin flip
# reported to two decimal places.
#
# The weights above are unchanged -- what changes is how much of the
# keyword term a thin posting gets to claim. A posting with >= this many
# skills keeps its full coverage score; below it, coverage is damped
# toward 0 in proportion to how little evidence there is (see
# _coverage_confidence). That keeps a well-described posting ranking on
# its merits without letting a one-word snippet outrank it.
MIN_SKILLS_FOR_FULL_CONFIDENCE = 3

# Coverage alone is a ratio, and a ratio rewards postings that name few
# requirements. Measured on a live 317-posting Bangalore run across Adzuna
# and Greenhouse: Greenhouse postings name 6.9 skills on average (5700-char
# descriptions) against Adzuna's 2.1 (500-char snippets), and matched more
# of the resume's skills in absolute terms (0.22 vs 0.07 per posting) --
# yet scored 12.1 average against Adzuna's 22.8, because matching 1 of 13
# named skills is 8% while matching 1 of 1 is 100%.
#
# That penalises the better-documented posting for being better documented.
# So the keyword term blends the ratio with an absolute-match component:
# how many of the resume's skills this posting actually wants, saturating
# at MATCH_COUNT_SATURATION. A posting naming one skill you have can still
# score, but it can no longer outrank one naming five you have.
MATCH_COUNT_SATURATION = 4
COVERAGE_RATIO_WEIGHT = 0.5
MATCH_COUNT_WEIGHT = 0.5


class DiscoveryUnavailableError(Exception):
    """DEMO_MODE is on and the requested run isn't cached. main.py maps
    this to a 503.
    """


class AnalysisNotFoundError(Exception):
    """No analysis with that id owned by this user. main.py maps this to a
    404.
    """


class ResumeUnavailableError(Exception):
    """The analysis predates resume_file_key, or its cached parse is gone,
    so there's no resume to score postings against. main.py maps this to a
    409 -- re-running the analysis fixes it.
    """


# --------------------------------------------------------------------------
# 1. Query generation -- deterministic, no LLM
# --------------------------------------------------------------------------

# A role's "obvious parent", for the broader query. Deliberately a small
# explicit table rather than anything clever: a wrong guess here spends
# Adzuna quota on irrelevant postings.
_ROLE_PARENTS = {
    "backend developer": "software developer",
    "backend engineer": "software engineer",
    "frontend developer": "software developer",
    "frontend engineer": "software engineer",
    "full stack developer": "software developer",
    "fullstack developer": "software developer",
    "mern developer": "full stack developer",
    "mean developer": "full stack developer",
    "android developer": "mobile developer",
    "ios developer": "mobile developer",
    "data scientist": "data analyst",
    "machine learning engineer": "data scientist",
    "ml engineer": "data scientist",
    "devops engineer": "software engineer",
    "qa engineer": "software engineer",
}

MAX_QUERIES = 5
TOP_SKILLS_IN_QUERIES = 2


def _top_technical_skills(resume: ParsedResume, limit: int) -> list[str]:
    """The resume's technical skills, in resume order, normalised and
    deduped.

    Resume order is the candidate's own ordering, which is a better
    relevance signal than anything we'd invent -- and it's stable, which
    deterministic query generation requires.
    """
    seen: set[str] = set()
    technical: list[str] = []
    for raw in resume.skills:
        skill = normalize_skill(raw)
        if not skill or skill in seen:
            continue
        seen.add(skill)
        # Must be a *known* vocabulary skill, not merely categorised as
        # technical. skill_category() fails open to "technical" for
        # anything it doesn't recognise, which is right for display but
        # wrong here: a resume listing "Deep Learning (CNNs, Transformers)"
        # would otherwise become the query "backend developer deep learning
        # cnns transformers", which matches nothing and wastes a call.
        if skill not in SKILL_CATEGORIES:
            continue
        if skill_category(skill) == "technical":
            technical.append(skill)
    return technical[:limit]


def generate_queries(role: str, resume: ParsedResume) -> list[str]:
    """3-5 Adzuna queries for this role and resume, deterministically.

    The same (role, resume) always yields the same list in the same order:
    the role verbatim, the role paired with each of the resume's top two
    technical skills, and the role's parent title if it has an obvious one.
    """
    role = role.strip()
    queries = [role]

    for skill in _top_technical_skills(resume, limit=TOP_SKILLS_IN_QUERIES):
        queries.append(f"{role} {skill}")

    parent = _ROLE_PARENTS.get(role.lower())
    if parent:
        queries.append(parent)

    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = query.lower()
        if key not in seen:
            seen.add(key)
            unique.append(query)
    return unique[:MAX_QUERIES]


# --------------------------------------------------------------------------
# 4. Normalisation and fingerprinting
# --------------------------------------------------------------------------


def _normalise_phrase(text: str | None) -> str:
    """lowercase, punctuation stripped, whitespace collapsed."""
    if not text:
        return ""
    depunctuated = _PUNCTUATION_RE.sub(" ", text.lower())
    return _WHITESPACE_RE.sub(" ", depunctuated).strip()


def _strip_corporate_suffixes(company: str | None) -> str:
    """Drop trailing corporate suffix words so "Acme Pvt Ltd" and "Acme
    Technologies" both reduce to "acme".

    Applied repeatedly, since suffixes stack ("Acme Technologies Pvt Ltd").
    Never strips a name down to nothing: a company literally called
    "Technology Solutions" keeps its normalised form rather than becoming
    "", which would otherwise collide with every other suffix-only name.
    """
    normalised = _normalise_phrase(company)
    changed = True
    while changed:
        changed = False
        for suffix in _CORPORATE_SUFFIXES:
            if normalised == suffix:
                continue
            if normalised.endswith(" " + suffix):
                candidate = normalised[: -(len(suffix) + 1)].strip()
                if candidate:
                    normalised = candidate
                    changed = True
                    break
    return normalised


def _city_of(location: str | None) -> str:
    """The city portion of an Adzuna location string ("Ahmedabad, Gujarat"
    -> "ahmedabad"). Adzuna orders these most-specific-first.
    """
    if not location:
        return ""
    return _normalise_phrase(location.split(",")[0])


def fingerprint(title: str | None, company: str | None, location: str | None) -> str:
    """sha1 of normalised title + company + city.

    Company normalisation drops corporate suffixes, so the same job posted
    under "Acme Pvt Ltd" and "Acme Technologies" produces one fingerprint.
    """
    raw = "|".join(
        [_normalise_phrase(title), _strip_corporate_suffixes(company), _city_of(location)]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _is_remote(posting: dict) -> bool:
    """Whether this posting is remote.

    A provider that states it outright wins: JSearch returns job_is_remote,
    which is the employer's own answer rather than our inference. Only when
    that's absent (None) do we fall back to sniffing the text, which is all
    Adzuna and Greenhouse give us.
    """
    stated = posting.get("is_remote")
    if stated is not None:
        return bool(stated)

    haystack = " ".join(
        str(posting.get(field) or "") for field in ("title", "location", "description")
    )
    return bool(_REMOTE_RE.search(haystack))


def normalise_posting(posting: dict) -> Job:
    """One raw Adzuna result -> a Job, before scoring."""
    has_salary = posting.get("salary_min") is not None or posting.get("salary_max") is not None
    return Job(
        fingerprint=fingerprint(
            posting.get("title"), posting.get("company"), posting.get("location")
        ),
        # Set by whichever adapter produced this posting (see
        # services/jobs/base.py). Defaults to adzuna so a dict built by
        # hand -- as the older tests do -- still validates.
        source=posting.get("source") or "adzuna",
        source_label=posting.get("source_label") or default_source_label(posting.get("source")),
        employment_type=posting.get("employment_type"),
        title=posting.get("title"),
        company=posting.get("company"),
        location=posting.get("location"),
        is_remote=_is_remote(posting),
        description=posting.get("description") or "",
        url=posting.get("url"),
        salary_min=posting.get("salary_min"),
        salary_max=posting.get("salary_max"),
        # Each adapter supplies its own currency where the provider gives
        # one; INR is the fallback because the Adzuna adapter targets the
        # India endpoint and that API returns no currency field at all.
        salary_currency=posting.get("salary_currency") or ("INR" if has_salary else None),
        posted_at=posting.get("created"),
    )


# --------------------------------------------------------------------------
# 5. Dedupe
# --------------------------------------------------------------------------


def _expand_title_abbreviations(title: str) -> str:
    """"Sr. SDE" -> "senior software engineer", so pass-2 dedupe compares
    like with like. See _TITLE_ABBREVIATIONS for why this is normalisation
    rather than a looser fuzzy threshold.
    """
    words = _normalise_phrase(title).split()
    return " ".join(_TITLE_ABBREVIATIONS.get(word, word) for word in words)


def _prefer(candidate: Job, incumbent: Job) -> bool:
    """Should `candidate` replace `incumbent` for the same fingerprint?

    Source priority first (jsearch > greenhouse > adzuna, see
    base.SOURCE_PRIORITY), then the fullest description. Both rules serve
    the same end: the per-posting keyword score is computed over the
    description, so the copy with the most text is the one worth keeping.

    Description length breaks ties *within* a source too, which matters
    when one Adzuna query returns a fuller snippet than another.
    """
    candidate_rank = source_rank(candidate.source)
    incumbent_rank = source_rank(incumbent.source)
    if candidate_rank != incumbent_rank:
        return candidate_rank < incumbent_rank
    return len(candidate.description or "") > len(incumbent.description or "")


def dedupe(jobs: list[Job]) -> tuple[list[Job], int]:
    """Two-pass dedupe. Returns (unique jobs, number removed).

    Pass 1 is exact fingerprint. Pass 2 is rapidfuzz token_sort_ratio on
    "title company", which catches abbreviation variants ("Sr. SDE" vs
    "Senior Software Engineer") that normalise to different fingerprints.
    Input order is preserved, so the first-seen posting wins.
    """
    by_fingerprint: dict[str, Job] = {}
    exact_duplicates = 0
    for job in jobs:
        incumbent = by_fingerprint.get(job.fingerprint)
        if incumbent is not None:
            exact_duplicates += 1
            # Same posting from two sources: keep the better one rather
            # than whichever arrived first. See _prefer.
            if _prefer(job, incumbent):
                by_fingerprint[job.fingerprint] = job
            continue
        by_fingerprint[job.fingerprint] = job

    # Best-first, so pass 2 keeps the preferred copy of a near-duplicate
    # instead of whichever the dict happened to yield first.
    ordered = sorted(
        by_fingerprint.values(),
        key=lambda job: (source_rank(job.source), -len(job.description or "")),
    )

    kept: list[Job] = []
    kept_keys: list[tuple[str, str]] = []
    near_duplicates = 0
    for job in ordered:
        company = _strip_corporate_suffixes(job.company)
        title = _expand_title_abbreviations(job.title or "")
        # The company must match exactly before titles are compared at all.
        # Fuzzing the whole "title company" string instead lets two
        # different employers with similar names merge -- "Dev Wide 0" vs
        # "Dev Wide 1" scores 90 on token_sort_ratio, above any threshold
        # loose enough to be useful on titles. Two postings are only the
        # same job if they're at the same company.
        is_duplicate = any(
            company == seen_company
            and fuzz.token_sort_ratio(title, seen_title) >= NEAR_DUPLICATE_THRESHOLD
            for seen_title, seen_company in kept_keys
        )
        if title and is_duplicate:
            near_duplicates += 1
            continue
        kept.append(job)
        kept_keys.append((title, company))

    return kept, exact_duplicates + near_duplicates


# --------------------------------------------------------------------------
# 6. Ranking
# --------------------------------------------------------------------------


def _parse_posted_at(posted_at: str | None) -> datetime | None:
    if not posted_at:
        return None
    try:
        parsed = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def recency_score(posted_at: str | None, now: datetime | None = None) -> float:
    """1.0 for <=3 days old, decaying to RECENCY_FLOOR beyond 30 days.

    A missing or unparseable date gets the floor rather than a guess: we
    don't know it's fresh, so it doesn't get to rank as though it were.
    """
    parsed = _parse_posted_at(posted_at)
    if parsed is None:
        return RECENCY_FLOOR

    now = now or datetime.now(timezone.utc)
    age_days = (now - parsed).total_seconds() / 86400
    for max_days, score in RECENCY_BANDS:
        if age_days <= max_days:
            return score
    return RECENCY_FLOOR


def location_fit_score(job: Job, target_location: str) -> float:
    """Remote or same-city is a full match, same-state is half, anything
    else scores LOCATION_FIT_OTHER rather than zero -- a distant posting is
    worth less, not worthless.
    """
    if job.is_remote:
        return LOCATION_FIT_REMOTE

    target = _normalise_phrase(target_location)
    if not target or target == _normalise_phrase(WIDEN_TARGET):
        # Country-wide search: there's no city to be nearer to or further
        # from, so location can't discriminate between these postings.
        return LOCATION_FIT_CITY

    job_parts = [_normalise_phrase(part) for part in (job.location or "").split(",")]
    job_parts = [part for part in job_parts if part]
    if not job_parts:
        return LOCATION_FIT_OTHER

    if job_parts[0] == target:
        return LOCATION_FIT_CITY
    if target in job_parts[1:]:
        return LOCATION_FIT_STATE
    return LOCATION_FIT_OTHER


def posting_skill_frequencies(job: Job) -> dict[str, dict]:
    """The skills one posting names, shaped like a role profile's
    skill_frequencies so keyword_score scores against it unchanged.

    Every skill in a single posting carries frequency 1.0: within one
    posting there's no frequency distribution to weight by -- a skill is
    either named or it isn't. That makes the per-posting keyword score a
    plain proportion of that posting's named skills that the resume
    covers, which is the honest reading of the number.

    Scans the title alongside the description because Adzuna truncates only
    the description, and the title routinely names the core technology
    ("React Native Developer").
    """
    haystack = "\n".join([job.title or "", job.company or "", job.description or ""])
    return {skill: {"frequency": 1.0, "count": 1} for skill in extract_skills_from_text(haystack)}


def _coverage_confidence(skills_detected: int) -> float:
    """How much of a posting's keyword coverage we're willing to believe,
    given how many skills its (truncated) text actually named.

    Linear from 0.0 at no skills to 1.0 at MIN_SKILLS_FOR_FULL_CONFIDENCE.
    A posting naming one skill gets a third of the weight of one naming
    three, so "100% of the single word we could read" can't outrank a
    posting we genuinely understood. See MIN_SKILLS_FOR_FULL_CONFIDENCE
    for the measurements behind this.
    """
    if skills_detected <= 0:
        return 0.0
    return min(skills_detected / MIN_SKILLS_FOR_FULL_CONFIDENCE, 1.0)


def score_job(
    job: Job,
    resume_text: str,
    resume_skills: set[str],
    target_location: str,
    now: datetime | None = None,
) -> Job:
    """Attach match_score and its component breakdown to one job.

        match = 0.70 * keyword_coverage_vs_this_posting
              + 0.20 * recency
              + 0.10 * location_fit

    The keyword term reuses scoring/keywords.py against a skill table built
    from this posting alone, so the very same engine that scores a resume
    against a role profile scores it against a single posting.

    A posting whose snippet names no known skill at all gets a keyword
    component of 0.0 and meta.skills_detected_in_posting == 0; the UI reads
    those as unscored, not as a genuine zero.
    """
    frequencies = posting_skill_frequencies(job)
    keyword_detail = keyword_score(frequencies, resume_text, resume_skills)

    recency = recency_score(job.posted_at, now=now)
    location_fit = location_fit_score(job, target_location)
    confidence = _coverage_confidence(len(frequencies))

    # Ratio damped by confidence, blended with absolute matches. See
    # MATCH_COUNT_SATURATION for the measurements behind the blend.
    match_count = len(keyword_detail["matched"])
    match_count_component = min(match_count / MATCH_COUNT_SATURATION, 1.0)
    keyword_component = (
        COVERAGE_RATIO_WEIGHT * keyword_detail["score"] * confidence
        + MATCH_COUNT_WEIGHT * match_count_component
    )

    match = (
        RANK_WEIGHTS["keyword"] * keyword_component
        + RANK_WEIGHTS["recency"] * recency
        + RANK_WEIGHTS["location"] * location_fit
    )

    description = job.description or ""
    job.match_score = round(match * 100, 2)
    job.matched_skills = [item["skill"] for item in keyword_detail["matched"]]
    job.missing_skills = [item["skill"] for item in keyword_detail["missing"]]
    job.score_breakdown = {
        "keyword_coverage": keyword_detail["score"],
        # What the keyword term actually contributed, so the composite is
        # reproducible by hand from this dict alone.
        "keyword_confidence": confidence,
        "matched_count": match_count,
        "match_count_component": round(match_count_component, 4),
        "keyword_effective": round(keyword_component, 4),
        "recency": recency,
        "location_fit": location_fit,
        "weights": dict(RANK_WEIGHTS),
    }
    job.meta = JobMeta(
        # Adzuna's free tier always truncates the description, so this is
        # flagged per job rather than inferred from length -- the UI says
        # the score is preliminary instead of quietly implying we read the
        # whole posting.
        description_truncated=True,
        description_chars=len(description),
        skills_detected_in_posting=len(frequencies),
        # Only postings that named *something* are "low confidence"; one
        # that named nothing is a separate case (skills_unscored), since
        # there's no coverage figure to be less confident about.
        low_confidence=0 < len(frequencies) < MIN_SKILLS_FOR_FULL_CONFIDENCE,
        skills_unscored=not frequencies,
    )
    return job


def rank_jobs(
    jobs: list[Job],
    resume_text: str,
    resume_skills: set[str],
    target_location: str,
    now: datetime | None = None,
) -> list[Job]:
    """Score every job and sort by match_score descending.

    Ties break on recency then title, so the order is stable across runs
    rather than falling back to whatever order Adzuna happened to return.
    """
    scored = [score_job(job, resume_text, resume_skills, target_location, now=now) for job in jobs]
    scored.sort(
        key=lambda job: (
            -job.match_score,
            -job.score_breakdown["recency"],
            job.title or "",
        )
    )
    return scored


# --------------------------------------------------------------------------
# 2-3. Fetch and location widening
# --------------------------------------------------------------------------


SOURCE_CACHE_COLLECTION = "job_source_cache"
SOURCE_CACHE_TTL_HOURS = 6


def _source_cache_key(source: str, role: str, location: str) -> str:
    raw = f"{source}|{role.lower().strip()}|{location.lower().strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cached_source_postings(source: str, role: str, location: str) -> list[dict] | None:
    """Cached postings for one source, or None if absent or stale.

    Cached per source, not per run, so a JSearch outage or an exhausted
    quota doesn't invalidate the Adzuna and Greenhouse results sitting
    next to it -- each source's freshness stands on its own.
    """
    entry = load_json(SOURCE_CACHE_COLLECTION, _source_cache_key(source, role, location))
    if not entry:
        return None

    try:
        cached_at = datetime.fromisoformat(entry["cached_at"])
    except (KeyError, ValueError):
        return None
    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=timezone.utc)

    age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
    if age_hours >= SOURCE_CACHE_TTL_HOURS:
        return None
    return entry.get("postings", [])


def _store_source_postings(source: str, role: str, location: str, postings: list[dict]) -> None:
    save_json(
        SOURCE_CACHE_COLLECTION,
        _source_cache_key(source, role, location),
        {
            "source": source,
            "role": role,
            "location": location,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "postings": postings,
        },
    )


async def _cached_fetch(source: str, role: str, location: str, fetcher) -> list[dict]:
    """Serve one source from cache when fresh, otherwise fetch and store.

    An empty live result is deliberately not cached: an empty list is
    usually a transient failure (rate limit, timeout) that the adapter
    already swallowed, and caching it would lock that source out for six
    hours over a blip.
    """
    cached = _cached_source_postings(source, role, location)
    if cached is not None:
        logger.info("Serving %s from cache for role=%r location=%r", source, role, location)
        return cached

    postings = await fetcher()
    if postings:
        _store_source_postings(source, role, location, postings)
    return postings


async def _fetch_adzuna(queries: list[str], location: str) -> list[dict]:
    """Adzuna gets the full generated query set -- it's the broad, cheap
    source, and title-phrasing variety is where its value is.
    """
    postings: list[dict] = []
    for query in queries:
        postings.extend(await adzuna_source.fetch(query, location))
        if len(postings) >= TARGET_UNIQUE_JOBS * 2:
            # Enough raw material; dedupe will cut this down anyway and
            # further queries cost calls for diminishing returns.
            break
    return postings


async def _fetch_jsearch(role: str, location: str) -> list[dict]:
    """One JSearch call for the role itself, not per query variant.

    The free tier is ~200 calls/month. Fanning the whole query set at it
    would exhaust that in a couple of dozen runs, so it gets the single
    highest-value query and nothing more.
    """
    return await jsearch.fetch(role, location)


async def _gather_sources(role: str, queries: list[str], location: str) -> tuple[list[dict], dict]:
    """Fan out across all three sources concurrently.

    Returns (raw postings, per-source counts). Every source is wrapped in
    return_exceptions=True on top of each adapter's own never-raise
    contract: a source that fails contributes nothing and the other two
    still produce a run.
    """
    tasks = {
        "adzuna": _cached_fetch(
            "adzuna", role, location, lambda: _fetch_adzuna(queries, location)
        ),
        "jsearch": _cached_fetch(
            "jsearch", role, location, lambda: _fetch_jsearch(role, location)
        ),
        "greenhouse": _cached_fetch(
            "greenhouse", role, location, lambda: greenhouse.fetch(role, location)
        ),
    }

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    postings: list[dict] = []
    counts: dict[str, int] = {}
    for source_name, result in zip(tasks.keys(), results):
        if isinstance(result, BaseException):
            logger.warning("Job source %r failed: %s", source_name, result)
            counts[source_name] = 0
            continue
        counts[source_name] = len(result)
        postings.extend(result)

    return postings, counts


async def _fetch_queries(
    queries: list[str], location: str, role: str | None = None
) -> tuple[list[Job], dict, int]:
    """All sources for this role/location, normalised into Jobs.

    Returns (jobs, per-source counts, total raw postings fetched). `role`
    defaults to the first query, which is the role verbatim -- see
    generate_queries.
    """
    role = role or (queries[0] if queries else "")
    postings, counts = await _gather_sources(role, queries, location)
    jobs = [normalise_posting(posting) for posting in postings]
    return jobs, counts, len(postings)


def _widen_reason(unique_count: int, location: str) -> str:
    postings = "posting" if unique_count == 1 else "postings"
    return f"only {unique_count} {postings} found in {location}"


# --------------------------------------------------------------------------
# 7. Run orchestration and persistence
# --------------------------------------------------------------------------


def _resume_for_analysis(analysis: dict) -> ParsedResume:
    """The ParsedResume behind a saved analysis.

    An analysis record stores scoring output, not the resume, so this goes
    through resume_file_key (stamped on by analyze.py) into the parse
    cache. Records written before that field existed can't be resolved --
    re-running the analysis fixes it, which is what the error says.
    """
    key = analysis.get("resume_file_key")
    if not key:
        raise ResumeUnavailableError(
            "This analysis was saved before resumes were linked to their parsed "
            "data, so we can't match jobs against it. Re-run the analysis on the "
            "same file and try again."
        )

    resume = load_cached_parse(key)
    if resume is None:
        raise ResumeUnavailableError(
            "The parsed data for this resume is no longer cached. Re-run the "
            "analysis on the same file and try again."
        )
    return resume


def _resume_search_text(resume: ParsedResume) -> str:
    """Text for keyword matching's tier-2 (word-boundary regex) pass.

    The original resume text isn't stored on an analysis, so this
    reconstructs a searchable body from the parsed resume: skills, summary,
    experience bullets and project descriptions. That's where skill
    mentions actually live, so tier 2 keeps working -- it just reads the
    structured version rather than the raw page.
    """
    parts: list[str] = list(resume.skills)
    if resume.summary:
        parts.append(resume.summary)
    for experience in resume.experience:
        parts.extend(filter(None, [experience.title, experience.company]))
        parts.extend(experience.bullets)
    for project in resume.projects:
        parts.extend(filter(None, [project.name, project.description]))
        parts.extend(project.technologies)
    parts.extend(resume.certifications)
    return "\n".join(part for part in parts if part)


def load_run(run_id: str, user_id: str) -> dict | None:
    """A discovery run, if it exists and belongs to this user.

    Same 404-for-both-cases ownership rule as analyses: returning None for
    "someone else's run" rather than a distinct error means the API can't
    be used to confirm a run id exists.
    """
    run = load_json(COLLECTION, run_id)
    if run is None or run.get("user_id") != user_id:
        return None
    return run


def _save(run: dict) -> None:
    save_json(COLLECTION, run["run_id"], run)


def create_run(analysis_id: str, user_id: str) -> dict:
    """Create and persist a pending run for this user's analysis.

    Validates ownership and resume availability up front, so POST fails
    fast with a real status code instead of returning a run_id that's
    guaranteed to fail in the background a moment later.
    """
    analysis = load_json(ANALYSES_COLLECTION, analysis_id)
    if analysis is None or analysis.get("user_id") != user_id:
        raise AnalysisNotFoundError("No analysis found with that ID.")

    _resume_for_analysis(analysis)  # raises ResumeUnavailableError if missing

    run = {
        "run_id": uuid.uuid4().hex,
        "user_id": user_id,
        "analysis_id": analysis_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "target": {"role": analysis.get("role"), "location": analysis.get("location")},
        "stats": {"requested": 0, "fetched": 0, "unique": 0, "ranked": 0, "duplicate_rate": 0.0},
        "widened": None,
        "jobs": [],
        "error": None,
    }
    _save(run)
    return run


async def execute_run(run_id: str) -> dict:
    """Run the discovery pipeline for an existing pending run and persist
    the result.

    Every failure is caught and written to the run as status="failed" with
    a plain-language message -- a background task that raises would
    otherwise leave the run stuck on "running" forever with nothing to
    show the user.
    """
    run = load_json(COLLECTION, run_id)
    if run is None:
        logger.warning("execute_run called for unknown run_id=%r", run_id)
        return {}

    run["status"] = "running"
    _save(run)

    try:
        analysis = load_json(ANALYSES_COLLECTION, run["analysis_id"])
        if analysis is None:
            raise AnalysisNotFoundError("No analysis found with that ID.")

        resume = _resume_for_analysis(analysis)
        resume_text = _resume_search_text(resume)
        resume_skills = set(resume.skills)

        role = run["target"]["role"] or ""
        location = run["target"]["location"] or WIDEN_TARGET

        if settings.demo_mode:
            # Serve a warmed run for this (role, location) rather than
            # hitting Adzuna -- same contract as roleprofile/cache.py.
            # Copies the cached run's results onto this run rather than
            # returning the old record, so the run_id the client is
            # already polling is the one that completes.
            cached = find_cached_run(run["user_id"], role, location, exclude_run_id=run_id)
            if cached is None:
                raise DiscoveryUnavailableError(
                    f"DEMO_MODE is on and no cached discovery run exists for "
                    f"role={role!r}, location={location!r}. Warm it with "
                    f"scripts/prep_demo.py (DEMO_MODE off) before relying on it "
                    f"in demo mode."
                )
            run["stats"] = cached["stats"]
            run["queries"] = cached.get("queries", [])
            run["widened"] = cached.get("widened")
            run["jobs"] = cached.get("jobs", [])
            run["status"] = "complete"
            run["completed_at"] = datetime.now(timezone.utc).isoformat()
            run["served_from_cache"] = True
            _save(run)
            return run

        queries = generate_queries(role, resume)
        jobs, source_counts, fetched = await _fetch_queries(queries, location, role=role)
        unique_jobs, duplicates = dedupe(jobs)

        widened = None
        search_location = location
        if len(unique_jobs) < WIDEN_MIN_UNIQUE and location.strip().lower() != WIDEN_TARGET.lower():
            reason = _widen_reason(len(unique_jobs), location)
            wider_jobs, wider_counts, wider_fetched = await _fetch_queries(
                queries, WIDEN_TARGET, role=role
            )
            fetched += wider_fetched
            for source_name, count in wider_counts.items():
                source_counts[source_name] = source_counts.get(source_name, 0) + count
            unique_jobs, duplicates = dedupe(jobs + wider_jobs)
            search_location = WIDEN_TARGET
            widened = {"from": location, "to": WIDEN_TARGET, "reason": reason}

        ranked = rank_jobs(unique_jobs, resume_text, resume_skills, search_location)

        total_seen = len(unique_jobs) + duplicates
        run["stats"] = {
            # Kept for the existing UI copy ("N unique from M fetched").
            "requested": len(queries),
            "fetched": fetched,
            "unique": len(unique_jobs),
            "ranked": len(ranked),
            "duplicate_rate": round(duplicates / total_seen, 4) if total_seen else 0.0,
            # Per-source contribution, before dedupe, plus what survived it.
            "by_source": {**source_counts, "after_dedupe": len(unique_jobs)},
        }
        run["queries"] = queries
        run["widened"] = widened
        run["jobs"] = [job.model_dump() for job in ranked]
        run["status"] = "complete"
        run["completed_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as exc:  # noqa: BLE001 -- see docstring
        logger.exception("Discovery run %s failed", run_id)
        run["status"] = "failed"
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        run["error"] = str(exc) or "Something went wrong while finding matching jobs."

    _save(run)
    return run


def get_job(run: dict, job_fingerprint: str) -> dict | None:
    for job in run.get("jobs", []):
        if job.get("fingerprint") == job_fingerprint:
            return job
    return None


def rescore_job_against_posting(run: dict, job_fingerprint: str) -> dict | None:
    """One job re-scored against its own posting text, for the per-posting
    detail view.

    This recomputes rather than trusting the stored number, so the detail
    panel is a genuine re-derivation from the posting text -- if the two
    ever disagreed, that's a bug worth seeing, not one worth hiding.
    """
    stored = get_job(run, job_fingerprint)
    if stored is None:
        return None

    analysis = load_json(ANALYSES_COLLECTION, run["analysis_id"])
    if analysis is None:
        return stored

    resume = _resume_for_analysis(analysis)
    job = Job.model_validate(stored)
    target_location = (run.get("widened") or {}).get("to") or run["target"]["location"] or ""

    scored = score_job(
        job, _resume_search_text(resume), set(resume.skills), target_location
    )
    return scored.model_dump()


def find_cached_run(
    user_id: str, role: str, location: str, exclude_run_id: str | None = None
) -> dict | None:
    """The most recent complete run for (role, location), preferring one
    this user already owns.

    Used by DEMO_MODE to serve a warmed run instead of hitting the network,
    the same way roleprofile/cache.py serves a warmed profile.
    exclude_run_id keeps a run from matching itself when it's the one
    currently being executed.

    Deliberately NOT scoped to user_id alone. prep_demo.py warms its runs
    under the demo account, so scoping strictly by owner meant anyone
    presenting from their own login got "no cached run exists" -- the exact
    failure DEMO_MODE is supposed to make impossible. This is not an
    ownership hole: it only ever copies posting data (which is public job
    listings from Adzuna, keyed on role and location, not anything derived
    from another user's resume) into the caller's own new run, and it only
    runs in DEMO_MODE. Reading an existing run still goes through
    load_run(), which enforces ownership strictly.

    A run the caller already owns still wins, so a user who has run this
    role/location themselves sees their own results.
    """
    candidates = []
    for key in list_keys(COLLECTION):
        if key == exclude_run_id:
            continue
        run = load_json(COLLECTION, key)
        if not run or run.get("status") != "complete":
            continue
        target = run.get("target") or {}
        if (target.get("role") or "").lower() != role.lower():
            continue
        if (target.get("location") or "").lower() != location.lower():
            continue
        candidates.append(run)

    if not candidates:
        return None
    # The caller's own run wins over anyone else's, then most recent.
    return max(
        candidates,
        key=lambda run: (run.get("user_id") == user_id, run.get("completed_at") or ""),
    )
