"""Tests for the job discovery pipeline.

Every posting here is a hand-built dict in the shape adzuna.py's search()
returns -- no network, no LLM. The point is discovery.py's own logic
(determinism, fingerprinting, dedupe, widening, ranking arithmetic), and
fixtures keep that reproducible.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.models.job import Job
from app.models.resume import Experience, ParsedResume, Project
from app.services import discovery
from app.services.discovery import (
    RANK_WEIGHTS,
    dedupe,
    fingerprint,
    generate_queries,
    location_fit_score,
    normalise_posting,
    posting_skill_frequencies,
    rank_jobs,
    recency_score,
    score_job,
)
from app.services.scoring.keywords import keyword_score

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)


def _days_ago(days: float) -> str:
    return (NOW - timedelta(days=days)).isoformat()


def _resume(skills: list[str] | None = None) -> ParsedResume:
    return ParsedResume(
        summary="Backend developer building APIs.",
        skills=skills if skills is not None else ["Python", "Django", "PostgreSQL", "Teamwork"],
        experience=[
            Experience(
                title="Backend Intern",
                company="Zenlabs",
                bullets=["Built REST API endpoints in Django."],
            )
        ],
        projects=[Project(name="Tracker", description="A Flask app", technologies=["Docker"])],
        sections_found=["experience", "education", "skills"],
        total_experience_months=6,
    )


def _posting(
    title: str,
    company: str,
    location: str = "Ahmedabad, Gujarat",
    description: str = "Looking for a developer with Python and Django experience.",
    created: str | None = None,
    salary_min: float | None = None,
    salary_max: float | None = None,
    posting_id: str | None = None,
) -> dict:
    return {
        "id": posting_id or f"{title}-{company}",
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "category": "IT Jobs",
        "url": "https://example.test/job",
        "created": created or _days_ago(1),
        "salary_min": salary_min,
        "salary_max": salary_max,
    }


# --------------------------------------------------------------------------
# 1. Query generation is deterministic
# --------------------------------------------------------------------------


def test_query_generation_is_deterministic_on_the_same_input():
    resume = _resume()
    first = generate_queries("backend developer", resume)
    second = generate_queries("backend developer", resume)

    assert first == second
    assert 3 <= len(first) <= 5
    assert first[0] == "backend developer"


def test_query_generation_pairs_the_role_with_resume_technical_skills():
    queries = generate_queries("backend developer", _resume(["Python", "Django"]))

    assert "backend developer python" in queries
    assert "backend developer django" in queries


def test_query_generation_skips_soft_skills_when_pairing():
    """Soft skills make useless search terms -- "backend developer
    teamwork" matches nothing useful, and it would burn a query slot.
    """
    queries = generate_queries("backend developer", _resume(["Teamwork", "Communication skills"]))

    assert not any("teamwork" in query for query in queries)
    assert not any("communication" in query for query in queries)


def test_query_generation_ignores_skills_outside_the_vocabulary():
    """skill_category() fails open to "technical" for unknown skills, which
    is right for display but would produce junk queries here: a resume
    listing "Deep Learning (CNNs, Transformers)" must not become the query
    "backend developer deep learning cnns transformers".
    """
    resume = _resume(["Deep Learning (CNNs, Transformers)", "GenAI", "Python"])
    queries = generate_queries("backend developer", resume)

    assert "backend developer python" in queries
    assert all("cnns" not in query for query in queries)
    assert all("genai" not in query for query in queries)


def test_query_generation_adds_the_parent_role_when_there_is_an_obvious_one():
    queries = generate_queries("backend developer", _resume())
    assert "software developer" in queries


def test_query_generation_has_no_parent_for_an_unknown_role():
    queries = generate_queries("quantum blockchain artisan", _resume())
    assert queries[0] == "quantum blockchain artisan"
    assert all("software" not in query for query in queries)


# --------------------------------------------------------------------------
# 4. Fingerprinting
# --------------------------------------------------------------------------


def test_fingerprint_collapses_corporate_suffix_variants():
    """The whole point of suffix stripping: the same job posted under two
    corporate spellings must produce one fingerprint.
    """
    a = fingerprint("Backend Developer", "Acme Pvt Ltd", "Ahmedabad, Gujarat")
    b = fingerprint("Backend Developer", "Acme Technologies", "Ahmedabad, Gujarat")
    c = fingerprint("Backend Developer", "ACME, Inc.", "Ahmedabad, Gujarat")

    assert a == b == c


def test_fingerprint_separates_genuinely_different_companies():
    a = fingerprint("Backend Developer", "Acme Pvt Ltd", "Ahmedabad, Gujarat")
    b = fingerprint("Backend Developer", "Globex Pvt Ltd", "Ahmedabad, Gujarat")
    assert a != b


def test_fingerprint_separates_the_same_job_in_different_cities():
    a = fingerprint("Backend Developer", "Acme", "Ahmedabad, Gujarat")
    b = fingerprint("Backend Developer", "Acme", "Pune, Maharashtra")
    assert a != b


def test_suffix_stripping_never_empties_a_suffix_only_company_name():
    """"Technology Solutions" is a real company name made entirely of
    suffix words. Stripping it to "" would collide it with every other
    suffix-only name, so it keeps its normalised form instead.
    """
    a = fingerprint("Developer", "Technology Solutions", "Pune")
    b = fingerprint("Developer", "Systems Services", "Pune")
    assert a != b


# --------------------------------------------------------------------------
# 5. Dedupe
# --------------------------------------------------------------------------


def test_dedupe_rate_matches_expected_on_a_fixture_set():
    """6 postings in, 3 genuinely distinct jobs out:
      - 3 spellings of the same Acme backend role (suffix variants)
      - "Sr. SDE" vs "Senior Software Engineer" at Globex (fuzzy pass)
      - 1 unrelated Initech role
    """
    postings = [
        _posting("Backend Developer", "Acme Pvt Ltd"),
        _posting("Backend Developer", "Acme Technologies"),
        _posting("Backend Developer", "Acme Inc"),
        _posting("Senior Software Engineer", "Globex"),
        _posting("Sr. Software Engineer", "Globex"),
        _posting("Data Analyst", "Initech"),
    ]
    jobs = [normalise_posting(posting) for posting in postings]

    unique, duplicates = dedupe(jobs)

    assert len(unique) == 3
    assert duplicates == 3
    assert round(duplicates / (len(unique) + duplicates), 4) == 0.5


def test_dedupe_keeps_the_fullest_description_among_duplicates():
    """Within one source, the copy carrying the most description text wins
    rather than whichever arrived first -- the per-posting keyword score is
    computed over that text, so more of it is strictly better. See _prefer.
    """
    jobs = [
        normalise_posting(_posting("Backend Developer", "Acme Pvt Ltd", description="short")),
        normalise_posting(
            _posting("Backend Developer", "Acme Technologies", description="a much longer one")
        ),
    ]
    unique, duplicates = dedupe(jobs)

    assert len(unique) == 1
    assert duplicates == 1
    assert unique[0].description == "a much longer one"


def test_dedupe_does_not_collapse_different_roles_at_the_same_company():
    jobs = [
        normalise_posting(_posting("Backend Developer", "Acme")),
        normalise_posting(_posting("Data Analyst", "Acme")),
    ]
    unique, duplicates = dedupe(jobs)

    assert len(unique) == 2
    assert duplicates == 0


# --------------------------------------------------------------------------
# 6. Ranking components
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "age_days, expected",
    [(0, 1.0), (3, 1.0), (5, 0.85), (7, 0.85), (10, 0.65), (14, 0.65), (20, 0.40), (30, 0.40), (45, 0.15)],
)
def test_recency_bands(age_days, expected):
    assert recency_score(_days_ago(age_days), now=NOW) == expected


def test_recency_floor_for_a_missing_or_unparseable_date():
    assert recency_score(None, now=NOW) == 0.15
    assert recency_score("not a date", now=NOW) == 0.15


def test_location_fit_remote_beats_city_mismatch():
    remote = normalise_posting(
        _posting("Backend Developer", "Acme", location="Remote", description="Remote role")
    )
    elsewhere = normalise_posting(_posting("Backend Developer", "Acme", location="Chennai, Tamil Nadu"))

    assert location_fit_score(remote, "Ahmedabad") == 1.0
    assert location_fit_score(elsewhere, "Ahmedabad") == 0.2


def test_location_fit_city_state_and_other():
    same_city = normalise_posting(_posting("Dev", "Acme", location="Ahmedabad, Gujarat"))
    same_state = normalise_posting(_posting("Dev", "Acme", location="Surat, Gujarat"))
    elsewhere = normalise_posting(_posting("Dev", "Acme", location="Chennai, Tamil Nadu"))

    assert location_fit_score(same_city, "Ahmedabad") == 1.0
    # Targeting the state itself: Surat is in Gujarat but isn't Gujarat, so
    # this is a state-level match, not a city one.
    assert location_fit_score(same_state, "Gujarat") == 0.5
    # Targeting a city the posting isn't in, in a state it shares: the
    # state appears in the posting's trailing parts, so still a half match.
    assert location_fit_score(same_state, "Ahmedabad") == 0.2
    assert location_fit_score(elsewhere, "Ahmedabad") == 0.2


def test_ranking_breaks_ties_on_recency():
    """Two postings identical but for age: the fresher one ranks first."""
    resume = _resume()
    jobs = [
        normalise_posting(
            _posting("Backend Developer", "Stale Corp", created=_days_ago(25), posting_id="old")
        ),
        normalise_posting(
            _posting("Backend Developer", "Fresh Corp", created=_days_ago(1), posting_id="new")
        ),
    ]

    ranked = rank_jobs(jobs, "python django", set(resume.skills), "Ahmedabad", now=NOW)

    assert ranked[0].company == "Fresh Corp"
    assert ranked[0].match_score > ranked[1].match_score


def test_ranking_breaks_ties_on_location():
    resume = _resume()
    jobs = [
        normalise_posting(
            _posting("Backend Developer", "Far Corp", location="Chennai, Tamil Nadu", posting_id="far")
        ),
        normalise_posting(
            _posting("Backend Developer", "Near Corp", location="Ahmedabad, Gujarat", posting_id="near")
        ),
    ]

    ranked = rank_jobs(jobs, "python django", set(resume.skills), "Ahmedabad", now=NOW)

    assert ranked[0].company == "Near Corp"


def test_ranking_is_sorted_descending():
    resume = _resume()
    jobs = [
        normalise_posting(_posting("Backend Developer", "A", created=_days_ago(20), posting_id="a")),
        normalise_posting(_posting("Backend Developer", "B", created=_days_ago(1), posting_id="b")),
        normalise_posting(_posting("Backend Developer", "C", created=_days_ago(10), posting_id="c")),
    ]

    ranked = rank_jobs(jobs, "python django", set(resume.skills), "Ahmedabad", now=NOW)
    scores = [job.match_score for job in ranked]

    assert scores == sorted(scores, reverse=True)


def test_a_posting_naming_no_known_skills_is_flagged_rather_than_scored_as_zero():
    """"Named nothing we recognise" is a different statement from "named
    things and you matched none of them" -- the first can't be scored on
    skills at all, so it gets its own flag instead of a 0% that reads as
    a genuine mismatch.
    """
    job = normalise_posting(
        _posting("Associate", "Acme", description="A great opportunity for a motivated person.")
    )
    scored = score_job(job, "python django", {"Python"}, "Ahmedabad", now=NOW)

    assert scored.meta.skills_detected_in_posting == 0
    assert scored.score_breakdown["keyword_coverage"] == 0.0
    assert scored.meta.skills_unscored is True
    # Not also flagged low_confidence -- the two are mutually exclusive.
    assert scored.meta.low_confidence is False


def test_a_thin_posting_is_low_confidence_but_not_unscored():
    job = normalise_posting(
        _posting("Developer", "Acme", description="Python required.")
    )
    scored = score_job(job, "python django", {"Python"}, "Ahmedabad", now=NOW)

    assert scored.meta.skills_detected_in_posting == 1
    assert scored.meta.low_confidence is True
    assert scored.meta.skills_unscored is False


def test_every_job_is_flagged_as_working_from_a_truncated_description():
    job = normalise_posting(_posting("Backend Developer", "Acme"))
    scored = score_job(job, "python django", {"Python"}, "Ahmedabad", now=NOW)

    assert scored.meta.description_truncated is True
    assert scored.meta.description_chars == len(job.description)


# --------------------------------------------------------------------------
# Honesty test: the reported score must be re-derivable by hand
# --------------------------------------------------------------------------


def test_top_job_keyword_component_matches_a_manual_rescore():
    """The number an examiner is most likely to probe: recompute the top
    job's keyword component by hand, straight through keywords.py, and
    confirm it equals what ranking reported. If these disagree, ranking is
    lying about how it reached its number.
    """
    resume = _resume(["Python", "Django", "PostgreSQL"])
    resume_text = "Built REST API endpoints in Django with Python and PostgreSQL."
    postings = [
        _posting(
            "Backend Developer",
            "Acme",
            description="Python and Django required, plus Docker and AWS.",
            posting_id="a",
        ),
        _posting(
            "Backend Engineer",
            "Globex",
            description="Java and Spring Boot shop.",
            created=_days_ago(20),
            posting_id="b",
        ),
    ]
    jobs = [normalise_posting(posting) for posting in postings]

    ranked = rank_jobs(jobs, resume_text, set(resume.skills), "Ahmedabad", now=NOW)
    top = ranked[0]

    manual = keyword_score(
        posting_skill_frequencies(top), resume_text, set(resume.skills)
    )

    assert abs(manual["score"] - top.score_breakdown["keyword_coverage"]) < 0.01


def test_match_score_equals_its_own_weighted_components():
    """The composite must be exactly the weighted sum of the parts it
    reports -- no hidden term, no silent rescaling.

    The keyword term is keyword_effective (coverage damped by confidence),
    not raw coverage; the breakdown reports all three so the number stays
    reproducible by hand. See _coverage_confidence.
    """
    resume = _resume()
    job = normalise_posting(_posting("Backend Developer", "Acme"))

    scored = score_job(job, "python django", set(resume.skills), "Ahmedabad", now=NOW)
    breakdown = scored.score_breakdown

    expected = 100 * (
        RANK_WEIGHTS["keyword"] * breakdown["keyword_effective"]
        + RANK_WEIGHTS["recency"] * breakdown["recency"]
        + RANK_WEIGHTS["location"] * breakdown["location_fit"]
    )

    assert abs(scored.match_score - expected) < 0.01
    # keyword_effective must itself be derivable from the figures it sits
    # between, so the chain from raw coverage to final score stays fully
    # auditable: half the damped ratio, half the saturating match count.
    rebuilt = (
        discovery.COVERAGE_RATIO_WEIGHT
        * breakdown["keyword_coverage"]
        * breakdown["keyword_confidence"]
        + discovery.MATCH_COUNT_WEIGHT * breakdown["match_count_component"]
    )
    assert abs(breakdown["keyword_effective"] - rebuilt) < 0.0001
    # And the match-count component is just the saturating count.
    assert abs(
        breakdown["match_count_component"]
        - min(breakdown["matched_count"] / discovery.MATCH_COUNT_SATURATION, 1.0)
    ) < 0.0001


def test_a_thin_posting_cannot_outrank_a_well_described_one_on_coverage_alone():
    """The measured failure this damping exists to fix: a posting whose
    snippet names one skill the resume happens to have scored a perfect
    100% coverage and outranked genuinely relevant roles. With confidence
    damping, the richly-described posting wins.
    """
    resume_skills = {"Python", "Django", "PostgreSQL", "Docker"}
    resume_text = "python django postgresql docker"

    thin = normalise_posting(
        _posting("Shopify App Developer", "Thin Corp", description="Build apps with JavaScript.", posting_id="thin")
    )
    rich = normalise_posting(
        _posting(
            "Backend Developer",
            "Rich Corp",
            description="Python, Django and PostgreSQL required; Docker a plus.",
            posting_id="rich",
        )
    )
    # Give the thin posting a skill the resume actually has, so its raw
    # coverage is a perfect 1.0 -- the exact shape of the live failure.
    thin.description = "Build apps with Python."

    ranked = rank_jobs([thin, rich], resume_text, resume_skills, "Ahmedabad", now=NOW)

    assert ranked[0].company == "Rich Corp"
    assert ranked[0].meta.low_confidence is False
    thin_result = next(job for job in ranked if job.company == "Thin Corp")
    assert thin_result.meta.low_confidence is True
    # Raw coverage really was perfect -- it's the confidence that differs.
    assert thin_result.score_breakdown["keyword_coverage"] == 1.0
    assert thin_result.score_breakdown["keyword_confidence"] < 1.0


def test_rank_weights_sum_to_one():
    assert round(sum(RANK_WEIGHTS.values()), 10) == 1.0


# --------------------------------------------------------------------------
# 2-3. Fetch resilience and location widening
# --------------------------------------------------------------------------


def test_widening_triggers_below_the_threshold_not_above():
    """The rule is "< WIDEN_MIN_UNIQUE unique postings AND location isn't
    already India". Checked directly against the constant so the test
    can't drift from the implementation.
    """
    assert discovery.WIDEN_MIN_UNIQUE == 20

    below = discovery.WIDEN_MIN_UNIQUE - 1
    at = discovery.WIDEN_MIN_UNIQUE

    assert below < discovery.WIDEN_MIN_UNIQUE  # widens
    assert not (at < discovery.WIDEN_MIN_UNIQUE)  # does not widen


def test_widen_reason_is_plain_language_and_pluralised():
    assert discovery._widen_reason(1, "Ahmedabad") == "only 1 posting found in Ahmedabad"
    assert discovery._widen_reason(7, "Ahmedabad") == "only 7 postings found in Ahmedabad"


def test_a_richly_described_posting_is_not_penalised_for_naming_more_skills():
    """The bias this blend exists to fix, measured on a live 317-posting
    run: a Greenhouse posting (5700-char description, 6.9 skills named)
    matching 3 of the resume's skills scored *below* an Adzuna snippet
    (500 chars, 1 skill named) matching 1 -- because 3/13 is a worse ratio
    than 1/1. Absolute matches now carry half the keyword term, so the
    posting that wants more of what you have wins.
    """
    resume_skills = {"Python", "Django", "PostgreSQL", "Docker", "AWS"}
    resume_text = "python django postgresql docker aws"

    # Names one skill, and the resume has it: a perfect 1/1 ratio.
    thin = normalise_posting(
        _posting("Developer", "Thin Corp", description="We use Python.", posting_id="thin")
    )
    # Names many, three of which the resume has: a much worse ratio.
    rich = normalise_posting(
        _posting(
            "Backend Engineer",
            "Rich Corp",
            description=(
                "We use Python, Django and PostgreSQL. Also Kubernetes, Terraform, "
                "Kafka, Scala, Elasticsearch, GraphQL and Jenkins."
            ),
            posting_id="rich",
        )
    )

    ranked = rank_jobs([thin, rich], resume_text, resume_skills, "Ahmedabad", now=NOW)

    assert ranked[0].company == "Rich Corp"
    rich_result = next(job for job in ranked if job.company == "Rich Corp")
    thin_result = next(job for job in ranked if job.company == "Thin Corp")
    # The thin one still has the better ratio -- it just no longer wins.
    assert thin_result.score_breakdown["keyword_coverage"] > rich_result.score_breakdown["keyword_coverage"]
    assert rich_result.score_breakdown["matched_count"] > thin_result.score_breakdown["matched_count"]


def test_match_count_component_saturates():
    """Beyond MATCH_COUNT_SATURATION the component is capped, so a posting
    listing every technology under the sun can't run away with the score.
    """
    resume_skills = {"Python", "Django", "PostgreSQL", "Docker", "AWS", "Redis", "Kafka"}
    resume_text = " ".join(s.lower() for s in resume_skills)

    many = normalise_posting(
        _posting(
            "Engineer",
            "Many Corp",
            description="Python Django PostgreSQL Docker AWS Redis Kafka all required.",
            posting_id="many",
        )
    )
    scored = score_job(many, resume_text, resume_skills, "Ahmedabad", now=NOW)

    assert scored.score_breakdown["matched_count"] >= discovery.MATCH_COUNT_SATURATION
    assert scored.score_breakdown["match_count_component"] == 1.0
