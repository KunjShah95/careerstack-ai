"""Tests for per-requirement semantic fit in semantic.py.

The model-backed tests use the real MiniLM model (cached locally after the
first run) rather than a mock, since the whole point of these checks is
the per-requirement max-then-average behaviour and the calibration curve,
not just that some function got called. They assert on thresholds and
relative ordering rather than exact cosine floats, since embedding values
can shift by tiny amounts across hardware/BLAS backends.
"""

from app.services.scoring.semantic import (
    _is_usable_requirement,
    _mentions_a_vocab_skill,
    _synthesize_requirement_sentences,
    semantic_score,
)


def test_is_usable_requirement_filters_truncated_fragments():
    assert _is_usable_requirement("Experience with REST APIs and Python…") is False
    assert _is_usable_requirement("Familiarity with cloud infrastructure and dep...") is False
    assert _is_usable_requirement("Strong background in relational databases") is True


def test_mentions_a_vocab_skill_filters_generic_preamble():
    """This is the actual root-cause fix: most harvested Adzuna fragments
    are generic preamble that every resume matches about equally, and a
    requirement sentence with no real skill in it can't discriminate
    between resumes at all.
    """
    assert _mentions_a_vocab_skill("We are looking for a technically strong software engineer") is False
    assert _mentions_a_vocab_skill("Bachelor degree in computer science or related field required") is False
    assert _mentions_a_vocab_skill("3+ years of experience with Python and Django required") is True


def test_synthesized_sentences_vary_phrasing():
    """15 sentences for 15 different skills shouldn't collapse into
    near-identical vectors -- confirms the template cycling actually
    produces distinct phrasings, not just distinct skill names.
    """
    skills = ["python", "django", "postgresql", "docker", "kubernetes", "git"]
    sentences = _synthesize_requirement_sentences(skills)

    assert len(sentences) == len(skills)
    assert len(set(sentences)) == len(skills)  # all distinct

    opening_words = {sentence.split(" ")[0] for sentence in sentences}
    assert len(opening_words) > 1  # not all the same template


def test_harvested_only_path_discards_the_unfiltered_generic_sentence():
    """No top_skills given, so only the harvested-and-filtered path runs.
    Of two harvested fragments, only the one naming a real skill should
    survive to be scored; the generic one is discarded outright rather
    than merely scoring low.
    """
    resume_text = "Backend engineer with production experience in Python and Django."
    requirement_sentences = [
        "We are looking for a technically strong software engineer to join our team",
        "3+ years of experience with Python and Django required",
    ]

    result = semantic_score(resume_text, requirement_sentences)

    assert result["requirements_synthesised"] is False
    assert len(result["weakest_requirements"]) == 1
    assert result["weakest_requirements"][0]["requirement"] == (
        "3+ years of experience with Python and Django required"
    )


def test_prefers_synthesised_requirements_and_varies_their_phrasing():
    """All 3 harvested fragments survive truncation-filtering but name no
    real skill, so they're discarded; synthesised sentences for top_skills
    take over entirely, with varied phrasing rather than one template.
    """
    resume_text = """
    SKILLS
    Python, Django, PostgreSQL, Docker, AWS, Git

    EXPERIENCE
    - Built REST API endpoints in Django REST Framework.
    - Deployed services to AWS using Docker containers.
    """
    requirement_sentences = [
        "We are looking for a passionate, driven, technically strong candidate",
        "This is a great opportunity to grow your career with us",
        "Join our fast-paced, collaborative, mission-driven team",
    ]
    top_skills = ["python", "django", "aws", "docker", "kubernetes"]

    result = semantic_score(resume_text, requirement_sentences, top_skills=top_skills)

    assert result["requirements_synthesised"] is True
    assert len(result["weakest_requirements"]) == 5

    skill_mentions = {item["requirement"].lower() for item in result["weakest_requirements"]}
    assert any("kubernetes" in text for text in skill_mentions)

    opening_words = {item["requirement"].split(" ")[0] for item in result["weakest_requirements"]}
    assert len(opening_words) > 1


def test_no_requirements_and_no_top_skills_returns_a_safe_zero_result():
    """Nothing usable survives harvested filtering, and no top_skills
    fallback is given -- there is nothing to score against, so this must
    degrade gracefully rather than crash (e.g. on an empty embedding
    batch). requirements_synthesised is False here since no synthesis
    actually happened (top_skills was empty).
    """
    result = semantic_score("Some resume text here.", ["Cut off mid-sente…"], top_skills=None)

    assert result == {
        "score": 0.0,
        "raw_cosine_mean": 0.0,
        "weakest_requirements": [],
        "requirements_synthesised": False,
    }


def test_per_requirement_scoring_surfaces_the_genuinely_uncovered_requirement():
    """The resume clearly covers Python/Django/PostgreSQL/Docker, and just
    as clearly says nothing about Kubernetes or leadership/communication.
    Per-requirement best-match scoring should surface those as weak, while
    a well-covered skill (Docker) should not appear among the weakest 5.
    """
    resume_text = """
    SUMMARY
    Backend developer with experience building REST APIs in Python and Django.

    EXPERIENCE
    - Built and documented REST API endpoints in Django REST Framework for the internal analytics dashboard.
    - Optimised slow PostgreSQL queries by adding composite indexes to speed up dashboard load time.
    - Containerised the local development environment with Docker Compose for faster onboarding.
    - Wrote pytest unit tests and wired them into the GitHub Actions CI pipeline.
    - Collaborated with product and infrastructure engineers on a two-week sprint cadence.
    """
    requirement_sentences = [
        "Strong experience building REST APIs with Python and Django",
        "Experience optimising SQL database queries for performance",
        "Familiarity with Docker containerisation for local development",
        "Experience with Kubernetes cluster orchestration at scale",
        "Strong communication skills and cross-functional collaboration",
        "Bachelor degree in computer science or related field required",
    ]
    top_skills = ["python", "django", "postgresql", "docker", "kubernetes", "leadership"]

    result = semantic_score(resume_text, requirement_sentences, top_skills=top_skills)

    assert result["requirements_synthesised"] is True
    assert 0.0 <= result["score"] <= 1.0
    assert len(result["weakest_requirements"]) == 5

    weakest_texts = {item["requirement"].lower() for item in result["weakest_requirements"]}
    assert any("kubernetes" in text for text in weakest_texts)
    assert not any("docker" in text for text in weakest_texts)

    # weakest first
    strengths = [item["evidence_strength"] for item in result["weakest_requirements"]]
    assert strengths == sorted(strengths)
