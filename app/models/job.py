"""Pydantic schemas for job discovery.

Shapes only -- every algorithm (query generation, dedupe, ranking) lives
in app/services/discovery.py, per CLAUDE.md's services rule.
"""

from pydantic import BaseModel, Field


class JobMeta(BaseModel):
    """Per-job caveats the UI should surface rather than hide.

    description_truncated is the important one: Adzuna's free tier returns
    a ~250-char snippet, so a posting's keyword coverage is computed over
    whatever skills happen to fall inside that snippet. The score is real
    but preliminary -- it says "of what this posting told us, you cover X",
    not "of what this job needs, you cover X".
    """

    description_truncated: bool = True
    description_chars: int = 0
    skills_detected_in_posting: int = 0
    # True when the posting named too few skills for its coverage figure
    # to mean much, but did name some; the UI labels these rather than
    # showing the percentage as though it were as solid as any other.
    low_confidence: bool = False
    # True when the posting's snippet named NO skill we recognise at all.
    # Distinct from low_confidence: there's no coverage figure to damp
    # here, the skills term simply couldn't be computed, so the job is
    # ranked on recency and location alone. Worth saying plainly rather
    # than showing a 0% that reads as "you match nothing".
    skills_unscored: bool = False


class Job(BaseModel):
    fingerprint: str
    source: str = "adzuna"
    # Attribution text required by each provider's terms, travelling with
    # the posting so the UI never has to infer it. See jobs/base.py.
    source_label: str = "via Adzuna"
    title: str | None = None
    company: str | None = None
    location: str | None = None
    is_remote: bool = False
    description: str | None = None
    url: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    posted_at: str | None = None
    # Only some providers state this (JSearch does). None where unknown.
    employment_type: str | None = None

    match_score: float = 0.0
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)

    # Component breakdown behind match_score, so the UI (and the honesty
    # test) can show how the number was reached instead of asserting it.
    score_breakdown: dict = Field(default_factory=dict)
    meta: JobMeta = Field(default_factory=JobMeta)


class DiscoveryRunRequest(BaseModel):
    analysis_id: str
