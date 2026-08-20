"""Semantic fit: how well a resume's actual wording matches what a role's
requirement sentences are asking for.

PER-REQUIREMENT, not whole-document. Whole-document cosine similarity is
deliberately not used: two long technical documents tend to land around
~0.7 similarity regardless of real fit, which destroys the signal. Instead,
for each requirement sentence we find its single best-matching resume line
and average those best-match scores -- a resume only needs to say the
right thing somewhere, not read similarly to the whole posting.
"""

import functools
import os
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.services.roleprofile.miner import extract_skills_from_text

MIN_UNIT_CHARS = 25

# Measured 2026-08-20 with scripts/check_semantic.py against the 5
# fixtures in tests/fixtures/ (demo_after, demo_before, sample_kian,
# sample_salvador, sample_tasiana), role profile "backend developer" /
# "India", *after* change 1 below (skill-filtered harvested sentences +
# always-appended synthesised ones) was in place. Observed raw cosine
# means ranged 0.347-0.460; LOW/HIGH below round that outward by ~0.05 for
# headroom. The original 0.30-0.85 band was an unmeasured guess and
# compressed all 5 resumes into an 8.5-point score spread (0.368-0.414
# raw); this band gives the same 5 resumes a ~57-point spread instead. If
# the spread collapses again, re-measure with check_semantic.py rather
# than re-guessing.
RAW_COSINE_FLOOR = 0.30
RAW_COSINE_CEILING = 0.50

TOP_SKILLS_FOR_SYNTHESIS = 15
WEAKEST_REQUIREMENTS_COUNT = 5

_LEADING_BULLET_RE = re.compile(r"^[-•▪◦‣●·○■□▶➤➔❖✦∙\s]+")

# Confirmed by inspecting live Adzuna output: every description it
# truncates ends with a literal "…" (U+2026), appended regardless of
# whether the cut lands mid-word ("...backend inte…") or at a clean word
# boundary -- so checking for this one marker covers both "ends mid-word"
# and "ends with an ellipsis" for this data source. A literal "..." is
# treated the same way in case a future source signals truncation
# differently.
_TRUNCATION_MARKERS = ("…", "...")


def _model_appears_cached(model_name: str) -> bool:
    """Best-effort check of the default Hugging Face cache location, so we
    can print a heads-up before a first-run download that would otherwise
    look like a hang. Not foolproof -- a custom HF_HOME/cache_folder or an
    older sentence-transformers cache layout won't be detected -- but the
    worst case is an unnecessary notice for a model that was actually
    already cached, not a missed one.
    """
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    folder_name = "models--" + model_name.replace("/", "--")
    return (cache_home / "hub" / folder_name).exists()


@functools.lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    model_name = settings.embedding_model
    if not _model_appears_cached(model_name):
        print(f"Downloading embedding model '{model_name}' (~90MB, first run only)...")
    return SentenceTransformer(model_name)


def _split_resume_into_units(resume_text: str) -> list[str]:
    """Resume broken into short, independently-embeddable chunks: each
    non-blank line over MIN_UNIT_CHARS chars, with any leading bullet
    glyph stripped. Line-level rather than sentence-level, since resume
    bullets are already effectively one line each.
    """
    units = []
    for line in resume_text.split("\n"):
        stripped = _LEADING_BULLET_RE.sub("", line).strip()
        if len(stripped) > MIN_UNIT_CHARS:
            units.append(stripped)
    return units


def _is_usable_requirement(fragment: str) -> bool:
    """False if this fragment looks cut off by description truncation
    rather than being a complete clause -- a half-sentence produces a
    meaningless embedding. See _TRUNCATION_MARKERS.
    """
    return not fragment.rstrip().endswith(_TRUNCATION_MARKERS)


def _mentions_a_vocab_skill(fragment: str) -> bool:
    """True if this fragment names at least one real skill from
    skills_vocab.json (via miner.py's existing, alias-aware, boundary-
    correct scan).

    Most harvested Adzuna fragments are generic preamble ("We are looking
    for a technically strong...") rather than an actual requirement, and
    every resume matches generic preamble about equally -- that's what was
    compressing the raw cosine range to 0.368-0.414 across five very
    different resumes. Requiring a concrete skill mention is what turns a
    "requirement sentence" into one that can actually discriminate.
    """
    return bool(extract_skills_from_text(fragment))


# A few different phrasings so 15 synthesised sentences for 15 different
# skills don't end up as 15 near-identical vectors ("Experience with X"
# repeated) that all sit in the same tight cluster in embedding space.
# Cycled through by skill position, not chosen per-skill -- there's no
# semantic reason one template suits "python" better than another.
_REQUIREMENT_TEMPLATES = [
    "Hands-on experience building applications with {skill}",
    "Working knowledge of {skill} in a production environment",
    "Proficiency in {skill} for day-to-day development work",
    "Demonstrated ability to use {skill} to solve real engineering problems",
    "Comfortable working with {skill} as part of a development team",
]


def _synthesize_requirement_sentences(top_skills: list[str]) -> list[str]:
    """One requirement sentence per skill, phrased as a real job
    requirement rather than a bare list ("Experience with X"), cycling
    through _REQUIREMENT_TEMPLATES for phrasing variety. Casing is a
    simple capitalise-first-letter -- imperfect for acronyms ("Ci/cd") but
    harmless, since this only ever feeds an embedding model.
    """
    sentences = []
    for index, skill in enumerate(top_skills):
        template = _REQUIREMENT_TEMPLATES[index % len(_REQUIREMENT_TEMPLATES)]
        titled_skill = f"{skill[:1].upper()}{skill[1:]}"
        sentences.append(template.format(skill=titled_skill))
    return sentences


def semantic_score(
    resume_text: str,
    requirement_sentences: list[str],
    top_skills: list[str] | None = None,
) -> dict:
    """Per-requirement semantic fit between a resume and a role's
    requirement sentences.

    top_skills (role_profile["skill_frequencies"] keys, already frequency-
    ordered) drives synthesised requirements -- see
    _synthesize_requirement_sentences. Synthesised sentences are preferred
    over harvested ones: a harvested sentence is only kept if it survives
    truncation filtering AND names a real skill (most harvested Adzuna
    fragments are generic preamble that doesn't discriminate between
    resumes at all -- see _mentions_a_vocab_skill). Sentences for the top
    TOP_SKILLS_FOR_SYNTHESIS profile skills are then always appended on
    top of whatever real sentences survive.

    Returns {"score", "raw_cosine_mean", "weakest_requirements",
    "requirements_synthesised"}. weakest_requirements holds the
    WEAKEST_REQUIREMENTS_COUNT requirements with the lowest best-match
    score, weakest first. requirements_synthesised is true whenever
    synthesised sentences contributed to the set (i.e. top_skills was
    non-empty) -- in practice, almost always.
    """
    usable_requirements = [
        r
        for r in requirement_sentences
        if _is_usable_requirement(r) and _mentions_a_vocab_skill(r)
    ]

    synthesised = _synthesize_requirement_sentences((top_skills or [])[:TOP_SKILLS_FOR_SYNTHESIS])
    requirements_synthesised = bool(synthesised)
    usable_requirements = usable_requirements + synthesised

    if not usable_requirements:
        return {
            "score": 0.0,
            "raw_cosine_mean": 0.0,
            "weakest_requirements": [],
            "requirements_synthesised": requirements_synthesised,
        }

    units = _split_resume_into_units(resume_text)
    if not units:
        # Nothing to compare against -- every requirement is unmet.
        return {
            "score": 0.0,
            "raw_cosine_mean": 0.0,
            "weakest_requirements": [
                {"requirement": requirement, "evidence_strength": 0.0}
                for requirement in usable_requirements[:WEAKEST_REQUIREMENTS_COUNT]
            ],
            "requirements_synthesised": requirements_synthesised,
        }

    model = _get_model()
    requirement_vectors = model.encode(usable_requirements, normalize_embeddings=True)
    unit_vectors = model.encode(units, normalize_embeddings=True)

    similarity = requirement_vectors @ unit_vectors.T
    best_per_requirement = similarity.max(axis=1)

    raw_cosine_mean = float(best_per_requirement.mean())
    calibrated = (raw_cosine_mean - RAW_COSINE_FLOOR) / (RAW_COSINE_CEILING - RAW_COSINE_FLOOR)
    score = float(np.clip(calibrated, 0.0, 1.0))

    weakest_order = np.argsort(best_per_requirement)[:WEAKEST_REQUIREMENTS_COUNT]
    weakest_requirements = [
        {
            "requirement": usable_requirements[i],
            "evidence_strength": float(best_per_requirement[i]),
        }
        for i in weakest_order
    ]

    return {
        "score": round(score, 4),
        "raw_cosine_mean": round(raw_cosine_mean, 4),
        "weakest_requirements": weakest_requirements,
        "requirements_synthesised": requirements_synthesised,
    }
