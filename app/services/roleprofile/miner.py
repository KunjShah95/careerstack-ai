"""Role profile mining: turn a batch of live job postings into a skill
frequency table, requirement sentences, and a rough experience bar for one
role.

No LLM calls here. We're scanning ~40 postings against a fixed vocabulary;
an LLM call per posting would be slow and would burn quota for no benefit
over a word-boundary regex scan.
"""

import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from app.services.roleprofile.adzuna import search

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

_SKILLS_VOCAB: list[str] = json.loads((_DATA_DIR / "skills_vocab.json").read_text(encoding="utf-8"))
_SKILL_ALIASES: dict[str, str] = json.loads(
    (_DATA_DIR / "skill_aliases.json").read_text(encoding="utf-8")
)

MIN_REQUIREMENT_SENTENCE_CHARS = 25

# Adzuna's free tier truncates `description` to a ~250-char snippet, which
# cuts most requirement text off before it names more than a couple of
# skills. A fixed frequency floor (e.g. "must appear in >=15% of postings")
# assumes descriptions are rich enough for that percentage to mean
# anything; against a snippet-length sample it just starves the profile --
# see the mine_profile.py run that only cleared 5 skills out of 40
# postings. Counting raw mentions instead is robust to how much text we
# actually got per posting: "seen in >=3 of 40 postings" is a real signal
# regardless of whether each posting handed us 2000 characters or 250.
MIN_SKILL_MENTIONS = 3
MIN_SKILL_MENTIONS_SPARSE = 2
MAX_SKILLS_RETURNED = 25
SPARSE_PROFILE_SKILL_FLOOR = 8

_YEAR_RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)\s*years?", re.IGNORECASE)
_YEAR_PLUS_RE = re.compile(r"(\d+)\s*\+\s*years?", re.IGNORECASE)
_FRESHER_RE = re.compile(r"\b(fresher|entry[- ]level|graduate)\b", re.IGNORECASE)


def _boundary_pattern(term: str) -> re.Pattern:
    """Word-boundary regex for one skill spelling.

    Plain \\b doesn't work for skills with punctuation (".net", "c++",
    "ci/cd") since \\b sits between a word char and a non-word char, and
    those terms end or start on a non-word char themselves. This instead
    requires the character just outside the match (if any) to not be
    alphanumeric, which is boundary-correct for punctuated terms too.

    The left boundary also excludes "." specifically: without that, a
    short alias like "js" matches inside "react.js" / "node.js" / "vue.js"
    (the "." satisfies a plain not-alphanumeric check), which would count
    every JS-framework mention as a separate "javascript" hit too. "."
    stays allowed on the right, since that's just an ordinary
    end-of-sentence period ("...experience with React.") and excluding it
    there would break that far more common case.
    """
    escaped = re.escape(term)
    return re.compile(r"(?<![A-Za-z0-9.])" + escaped + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _build_skill_patterns() -> dict[str, list[re.Pattern]]:
    """One canonical skill -> patterns for every spelling that should match
    it (its own name plus every alias that normalises to it). This is the
    "normalise through skill_aliases" step: instead of rewriting the input
    text, every alias spelling is folded into the pattern set for its
    canonical skill up front, at import time.
    """
    variants_by_canonical: dict[str, set[str]] = {}
    for canonical in _SKILLS_VOCAB:
        variants_by_canonical.setdefault(canonical, set()).add(canonical)
    for variant, canonical in _SKILL_ALIASES.items():
        variants_by_canonical.setdefault(canonical, set()).add(variant)

    return {
        canonical: [_boundary_pattern(variant) for variant in variants]
        for canonical, variants in variants_by_canonical.items()
    }


_SKILL_PATTERNS = _build_skill_patterns()


def extract_skills_from_text(text: str) -> set[str]:
    """Canonical skills from skills_vocab that appear anywhere in text.

    Matches both a skill's canonical spelling and any of its alias
    spellings ("js", "reactjs", "k8s", ...); either counts as the skill
    itself being present.
    """
    return {
        canonical
        for canonical, patterns in _SKILL_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    }


_ROLE_SYNONYM_SWAPS = [("developer", "engineer"), ("engineer", "developer")]
_SENIORITY_PREFIX_RE = re.compile(
    r"^\s*(senior|junior|lead|principal|entry[- ]level)\s+", re.IGNORECASE
)


def _role_query_variants(role: str) -> list[str]:
    """Up to 3 phrasings of the same role, so a single exact title string
    isn't the only thing we search for. Not exhaustive -- just enough
    variety (a developer/engineer synonym swap, a seniority-stripped
    version) to catch postings titled differently for the same job.
    """
    variants = [role]

    lowered = role.lower()
    for source, target in _ROLE_SYNONYM_SWAPS:
        if source in lowered and target not in lowered:
            variants.append(re.sub(re.escape(source), target, role, flags=re.IGNORECASE))
            break
    if len(variants) < 2:
        variants.append(f"{role} developer")

    seniority_stripped = _SENIORITY_PREFIX_RE.sub("", role).strip()
    if seniority_stripped and seniority_stripped.lower() != role.lower():
        variants.append(seniority_stripped)
    else:
        variants.append(f"{role} jobs")

    seen: set[str] = set()
    unique_variants = []
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            unique_variants.append(variant)

    return unique_variants[:3]


def _extract_required_years(text: str) -> float | None:
    """One required-years estimate from a posting's text, or None if it
    doesn't mention experience in a pattern we recognise.

    "N-M years" -> midpoint. "N+ years" -> N. fresher/entry-level/graduate
    -> 0. Checked in that order so "3-5 years" isn't also picked up by a
    looser pattern.
    """
    range_match = _YEAR_RANGE_RE.search(text)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        return (low + high) / 2

    plus_match = _YEAR_PLUS_RE.search(text)
    if plus_match:
        return float(plus_match.group(1))

    if _FRESHER_RE.search(text):
        return 0.0

    return None


def _select_skills(
    skill_counts: dict[str, int], total_postings: int, min_mentions: int
) -> dict[str, dict]:
    """Skills mentioned in at least min_mentions postings, each carrying its
    raw count alongside its frequency, capped at MAX_SKILLS_RETURNED and
    ordered by frequency (count as tiebreak, name for determinism below
    that).
    """
    selected = {
        skill: {
            "frequency": round(count / total_postings, 4) if total_postings else 0.0,
            "count": count,
        }
        for skill, count in skill_counts.items()
        if count >= min_mentions
    }
    ordered = sorted(
        selected.items(),
        key=lambda item: (-item[1]["frequency"], -item[1]["count"], item[0]),
    )[:MAX_SKILLS_RETURNED]
    return dict(ordered)


async def mine_role_profile(role: str, location: str, target_postings: int = 40) -> dict:
    """Mine a role profile from live job postings.

    Runs 3 query variants of the role for title-phrasing variety, dedupes
    the combined results by Adzuna posting id (capped at target_postings),
    then derives a skill frequency table, a pool of requirement sentences,
    and a rough required-experience bar from the raw posting text.
    """
    variants = _role_query_variants(role)

    postings_by_id: dict[str, dict] = {}
    for variant in variants:
        for posting in await search(variant, location, limit=target_postings, country="in"):
            posting_id = posting.get("id")
            if posting_id and posting_id not in postings_by_id:
                postings_by_id[posting_id] = posting

    postings = list(postings_by_id.values())[:target_postings]
    total_postings = len(postings)

    skill_counts: dict[str, int] = {}
    requirement_sentences: list[str] = []
    required_years: list[float] = []

    for posting in postings:
        title = posting.get("title") or ""
        category = posting.get("category") or ""
        description = posting.get("description") or ""
        # title and category are never truncated by Adzuna; description is.
        # Scanning all three is what makes skill detection work on a
        # snippet-length description -- see the MIN_SKILL_MENTIONS comment
        # above for why the frequency floor also had to change to match.
        combined_text = f"{title}\n{category}\n{description}"

        for skill in extract_skills_from_text(combined_text):
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

        for fragment in re.split(r"[\n.]+", description):
            fragment = fragment.strip()
            if len(fragment) > MIN_REQUIREMENT_SENTENCE_CHARS:
                requirement_sentences.append(fragment)

        years = _extract_required_years(combined_text)
        if years is not None:
            required_years.append(years)

    sparse_profile = False
    skill_frequencies = _select_skills(skill_counts, total_postings, MIN_SKILL_MENTIONS)
    if len(skill_frequencies) < SPARSE_PROFILE_SKILL_FLOOR:
        sparse_profile = True
        skill_frequencies = _select_skills(skill_counts, total_postings, MIN_SKILL_MENTIONS_SPARSE)

    median_experience_years = (
        round(statistics.median(required_years), 1) if required_years else 0.0
    )

    return {
        "role": role,
        "location": location,
        "postings_sampled": total_postings,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "skill_frequencies": skill_frequencies,
        "sparse_profile": sparse_profile,
        "requirement_sentences": requirement_sentences,
        "median_experience_years": median_experience_years,
        "source_ids": [posting.get("id") for posting in postings],
    }
