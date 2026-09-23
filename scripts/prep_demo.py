"""Pre-mine and cache everything the demo needs, so it can run with
DEMO_MODE on and zero live network dependency.

Run this once with DEMO_MODE off (it needs real network access to mine
the role profile and to call the LLM parser), then flip DEMO_MODE on for
the actual demo.

Usage:
    python scripts/prep_demo.py
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services import auth as auth_service
from app.services.analyze import run_analysis
from app.services.extraction.parse_cache import get_or_parse
from app.services.extraction.text_extract import extract_text
from app.services import discovery as discovery_service
from app.services.roleprofile.cache import get_or_mine
from app.store import load_json, save_json

DEMO_ROLE = "backend developer"
DEMO_LOCATION = "India"

# Discovery runs warmed for the demo, so both paths work without a network.
#
# Rajkot, not Ahmedabad, is the thin-city case. Ahmedabad was the obvious
# choice and it does not work: a live probe returned 65 unique postings for
# "backend developer" there, well above WIDEN_MIN_UNIQUE (20), so the
# widening banner never fires and that path can't be demoed. Rajkot,
# Bhavnagar and Jamnagar all return 0 postings for this role, which does
# trigger widening and recovers ~94 jobs from India.
#
# Ahmedabad is still warmed: it's the location the demo resume is analysed
# against, so the "Find matching jobs" button needs a cached run for it.
DEMO_DISCOVERY_CITIES = ["Ahmedabad", "Rajkot", "India"]

# Not a real secret -- a fixed local account so the presenter can log in
# during the demo and see these pre-warmed analyses under History.
DEMO_USER_EMAIL = "demo@careerstack.ai"
DEMO_USER_PASSWORD = "demo12345"

FIXTURES_DIR = Path("tests/fixtures")

# CLAUDE.md's demo script names these tests/fixtures/resume_bad.pdf and
# resume_fixed.pdf; the fixtures were renamed at some point after that was
# written, and those filenames don't exist in this repo. Using the actual
# current fixtures -- demo_before.pdf / demo_after.pdf are the same
# deliberately-bad / fixed pair CLAUDE.md describes.
BAD_RESUME_PATH = FIXTURES_DIR / "demo_before.pdf"
FIXED_RESUME_PATH = FIXTURES_DIR / "demo_after.pdf"


def _ensure_demo_user() -> str:
    """Creates the demo account if it doesn't exist yet; returns its
    user_id either way.
    """
    existing = auth_service.authenticate(DEMO_USER_EMAIL, DEMO_USER_PASSWORD)
    if existing is not None:
        return existing.user_id
    user = auth_service.create_user(DEMO_USER_EMAIL, DEMO_USER_PASSWORD, "Demo User")
    print(f"Created demo account: {DEMO_USER_EMAIL} / {DEMO_USER_PASSWORD}\n")
    return user.user_id


def _warm_parse_cache(path: Path) -> None:
    """Parse and cache one fixture, so it can be uploaded through
    /api/analyze with DEMO_MODE on later -- get_or_parse never calls Groq
    in DEMO_MODE, only serves what's already cached.
    """
    file_bytes = path.read_bytes()
    extracted = extract_text(file_bytes, path.name)
    if extracted["needs_ocr"]:
        print(f"  ! {path.name}: no selectable text, skipping")
        return
    get_or_parse(file_bytes, extracted["text"])
    print(f"  - {path.name}: parse cached")


async def _run() -> None:
    if settings.demo_mode:
        print(
            "WARNING: DEMO_MODE is currently true. Mining and parsing need "
            "real network access to populate their caches -- this will only "
            "work for whatever's already cached. Set DEMO_MODE=false in .env "
            "and re-run this script first, then flip it back on for the "
            "actual demo.\n"
        )

    demo_user_id = _ensure_demo_user()

    fixture_paths = sorted(FIXTURES_DIR.glob("*.pdf"))
    print(f"Warming the resume-parse cache for all {len(fixture_paths)} fixtures ...")
    for path in fixture_paths:
        _warm_parse_cache(path)
    print()

    print(f"Mining role profile: {DEMO_ROLE!r} / {DEMO_LOCATION!r} ...")
    profile = await get_or_mine(DEMO_ROLE, DEMO_LOCATION)
    print(
        f"  -> {profile['postings_sampled']} postings sampled, "
        f"{len(profile['skill_frequencies'])} skills, "
        f"sparse_profile={profile['sparse_profile']}\n"
    )

    print(f"Analysing {BAD_RESUME_PATH.name} ...")
    bad_result = await run_analysis(
        BAD_RESUME_PATH.read_bytes(), BAD_RESUME_PATH.name, DEMO_ROLE, DEMO_LOCATION, demo_user_id
    )

    print(f"Analysing {FIXED_RESUME_PATH.name} ...")
    fixed_result = await run_analysis(
        FIXED_RESUME_PATH.read_bytes(), FIXED_RESUME_PATH.name, DEMO_ROLE, DEMO_LOCATION, demo_user_id
    )

    print()
    discovery_runs = await _warm_discovery_runs(demo_user_id, fixed_result["analysis_id"])

    print("\n" + "=" * 50)
    print(f"{'fixture':24} {'score':>8}  band")
    print("-" * 50)
    for path, result in [(BAD_RESUME_PATH, bad_result), (FIXED_RESUME_PATH, fixed_result)]:
        print(f"{path.name:24} {result['overall_score']:8.1f}  {result['band']}")
    gap = fixed_result["overall_score"] - bad_result["overall_score"]
    print("-" * 50)
    print(f"gap: {gap:+.1f} points")
    print(f"analysis_id (bad):   {bad_result['analysis_id']}")
    print(f"analysis_id (fixed): {fixed_result['analysis_id']}")

    for city, run in discovery_runs:
        if run is None:
            print(f"discovery run ({city}): FAILED -- see the warning above")
        else:
            stats = run["stats"]
            widened = " (widened to India)" if run.get("widened") else ""
            print(
                f"discovery run ({city}): {stats['unique']} unique / "
                f"{stats['fetched']} fetched{widened}"
            )

    print(
        "\nCached under ./data/parsed_resumes/, ./data/role_profiles/, "
        "./data/analyses/ and ./data/discovery_runs/ -- survives a restart."
    )
    print("Set DEMO_MODE=true and restart the server before the actual demo.")
    print(f"Log in during the demo as: {DEMO_USER_EMAIL} / {DEMO_USER_PASSWORD}")


async def _warm_discovery_runs(demo_user_id: str, analysis_id: str) -> list[tuple[str, dict | None]]:
    """Warm one discovery run per DEMO_DISCOVERY_CITIES entry.

    Each run needs its own analysis record, since a run takes its role and
    location from the analysis it belongs to. Those analyses are cloned
    from the already-computed fixed-resume analysis rather than re-scored:
    only the location differs, and rescoring would spend another LLM call
    on a record whose scores this step never reads.

    A failure here is reported and skipped rather than raised -- the
    resume analyses are the demo's core, and losing one discovery run
    shouldn't take the whole prep down with it.
    """
    base = load_json("analyses", analysis_id)
    runs: list[tuple[str, dict | None]] = []

    for city in DEMO_DISCOVERY_CITIES:
        print(f"Warming discovery run: {DEMO_ROLE!r} / {city!r} ...")
        clone_id = uuid.uuid4().hex
        save_json("analyses", clone_id, {**base, "analysis_id": clone_id, "location": city})

        try:
            run = discovery_service.create_run(clone_id, demo_user_id)
            result = await discovery_service.execute_run(run["run_id"])
            if result.get("status") != "complete":
                raise RuntimeError(result.get("error") or "run did not complete")
        except Exception as exc:  # noqa: BLE001 -- see docstring
            print(f"  -> WARNING: could not warm this run ({exc})")
            runs.append((city, None))
            continue

        stats = result["stats"]
        widened = f", widened to {result['widened']['to']}" if result.get("widened") else ""
        print(
            f"  -> {stats['unique']} unique from {stats['fetched']} fetched, "
            f"{round(stats['duplicate_rate'] * 100)}% duplicates{widened}"
        )
        runs.append((city, result))

    return runs


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
