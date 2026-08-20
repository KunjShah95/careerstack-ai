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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.analyze import run_analysis
from app.services.extraction.parse_cache import get_or_parse
from app.services.extraction.text_extract import extract_text
from app.services.roleprofile.cache import get_or_mine

DEMO_ROLE = "backend developer"
DEMO_LOCATION = "India"

FIXTURES_DIR = Path("tests/fixtures")

# CLAUDE.md's demo script names these tests/fixtures/resume_bad.pdf and
# resume_fixed.pdf; the fixtures were renamed at some point after that was
# written, and those filenames don't exist in this repo. Using the actual
# current fixtures -- demo_before.pdf / demo_after.pdf are the same
# deliberately-bad / fixed pair CLAUDE.md describes.
BAD_RESUME_PATH = FIXTURES_DIR / "demo_before.pdf"
FIXED_RESUME_PATH = FIXTURES_DIR / "demo_after.pdf"


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
        BAD_RESUME_PATH.read_bytes(), BAD_RESUME_PATH.name, DEMO_ROLE, DEMO_LOCATION
    )

    print(f"Analysing {FIXED_RESUME_PATH.name} ...")
    fixed_result = await run_analysis(
        FIXED_RESUME_PATH.read_bytes(), FIXED_RESUME_PATH.name, DEMO_ROLE, DEMO_LOCATION
    )

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

    print(
        "\nCached under ./data/parsed_resumes/, ./data/role_profiles/, and "
        "./data/analyses/ -- survives a restart."
    )
    print("Set DEMO_MODE=true and restart the server before the actual demo.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
