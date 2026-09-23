"""One-off diagnostic: mine two role profiles and print their frequency
tables, then score sample_kian.pdf against "full stack developer" / India
so its matched/missing skill lists can be sanity-checked before demo.
Not part of the app -- scratch script, mirrors scripts/check_ats.py's
pattern.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.extraction.layout import layout_signals
from app.services.extraction.text_extract import extract_text
from app.services.extraction.llm_parse import parse_resume
from app.services.roleprofile.cache import get_or_mine
from app.services.scoring.ats import compute_ats_score

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def print_profile(label: str, profile: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"role={profile['role']!r} location={profile['location']!r}")
    print(f"postings_sampled={profile['postings_sampled']} sparse_profile={profile.get('sparse_profile')}")
    print(f"sampled_at={profile['sampled_at']}")
    freqs = sorted(profile["skill_frequencies"].items(), key=lambda kv: kv[1]["frequency"], reverse=True)
    print(f"{len(freqs)} skills:")
    for skill, data in freqs:
        print(f"  {skill:<25} freq={data['frequency']:.3f}  count={data['count']}")


async def score_fixture(filename: str, role: str, location: str) -> None:
    file_bytes = (FIXTURES_DIR / filename).read_bytes()
    extracted = extract_text(file_bytes, filename)
    layout = layout_signals(file_bytes)
    resume = parse_resume(extracted["text"])
    profile = await get_or_mine(role, location)

    result = compute_ats_score(resume, extracted["text"], layout, profile)
    kd = result["keyword_detail"]
    matched = sorted(kd["matched"], key=lambda m: m["frequency"], reverse=True)
    missing = sorted(kd["missing"], key=lambda m: m["frequency"], reverse=True)

    print(f"\n=== {filename} vs role={role!r} location={location!r} ===")
    print(f"overall_score={result['overall_score']}  band={result['band']}")
    print(f"subscores={result['subscores']}")
    print(f"matched ({len(matched)}): {[m['skill'] for m in matched]}")
    print(f"missing ({len(missing)}): {[m['skill'] for m in missing]}")


async def main() -> None:
    fsd_profile = await get_or_mine("full stack developer", "India")
    print_profile('"full stack developer" / India', fsd_profile)

    mern_profile = await get_or_mine("MERN developer", "India")
    print_profile('"MERN developer" / India', mern_profile)

    await score_fixture("sample_kian.pdf", "full stack developer", "India")


if __name__ == "__main__":
    asyncio.run(main())
