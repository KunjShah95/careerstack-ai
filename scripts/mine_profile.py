"""CLI to mine a role profile and eyeball the results. Not a test.

Usage:
    python scripts/mine_profile.py "backend developer" "Bangalore"
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.roleprofile.miner import mine_role_profile


async def _run(role: str, location: str) -> None:
    profile = await mine_role_profile(role, location)

    print(f"role: {profile['role']}")
    print(f"location: {profile['location']}")
    print(f"postings_sampled: {profile['postings_sampled']}")
    print(f"median_experience_years: {profile['median_experience_years']}")
    print(f"sparse_profile: {profile['sparse_profile']}")
    print(f"sampled_at: {profile['sampled_at']}")

    postings_sampled = profile["postings_sampled"]
    top_skills = list(profile["skill_frequencies"].items())[:25]
    print(f"\ntop {len(top_skills)} skills:")
    print(f"{'skill':<35}{'frequency':>10}  mentions")
    print("-" * 60)
    for skill, stats in top_skills:
        mentions = f"{stats['count']} of {postings_sampled} postings"
        print(f"{skill:<35}{stats['frequency']:>10.2%}  {mentions}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", help="Target job role, e.g. 'backend developer'")
    parser.add_argument("location", help="Location to search, e.g. 'Bangalore'")
    args = parser.parse_args()

    asyncio.run(_run(args.role, args.location))


if __name__ == "__main__":
    main()
