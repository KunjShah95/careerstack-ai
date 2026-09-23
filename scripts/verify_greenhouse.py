"""Check every Greenhouse board token still resolves, and report coverage.

Board tokens change when companies rename or migrate ATS, and a dead token
fails silently -- the adapter logs a warning and contributes nothing, so
the only symptom is quietly fewer results. Run this before a demo.

Usage:
    python scripts/verify_greenhouse.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.services.jobs.greenhouse import _COMPANIES_FILE, CONCURRENCY, matches_location

BASE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
TIMEOUT_SECONDS = 15.0


async def _check(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, entry: dict) -> dict:
    token = entry["token"]
    async with semaphore:
        try:
            response = await client.get(BASE_URL.format(token=token), timeout=TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            return {**entry, "status": type(exc).__name__, "jobs": 0, "india": 0}

    if response.status_code != 200:
        return {**entry, "status": response.status_code, "jobs": 0, "india": 0}

    jobs = response.json().get("jobs", [])
    india = sum(
        1 for job in jobs if matches_location((job.get("location") or {}).get("name"), "India")
    )
    return {**entry, "status": 200, "jobs": len(jobs), "india": india}


async def _run() -> None:
    payload = json.loads(_COMPANIES_FILE.read_text(encoding="utf-8"))
    entries = payload.get("companies", [])

    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[_check(client, semaphore, e) for e in entries])

    print(f"{'token':34} {'status':>8} {'jobs':>6} {'india':>6}  drift")
    print("-" * 70)
    for result in sorted(results, key=lambda r: -r["india"]):
        recorded = result.get("jobs_at_verification")
        drift = ""
        if result["status"] == 200 and isinstance(recorded, int) and recorded:
            change = (result["jobs"] - recorded) / recorded
            if abs(change) >= 0.5:
                drift = f"{change:+.0%} vs recorded {recorded}"
        print(
            f"{result['token']:34} {str(result['status']):>8} "
            f"{result['jobs']:6} {result['india']:6}  {drift}"
        )

    broken = [r for r in results if r["status"] != 200]
    print()
    print(f"working: {len(results) - len(broken)}/{len(results)}")
    print(f"india-located postings: {sum(r['india'] for r in results)}")

    if broken:
        print("\nBROKEN -- remove or correct these in greenhouse_companies.json:")
        for result in broken:
            print(f"  {result['token']} -> {result['status']}")
        sys.exit(1)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
