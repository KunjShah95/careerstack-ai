"""Show API call consumption per source per month.

JSearch's free tier is 200 calls/month and RapidAPI exposes no usable
remaining-quota header on that plan, so app/services/jobs/quota.py counts
locally. This prints what it has recorded.

Usage:
    python scripts/check_quota.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.jobs import jsearch, quota
from app.store import list_keys, load_json


def main() -> None:
    records = [load_json(quota.COLLECTION, key) for key in list_keys(quota.COLLECTION)]
    records = [record for record in records if record]

    if not records:
        print("No API calls recorded yet.")
        return

    print(f"{'source':16} {'period':10} {'calls':>7}  {'last call':26} budget")
    print("-" * 74)

    for record in sorted(records, key=lambda r: (r.get("period", ""), r.get("source", "")), reverse=True):
        source = record.get("source", "?")
        calls = record.get("calls", 0)
        last = (record.get("last_call_at") or "")[:25]

        budget = ""
        if source == jsearch.SOURCE_NAME:
            remaining = jsearch.MONTHLY_CALL_BUDGET - calls
            budget = f"{remaining} of {jsearch.MONTHLY_CALL_BUDGET} left"
            if remaining <= 0:
                budget += "  ** SPENT -- source is skipping calls **"
            elif remaining <= 20:
                budget += "  ** low **"

        print(f"{source:16} {record.get('period', '?'):10} {calls:7}  {last:26} {budget}")

    current = quota.current_period()
    used = quota.usage(jsearch.SOURCE_NAME)
    print()
    print(
        f"JSearch this month ({current}): {used} counted locally. "
        f"Self-imposed budget {jsearch.MONTHLY_CALL_BUDGET}, "
        f"provider tier {jsearch.MONTHLY_CALL_LIMIT} "
        f"({jsearch.MONTHLY_CALL_RESERVE} held in reserve)."
    )


if __name__ == "__main__":
    main()
