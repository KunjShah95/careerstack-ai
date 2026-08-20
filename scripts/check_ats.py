import sys, glob
from pathlib import Path as P
sys.path.insert(0, str(P(__file__).resolve().parents[1]))
import asyncio
from app.services.extraction.text_extract import extract_text
from app.services.extraction.layout import layout_signals
from app.services.extraction.llm_parse import parse_resume
from app.services.roleprofile.cache import get_or_mine
from app.services.scoring.ats import compute_ats_score, build_action_plan

prof = asyncio.run(get_or_mine("backend developer", "India"))
print(f"profile: {prof['postings_sampled']} postings, {len(prof['skill_frequencies'])} skills\n")
print(f"{'fixture':24} {'ATS':>6} {'kw':>6} {'sem':>6} {'fmt':>6} {'exp':>6}  band")
print("-" * 76)

for p in sorted(glob.glob("tests/fixtures/*.pdf")):
    b = P(p).read_bytes()
    text = extract_text(b, p)["text"]
    r = compute_ats_score(parse_resume(text), text, layout_signals(b), prof)
    s = r["subscores"]
    print(f"{P(p).name:24} {r['overall_score']:6.1f} {s['keyword']:6.1f} "
          f"{s['semantic']:6.1f} {s['format']:6.1f} {s['experience']:6.1f}  {r['band']}")

b = P("tests/fixtures/demo_before.pdf").read_bytes()
text = extract_text(b, "demo_before.pdf")["text"]
r = compute_ats_score(parse_resume(text), text, layout_signals(b), prof)
plan = build_action_plan(r)
print(f"\ndemo_before plan keys: {list(plan.keys())}")
items = plan.get("actions") or plan.get("gaps") or plan.get("items") or []
for a in items:
    print(f"  +{a.get('estimated_gain', 0):5.1f}  {a.get('action', a)}")
