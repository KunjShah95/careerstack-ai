import sys, glob
from pathlib import Path as P
sys.path.insert(0, str(P(__file__).resolve().parents[1]))
import asyncio, json
from app.services.extraction.text_extract import extract_text
from app.services.extraction.llm_parse import parse_resume
from app.services.roleprofile.cache import get_or_mine
from app.services.scoring.keywords import keyword_score

prof = asyncio.run(get_or_mine("backend developer", "India"))
freqs = prof["skill_frequencies"]
print(f"profile: {prof['postings_sampled']} postings, {len(freqs)} skills\n")

for p in sorted(glob.glob("tests/fixtures/*.pdf")):
    text = extract_text(P(p).read_bytes(), p)["text"]
    r = parse_resume(text)
    k = keyword_score(freqs, text, r.skills)
    got = ", ".join(m["skill"] for m in k["matched"][:6]) or "none"
    print(f"{P(p).name:24} {k['score']*100:5.1f}   {got}")
