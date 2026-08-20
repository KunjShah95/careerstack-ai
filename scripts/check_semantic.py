import sys, glob
from pathlib import Path as P
sys.path.insert(0, str(P(__file__).resolve().parents[1]))
import asyncio
from app.services.extraction.text_extract import extract_text
from app.services.roleprofile.cache import get_or_mine
from app.services.scoring.semantic import semantic_score

prof = asyncio.run(get_or_mine("backend developer", "India"))
reqs = prof.get("requirement_sentences", [])
tops = list(prof["skill_frequencies"].keys())
print(f"{len(reqs)} requirement sentences\n")

for p in sorted(glob.glob("tests/fixtures/*.pdf")):
    text = extract_text(P(p).read_bytes(), p)["text"]
    s = semantic_score(text, reqs, tops)
    print(f"{P(p).name:24} {s['score']*100:5.1f}  raw={s['raw_cosine_mean']:.3f}")
