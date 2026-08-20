from pathlib import Path
import glob
from app.services.extraction.text_extract import extract_text
from app.services.extraction.layout import layout_signals
from app.services.scoring.formatting import format_score

for p in sorted(glob.glob("tests/fixtures/*.pdf")):
    b = Path(p).read_bytes()
    t = extract_text(b, p)["text"]
    l = layout_signals(b)
    r = format_score(t, l, ["experience", "education", "skills"])
    issues = [i["check"] for i in r["issues"]]
    print(f"{Path(p).name:24} {r['score']*100:5.1f}  {issues}")
