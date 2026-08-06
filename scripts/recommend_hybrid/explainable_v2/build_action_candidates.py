"""Generate five candidates only after the gated feature table exists."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]
def main():
    out=ROOT/"artifacts/recommend_hybrid/explainable_v2/data"
    out.mkdir(parents=True,exist_ok=True)
    (out/"ACTION_CANDIDATES_MANIFEST.json").write_text(json.dumps({"status":"BLOCKED","reason":"feature table unavailable","runtime_authorized":False},indent=2)+"\n",encoding="utf-8")
    return 2
if __name__=="__main__": raise SystemExit(main())
