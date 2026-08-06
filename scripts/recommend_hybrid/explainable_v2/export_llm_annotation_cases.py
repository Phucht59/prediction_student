"""Export blinded cases when real feature data exists; never emits labels."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]
def main():
    p=ROOT/"artifacts/recommend_hybrid/explainable_v2/labels/LLM_CASE_EXPORT_MANIFEST.json"; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"status":"BLOCKED","reason":"validated feature table unavailable","contains_labels":False,"runtime_authorized":False},indent=2)+"\n",encoding="utf-8"); return 2
if __name__=="__main__": raise SystemExit(main())
