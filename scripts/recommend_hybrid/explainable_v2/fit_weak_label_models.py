"""Train-only weak-label aggregation entry point."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]
def main():
    p=ROOT/"artifacts/recommend_hybrid/explainable_v2/labels/SOURCE_AUDIT.json"; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"status":"BLOCKED","reason":"feature table and real annotation inputs unavailable","llm_weak_labels":"MISSING","runtime_authorized":False},indent=2)+"\n",encoding="utf-8"); return 2
if __name__=="__main__": raise SystemExit(main())
