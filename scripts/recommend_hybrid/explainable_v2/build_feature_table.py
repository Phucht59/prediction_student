from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.recommend_hybrid.explainable_v2.data_builder import build, write_blocked_manifest

def main() -> int:
    root = ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=root / "artifacts/recommend_hybrid/explainable_v2/data/learner_stage_features.parquet")
    p.add_argument("--lineage", type=Path, default=root / "artifacts/recommend_hybrid/explainable_v2/data/feature_lineage.parquet")
    p.add_argument("--manifest", type=Path, default=root / "artifacts/recommend_hybrid/explainable_v2/data/FEATURE_TABLE_MANIFEST.json")
    a = p.parse_args()
    try: build(a.output, a.lineage, a.manifest)
    except RuntimeError as exc:
        write_blocked_manifest(a.manifest, str(exc)); print(str(exc)); return 2
    return 0
if __name__ == "__main__": raise SystemExit(main())
