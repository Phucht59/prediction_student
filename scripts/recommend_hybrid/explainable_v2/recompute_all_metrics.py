"""Recompute all ranking metrics from prediction-level artifacts."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRED_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/predictions"
REPORT_PATH = ROOT / "reports/recommend_hybrid_v2/METRIC_RECOMPUTATION_REPORT.md"


def _ndcg_at_k(relevance: list[float], scores: list[float], k: int) -> float:
    if not relevance:
        return float("nan")
    paired = sorted(zip(scores, relevance), reverse=True)[:k]
    dcg = sum(rel / np.log2(i + 2) for i, (_, rel) in enumerate(paired))
    ideal_rel = sorted(relevance, reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rel))
    return dcg / idcg if idcg > 0 else 0.0


def recompute(pred_path: Path) -> dict:
    if not pred_path.exists():
        return {"status": "NO_PREDICTION_ARTIFACT", "path": str(pred_path)}

    df = pd.read_parquet(pred_path)
    required = {"query_id", "action_id", "score", "relevance"}
    missing = required - set(df.columns)
    if missing:
        return {"status": "MISSING_COLUMNS", "missing": list(missing)}

    ndcg3_vals = []
    p1_vals = []
    for qid, grp in df.groupby("query_id"):
        rel = grp["relevance"].tolist()
        sc = grp["score"].tolist()
        ndcg3_vals.append(_ndcg_at_k(rel, sc, 3))
        top1_idx = int(np.argmax(sc))
        p1_vals.append(float(rel[top1_idx] > 0))

    return {
        "status": "RECOMPUTED",
        "ndcg_at_3": float(np.nanmean(ndcg3_vals)) if ndcg3_vals else None,
        "precision_at_1": float(np.mean(p1_vals)) if p1_vals else None,
        "query_count": len(ndcg3_vals),
    }


if __name__ == "__main__":
    pred_files = list(PRED_DIR.glob("*.parquet")) if PRED_DIR.exists() else []
    results = {}
    for pf in pred_files:
        results[pf.name] = recompute(pf)

    if not results:
        results["status"] = "NO_PREDICTION_FILES_FOUND"
        results["message"] = "Cannot recompute — no prediction-level artifacts exist"

    print(json.dumps(results, indent=2))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Metric Recomputation Report\n\n"
        + json.dumps(results, indent=2)
        + "\n\nStatus: "
        + ("METRIC_RECOMPUTATION_PASS" if any(v.get("status") == "RECOMPUTED" for v in results.values() if isinstance(v, dict)) else "METRIC_RECOMPUTATION_BLOCKED_NO_PREDICTIONS"),
        encoding="utf-8",
    )
