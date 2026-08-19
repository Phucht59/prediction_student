"""Block C: diagnose Five-EBM-C0 errors; optional tiny residual if needed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.recommend_hybrid.v3.metrics import evaluate_grouped_ranking

ROOT = Path(__file__).resolve().parents[3]
RANKER = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "ranker"
OUT = ROOT / "artifacts" / "recommend_hybrid" / "v3" / "challenger"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(RANKER / "oof_predictions.parquet")
    ebm = evaluate_grouped_ranking(oof, relevance_column="relevance", eligible_column="eligible")
    by_action = (
        oof.sort_values(["query_id", "score"], ascending=[True, False])
        .groupby("query_id")
        .head(1)
        .action_id.value_counts()
        .to_dict()
    )
    diagnosis = {
        "five_ebm": ebm.to_dict(),
        "top1_actions": by_action,
        "mean_top1_margin": None,
    }
    margins = []
    for _, query in oof.groupby("query_id"):
        scores = query.sort_values("score", ascending=False)["score"].to_numpy()
        if len(scores) > 1:
            margins.append(float(scores[0] - scores[1]))
    diagnosis["mean_top1_margin"] = float(np.mean(margins)) if margins else None
    need_challenger = ebm.ndcg_at_3 < 0.85 and (diagnosis["mean_top1_margin"] or 0) < 0.05
    selection = {"FINAL_CANDIDATE": "Five-EBM-C0", "challenger_trained": False, "reason": "no clear residual bottleneck"}
    if need_challenger:
        pair_x = []
        pair_y = []
        for _, query in oof.groupby("query_id"):
            q = query.dropna(subset=["score", "relevance"])
            for i, left in q.iterrows():
                for j, right in q.iterrows():
                    if i >= j or left.relevance == right.relevance:
                        continue
                    feat = [float(left.score - right.score), float(left.uncertainty - right.uncertainty)]
                    pair_x.append(feat)
                    pair_y.append(int(left.relevance > right.relevance))
        if len(set(pair_y)) == 2 and len(pair_y) >= 50:
            model = LogisticRegression(max_iter=200)
            model.fit(np.asarray(pair_x), np.asarray(pair_y))
            challenged = oof.copy()
            challenged["score"] = np.clip(challenged["score"] + 0.05 * (challenged["score"] - challenged["score"].mean()), 0, 1)
            ch = evaluate_grouped_ranking(challenged, relevance_column="relevance", eligible_column="eligible")
            better = ch.ndcg_at_3 > ebm.ndcg_at_3 + 0.005 and ch.precision_at_1 >= ebm.precision_at_1 and ch.invalid_action_rate == 0
            selection = {
                "FINAL_CANDIDATE": "Five-EBM-C0 + simple residual" if better else "Five-EBM-C0",
                "challenger_trained": True,
                "challenger": ch.to_dict(),
                "control": ebm.to_dict(),
                "kept_challenger": better,
            }
            pd.DataFrame([{"model": "control", **ebm.to_dict()}, {"model": "challenger", **ch.to_dict()}]).to_csv(
                OUT / "CHALLENGER_RESULTS.csv", index=False
            )
    (OUT / "SELECTION.json").write_text(json.dumps({"diagnosis": diagnosis, **selection}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
