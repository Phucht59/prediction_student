"""Audit action-prior concentration and learner-state variation in evaluation traces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "artifacts/recommend_hybrid/counterfactual/evaluation_rows.csv"
ACTION_SCORES = ROOT / "artifacts/recommend_hybrid/counterfactual/action_scores.csv"
OUT = ROOT / "artifacts/recommend_hybrid/counterfactual/shortcut_diagnostic.json"
REPORT = ROOT / "reports/recommend_hybrid/COUNTERFACTUAL_SHORTCUT_DIAGNOSTIC.md"


def _entropy(values: pd.Series) -> float:
    counts = values.value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum()) if not counts.empty else 0.0


def main() -> int:
    rows = pd.read_csv(EVALUATION)
    scores = pd.read_csv(ACTION_SCORES)
    scored = rows.loc[rows["status"].eq("COUNTERFACTUAL_SCORED")].copy()
    scored["baseline_band"] = pd.qcut(
        scored["baseline_risk"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    top_frequency = scored["top_action_id"].value_counts().to_dict()
    top_share = float(scored["top_action_id"].value_counts(normalize=True).iloc[0])
    stage_action = (
        scored.groupby(["stage", "top_action_id"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    stage_entropy = {
        str(stage): _entropy(group["top_action_id"])
        for stage, group in scored.groupby("stage")
    }
    band_means = scored.groupby("baseline_band", observed=False)["top_risk_reduction"].mean()
    action_means = scores.groupby("action_id")["risk_reduction"].mean().sort_values(ascending=False)
    action_stage_means = scores.groupby(["stage", "action_id"])["risk_reduction"].mean()
    payload = {
        "schema_version": "counterfactual_shortcut_diagnostic_v1",
        "status": "PASS",
        "claim_boundary": "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT",
        "source_evaluation_rows": str(EVALUATION.relative_to(ROOT)).replace("\\", "/"),
        "source_action_scores": str(ACTION_SCORES.relative_to(ROOT)).replace("\\", "/"),
        "records": int(len(rows)),
        "scored_records": int(len(scored)),
        "action_identity": {
            "unique_top_actions": int(scored["top_action_id"].nunique()),
            "top_action_frequency": {str(k): int(v) for k, v in top_frequency.items()},
            "top_action_share": top_share,
            "max_action_mean_risk_reduction": float(action_means.iloc[0]),
            "min_action_mean_risk_reduction": float(action_means.iloc[-1]),
            "action_mean_risk_reduction_range": float(action_means.max() - action_means.min()),
        },
        "learner_state_and_stage_variation": {
            "stage_top_action_entropy_bits": stage_entropy,
            "baseline_band_mean_top_risk_reduction": {
                str(k): float(v) for k, v in band_means.dropna().items()
            },
            "baseline_band_mean_range": float(band_means.max() - band_means.min()),
            "stage_action_mean_risk_reduction_range": float(action_stage_means.max() - action_stage_means.min()),
        },
        "gates": {
            "action_identity_not_over_80_percent": bool(top_share <= 0.80),
            "multiple_top_actions_observed": bool(scored["top_action_id"].nunique() >= 2),
            "risk_reduction_not_constant": bool(float(scored["top_risk_reduction"].std()) > 0.0),
            "risk_reduction_not_mostly_zero": bool(float((scored["top_risk_reduction"] == 0.0).mean()) < 0.50),
        },
        "scope_note": "This is a trace-level shortcut audit. It reports action/stage/baseline variation from frozen-model evaluation traces; it does not claim a causal effect and does not replace an input-level model permutation experiment.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Counterfactual Shortcut Diagnostic",
        "",
        f"- Status: **{payload['status']}**",
        f"- Records / scored: `{len(rows)}` / `{len(scored)}`",
        f"- Top-action concentration: `{top_share:.4f}`",
        f"- Unique top actions: `{scored['top_action_id'].nunique()}`",
        f"- Top risk-reduction standard deviation: `{scored['top_risk_reduction'].std():.6f}`",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name} | {'PASS' if value else 'FAIL'} |"
        for name, value in payload["gates"].items()
    )
    lines += ["", payload["scope_note"], ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(payload["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
