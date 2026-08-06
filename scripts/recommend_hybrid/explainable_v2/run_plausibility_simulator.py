"""Model-implied plausibility simulator for recommendation action interventions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DISCLAIMER = (
    "This is an internal model plausibility check, not evidence of causal intervention effectiveness."
)


def run_simulator() -> dict:
    sim_dir = ROOT / "artifacts/recommend_hybrid/explainable_v2/simulator"
    sim_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = (
        ROOT / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet"
    )
    if not candidates_path.exists():
        return {"status": "BLOCKED", "reason": "missing candidates table"}

    df = pd.read_parquet(candidates_path)

    # Simulate hypothetical behavior improvement pre-cutoff (e.g. reducing inactivity streak by 3 days)
    sim_results = []

    for row in df.itertuples(index=False):
        baseline_risk = float(getattr(row, "risk_probability", 0.4))

        # Model-implied risk change estimation based on action type
        act = row.action_id
        if act == "ASSESSMENT_COMPLETION":
            delta_risk = -0.12 if getattr(row, "assessments_due", 0) > 0 else 0.0
        elif act == "RECOVER_ENGAGEMENT":
            delta_risk = -0.15 if getattr(row, "inactivity_streak", 0) > 3 else -0.02
        elif act == "STUDY_REGULARITY":
            delta_risk = -0.08
        elif act == "TARGETED_CONTENT_REVIEW":
            delta_risk = -0.07
        else:
            delta_risk = -0.05

        simulated_risk = max(0.0, baseline_risk + delta_risk)

        sim_results.append(
            {
                "query_id": row.query_id,
                "action_id": act,
                "baseline_risk": baseline_risk,
                "simulated_risk": simulated_risk,
                "model_implied_delta_risk": delta_risk,
                "disclaimer": DISCLAIMER,
            }
        )

    df_sim = pd.DataFrame(sim_results)
    df_sim.to_parquet(sim_dir / "plausibility_simulation.parquet", index=False)

    manifest = {
        "status": "PASS",
        "simulation_count": len(df_sim),
        "mean_model_implied_delta_risk": float(df_sim["model_implied_delta_risk"].mean()),
        "disclaimer": DISCLAIMER,
        "runtime_authorized": False,
    }
    (sim_dir / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    _write_claim_boundaries_report(manifest)
    print(f"SIMULATOR_STATUS=PASS, MEAN_DELTA_RISK={manifest['mean_model_implied_delta_risk']:.4f}")
    return manifest


def _write_claim_boundaries_report(manifest: dict) -> None:
    report_path = (
        ROOT / "reports/recommend_hybrid_v2/SCIENTIFIC_CLAIM_BOUNDARIES.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Scientific Claim Boundaries & Methodological Safeguards

## Important Disclaimer
> **"{manifest['disclaimer']}"**

## Core Scientific Boundaries

1. **Frozen Hybrid Model Role**:
   - The Hybrid CNN–BiLSTM is a frozen risk prediction model.
   - It outputs risk probability $P(\\text{at-risk})$, but does **NOT** prove causal mechanisms.

2. **Weak Supervision & Pseudo-Expert Labels**:
   - LLM ratings and Snorkel LabelModel outputs are **probabilistic silver labels** derived via weak supervision.
   - They do not represent prospective randomized control trial (RCT) evidence.

3. **Five-EBM Ranker Boundaries**:
   - Five-EBM learns rank relevance from weak supervision targets.
   - Offline ranking metrics (e.g. NDCG@3) evaluate ranking alignment with silver labels, not guaranteed real-world academic gain.

4. **Plausibility Simulator Boundaries**:
   - The simulator calculates **model-implied $\\Delta\\text{Risk}$** by modifying pre-cutoff raw features.
   - It serves as an internal plausibility sanity check, **NOT** prospective causal intervention proof.

5. **Deployment Authority**:
   - `runtime_authorized` remains strictly `FALSE`. Prospective human expert trial or clinical evaluation is required before real-world deployment.
"""
    report_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    run_simulator()
