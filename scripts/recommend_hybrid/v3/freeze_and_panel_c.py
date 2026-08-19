"""Development freeze + Panel C protocol. Does not call Gemini."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V3 = ROOT / "artifacts" / "recommend_hybrid" / "v3"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    freeze_dir = V3 / "freeze"
    panel_dir = V3 / "panel_c"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    tracked = [
        V3 / "data" / "C0_PREDICTION_PROVENANCE.json",
        V3 / "data" / "FEATURE_MANIFEST.json",
        V3 / "data" / "learner_stage_features.parquet",
        V3 / "labels" / "LABEL_PORTABILITY_SUMMARY.json",
        V3 / "ranker" / "FIVE_EBM_MANIFEST.json",
        V3 / "ranker" / "BASELINE_RESULTS.csv",
        V3 / "router" / "ROUTER_CONFIG.json",
    ]
    checksums = {str(path.relative_to(ROOT)): sha(path) for path in tracked if path.exists()}
    freeze = {
        "status": "FROZEN_DEVELOPMENT",
        "prediction_authority": "Phase4 Hybrid C0",
        "ranker": "Five-EBM-C0",
        "panel_b_used_for_tuning": False,
        "gemini_full_relabel": False,
        "checksums": checksums,
        "post_freeze_tuning_permitted": False,
    }
    (freeze_dir / "DEVELOPMENT_FREEZE_MANIFEST.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    features = pd.read_parquet(V3 / "data" / "learner_stage_features.parquet")
    labeled = pd.read_parquet(V3 / "labels" / "v3_action_rows.parquet")
    used_students = set(labeled.loc[labeled.portability_status.eq("CONDITIONALLY_PORTABLE"), "student_key"].astype(str))
    pool = features.loc[~features.student_key.astype(str).isin(used_students)].drop_duplicates("student_key")
    rng = pd.Series(pool.student_key.astype(str).unique())
    sample = rng.sample(n=min(150, len(rng)), random_state=2026).tolist()
    cases = features.loc[features.student_key.astype(str).isin(sample)].drop_duplicates("query_id")
    cases[["query_id", "student_key", "course_key", "stage", "cutoff_day"]].to_parquet(panel_dir / "PANEL_C_SAMPLED_CASES.parquet", index=False)
    protocol = {
        "name": "PANEL_C",
        "heldout": True,
        "n_students_requested": 150,
        "n_students_sampled": int(len(sample)),
        "student_overlap_with_portable_panel_a": 0,
        "gemini_required": True,
        "gemini_executed": False,
        "prompt_must_not_include": [
            "risk_probability",
            "predicted_risk",
            "risk_band",
            "uncertainty",
            "model_id",
            "final_result",
        ],
        "historical_panel_b_is_not_v3_heldout": True,
        "status": "SAMPLED_AWAITING_GEMINI",
    }
    (panel_dir / "PANEL_C_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (panel_dir / "PANEL_C_FINAL_RESULTS.json").write_text(
        json.dumps({"status": "NOT_EVALUATED", "reason": "Gemini Panel C reviews not authorized in this run"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("FREEZE", freeze["status"], "PANEL_C", protocol["status"])


if __name__ == "__main__":
    main()
