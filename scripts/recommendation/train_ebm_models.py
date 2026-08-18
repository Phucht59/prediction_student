"""Train/freeze five confidence-weighted EBM models and fair Panel-A baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.feasibility.rules_v2 import a4_feasibility_audit, build_feasibility_frame_v2  # noqa: E402
from src.recommendation.models.baselines import fit_random_forest, fit_ridge, stage_column, stage_prior_predict  # noqa: E402
from src.recommendation.models.datasets import feature_matrix  # noqa: E402
from src.recommendation.models.ebm import fit_ebm, global_importances, save_model, top_local_reasons  # noqa: E402
from src.recommendation.models.features import ACTION_TO_KEY, APPROVED_FEATURES, load_phase8_config, validate_phase7_authority  # noqa: E402
from src.recommendation.models.train import search_ebm  # noqa: E402
from src.recommendation.weak_supervision.matrix import FINAL_ACTIONS  # noqa: E402
from src.recommendation.weak_supervision.silver import sha256_file, write_json  # noqa: E402


def _representative_cases(oof: pd.DataFrame) -> list[str]:
    ordered = oof.sort_values("absolute_error")
    picks = [ordered.iloc[0]["case_id"], ordered.iloc[len(ordered) // 2]["case_id"], ordered.iloc[-1]["case_id"]]
    return [str(value) for value in picks]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/recommendation/phase8.yaml")
    parser.add_argument("--silver", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/silver_labels.parquet")
    parser.add_argument("--phase7-manifest", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/phase7_manifest.json")
    parser.add_argument("--phase6-manifest", type=Path, default=ROOT / "artifacts/recommendation/labeling/phase6_source_manifest.json")
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--training-dir", type=Path, default=ROOT / "artifacts/recommendation/models/training")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/recommendation/models/ebm")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "artifacts/recommendation/models/baselines")
    parser.add_argument("--oof-dir", type=Path, default=ROOT / "artifacts/recommendation/models/oof")
    parser.add_argument("--explain-dir", type=Path, default=ROOT / "reports/recommendation/phase8_explanations")
    parser.add_argument("--run-output", type=Path, default=ROOT / "artifacts/recommendation/models/phase8_run.json")
    args = parser.parse_args()
    phase8 = load_phase8_config(args.config)
    phase7 = validate_phase7_authority(args.phase7_manifest, args.silver, args.phase6_manifest)
    panel_a = pd.read_parquet(args.panel_a)
    feas_v2 = build_feasibility_frame_v2(panel_a)
    feas_path = ROOT / "artifacts/recommendation/feasibility/panel_feasibility_v2.parquet"
    feas_path.parent.mkdir(parents=True, exist_ok=True)
    feas_v2.to_parquet(feas_path, index=False)
    run = {"actions": {}, "a4_feasibility_audit": a4_feasibility_audit(), "features": list(APPROVED_FEATURES), "phase7": phase7["version"]}
    oof_frames = []
    for action_id in FINAL_ACTIONS:
        key = ACTION_TO_KEY[action_id]
        table = pd.read_parquet(args.training_dir / f"{key}_training.parquet")
        if action_id == "retrieval_practice" and not table["silver_status"].eq("REVIEW").all():
            raise ValueError("A5 training rows must retain REVIEW status")
        if table["silver_status"].eq("NO_WEAK_EVIDENCE").any():
            raise ValueError(f"{action_id} training contains NO_WEAK_EVIDENCE")
        result = search_ebm(table, phase8, action_id)
        X, y, w = feature_matrix(table)
        ebm = fit_ebm(X, y, sample_weight=w, config=result["selected_config"])
        model_path = args.model_dir / f"{key}_ebm.pkl"
        save_model(ebm, model_path)
        ridge = fit_ridge(X, y, w, alpha=float(phase8["ridge"]["alpha"]))
        forest = fit_random_forest(X, y, w, config=phase8["random_forest"])
        args.baseline_dir.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"type": "RIDGE", "model": ridge, "features": list(APPROVED_FEATURES)}, args.baseline_dir / f"{key}_ridge.pkl")
        pd.to_pickle({"type": "RANDOM_FOREST", "model": forest, "features": list(APPROVED_FEATURES)}, args.baseline_dir / f"{key}_rf.pkl")
        prior = {float(stage): float(np.average(y[stage_column(X) == stage], weights=w[stage_column(X) == stage])) for stage in np.unique(stage_column(X))}
        pd.to_pickle({"type": "ACTION_STAGE_PRIOR", "priors": prior, "global": float(np.average(y, weights=w))}, args.baseline_dir / f"{key}_stage_prior.pkl")
        oof_path = args.oof_dir / f"{key}_oof.parquet"
        args.oof_dir.mkdir(parents=True, exist_ok=True)
        result["oof"].to_parquet(oof_path, index=False)
        oof_frames.append(result["oof"])
        importance = global_importances(ebm)
        args.explain_dir.mkdir(parents=True, exist_ok=True)
        importance.to_json(args.explain_dir / f"{key}_global.json", orient="records", indent=2)
        local_rows = []
        by_case = table.set_index("case_id")
        for case_id in _representative_cases(result["oof"]):
            x = by_case.loc[case_id, list(APPROVED_FEATURES)].to_numpy(dtype=float)
            item = top_local_reasons(ebm, x)
            item["case_id"] = case_id
            item["action_id"] = action_id
            item["source"] = "PANEL_A_OOF_REPRESENTATIVE"
            local_rows.append(item)
        (args.explain_dir / f"{key}_local.json").write_text(json.dumps(local_rows, indent=2, sort_keys=True), encoding="utf-8")
        excluded = int((pd.read_parquet(args.silver).query("action_id == @action_id")["silver_status"] == "NO_WEAK_EVIDENCE").sum())
        run["actions"][action_id] = {
            "action_key": key,
            "artifact_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "training_rows": int(len(table)),
            "excluded_no_evidence_rows": excluded,
            "feature_names": list(APPROVED_FEATURES),
            "selected_config": result["selected_config"],
            "cv_strategy": {"n_splits": result["n_splits"], "reason": result["cv_reason"], "seed": phase8["cv"]["seed"]},
            "cv_metrics": result["cv_metrics"],
            "unweighted_metrics": result["unweighted_metrics"],
            "baseline_metrics": result["baseline_metrics"],
            "quality_status": phase7["status_by_action"][action_id],
            "quality_reasons": phase7["quality_reasons_by_action"][action_id],
            "silver_lineage": {
                "aggregator_type": table["aggregator_type"].iloc[0],
                "label_model_version": table["label_model_version"].iloc[0],
                "phase6_source_manifest_version": table["phase6_source_manifest_version"].iloc[0],
            },
            "target_stats": {"mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()), "max": float(y.max())},
            "weight_stats": {"mean": float(w.mean()), "std": float(w.std()), "min": float(w.min()), "max": float(w.max())},
            "checksum": sha256_file(model_path),
            "top_global_terms": importance.head(5).to_dict(orient="records"),
        }
    pd.concat(oof_frames, ignore_index=True).to_parquet(args.oof_dir / "all_actions_oof.parquet", index=False)
    write_json(args.run_output, run)
    print(json.dumps({action: item["cv_metrics"]["mae"] for action, item in run["actions"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
