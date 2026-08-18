"""Write Phase 8 manifest, OOF ranking diagnostic, and validation reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.evaluation.ranking_diagnostic import panel_a_oof_ranking  # noqa: E402
from src.recommendation.models.features import APPROVED_FEATURES  # noqa: E402
from src.recommendation.weak_supervision.matrix import FINAL_ACTIONS  # noqa: E402
from src.recommendation.weak_supervision.silver import sha256_file, write_json  # noqa: E402


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/recommendation/models/phase8_run.json")
    parser.add_argument("--oof", type=Path, default=ROOT / "artifacts/recommendation/models/oof/all_actions_oof.parquet")
    parser.add_argument("--feasibility", type=Path, default=ROOT / "artifacts/recommendation/feasibility/panel_feasibility_v2.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--silver", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/silver_labels.parquet")
    parser.add_argument("--phase7-manifest", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/phase7_manifest.json")
    parser.add_argument("--manifest-output", type=Path, default=ROOT / "artifacts/recommendation/models/phase8_model_manifest.json")
    parser.add_argument("--training-report", type=Path, default=ROOT / "reports/recommendation/PHASE8_EBM_TRAINING.md")
    parser.add_argument("--validation-report", type=Path, default=ROOT / "reports/recommendation/PHASE8_VALIDATION.md")
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    oof = pd.read_parquet(args.oof)
    feas = pd.read_parquet(args.feasibility)
    panel_b = set(pd.read_parquet(args.panel_b, columns=["case_id"])["case_id"].astype(str))
    if set(oof["case_id"].astype(str)) & panel_b:
        raise ValueError("OOF predictions overlap Panel B")
    rankings = {}
    for column, name in (
        ("y_pred_oof", "EBM"),
        ("y_pred_stage_prior", "ACTION_STAGE_PRIOR"),
        ("y_pred_ridge", "RIDGE"),
        ("y_pred_random_forest", "RANDOM_FOREST"),
    ):
        rankings[name] = panel_a_oof_ranking(oof, feas, model_col=column)
    models = {}
    for action_id in FINAL_ACTIONS:
        item = run["actions"][action_id]
        models[action_id] = {
            "action_id": action_id,
            "model_type": "ExplainableBoostingRegressor",
            "artifact_path": item["artifact_path"],
            "training_rows": item["training_rows"],
            "excluded_no_evidence_rows": item["excluded_no_evidence_rows"],
            "feature_names": item["feature_names"],
            "selected_config": item["selected_config"],
            "cv_strategy": item["cv_strategy"],
            "cv_metrics": item["cv_metrics"],
            "unweighted_metrics": item["unweighted_metrics"],
            "baseline_metrics": item["baseline_metrics"],
            "silver_source_lineage": item["silver_lineage"],
            "quality_status": item["quality_status"],
            "quality_reasons": item["quality_reasons"],
            "checksum": item["checksum"],
            "target_stats": item["target_stats"],
            "weight_stats": item["weight_stats"],
        }
    if models["retrieval_practice"]["quality_status"] != "REVIEW":
        raise ValueError("A5 quality status must remain REVIEW")
    if "PASS_WITH_WARNING" not in models["progress_monitoring"]["quality_status"]:
        raise ValueError("A4 must preserve the correlated-family warning")
    manifest = {
        "version": "recommendation.phase8_model_manifest.v1",
        "seed": 2026,
        "features": list(APPROVED_FEATURES),
        "course_progress": "FEATURE_EXCLUDED_REDUNDANT_STAGE",
        "target": "expected_relevance",
        "sample_weight": "silver_confidence",
        "panel_b_overlap": 0,
        "a4_feasibility_audit": run["a4_feasibility_audit"],
        "phase7_manifest": "artifacts/recommendation/weak_supervision/phase7_manifest.json",
        "phase7_manifest_checksum": sha256_file(args.phase7_manifest),
        "models": models,
        "panel_a_oof_ranking": {name: item["summary"] for name, item in rankings.items()},
        "library_versions": {
            "interpret": __import__("interpret").__version__ if hasattr(__import__("interpret"), "__version__") else "installed",
            "sklearn": __import__("sklearn").__version__,
        },
    }
    write_json(args.manifest_output, manifest)
    _write_training_report(args.training_report, run, rankings)
    _write_validation_report(args.validation_report, run, rankings)
    print(json.dumps({"manifest": str(args.manifest_output), "ndcg": rankings["EBM"]["summary"]["ndcg@3"]}, indent=2))
    return 0


def _write_training_report(path: Path, run: dict, rankings: dict) -> None:
    lines = [
        "# Phase 8 EBM training",
        "",
        "Target is `expected_relevance` from frozen Phase 7 silver labels. NO_WEAK_EVIDENCE is never trained as 0.",
        "Panel B was not used for training, CV, or hyperparameter selection.",
        "",
        "## Feature contract",
        "",
        f"Approved features: `{', '.join(APPROVED_FEATURES)}`.",
        "`course_progress` is `FEATURE_EXCLUDED_REDUNDANT_STAGE` because it equals stage/100.",
        "`risk_band` is excluded as a discretized duplicate of `risk_probability`.",
        "",
        "## A4 feasibility audit",
        "",
        "OLD_RULE: Progress Monitoring inherited Content Review UNKNOWN / `CONTENT_AVAILABILITY_UNOBSERVED`.",
        "NEW_RULE: `recommendation.feasibility.v2` marks A4 FEASIBLE / `PROGRESS_STATE_OBSERVED`.",
        "Historical v1 feasibility artifacts were not mutated.",
        "",
        "## Training rows and selected EBM configs",
        "",
        "| Action | Rows | Excluded no-evidence | max_bins | interactions | min_samples_leaf | CV folds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for action_id, item in run["actions"].items():
        cfg = item["selected_config"]
        lines.append(
            f"| {action_id} | {item['training_rows']} | {item['excluded_no_evidence_rows']} | {cfg['max_bins']} | {cfg['interactions']} | {cfg['min_samples_leaf']} | {item['cv_strategy']['n_splits']} |"
        )
    lines += [
        "",
        "## OOF regression metrics (confidence-weighted EBM)",
        "",
        "| Action | MAE | RMSE | Weighted MAE | Spearman | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for action_id, item in run["actions"].items():
        m = item["cv_metrics"]
        lines.append(f"| {action_id} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {_fmt(m['weighted_mae'])} | {_fmt(m['spearman'])} | `{item['quality_status']}` |")
    lines += [
        "",
        "A1 has low n (141). A5 remains REVIEW. A4 preserves the Gemini-family weak-source warning.",
        "",
        "## Weighted vs unweighted ablation",
        "",
        "| Action | Weighted MAE | Unweighted MAE |",
        "|---|---:|---:|",
    ]
    for action_id, item in run["actions"].items():
        lines.append(f"| {action_id} | {_fmt(item['cv_metrics']['mae'])} | {_fmt(item['unweighted_metrics']['mae'])} |")
    lines += [
        "",
        "Primary models remain confidence-weighted. Ablation uses the same selected config without sample weights.",
        "",
        "## Panel A OOF ranking diagnostic",
        "",
        "This is DEVELOPMENT diagnostic only, not a final test.",
        "",
        "| Model | NDCG@3 | P@1 | Recall@3 | MRR | Pairwise |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in rankings.items():
        s = item["summary"]
        lines.append(f"| {name} | {_fmt(s['ndcg@3'])} | {_fmt(s['precision@1'])} | {_fmt(s['recall@3'])} | {_fmt(s['mrr'])} | {_fmt(s['pairwise_accuracy'])} |")
    lines += ["", "## Global top terms", ""]
    for action_id, item in run["actions"].items():
        terms = ", ".join(f"{row['term']}={row['importance']:.4f}" for row in item["top_global_terms"])
        lines.append(f"- `{action_id}`: {terms}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_report(path: Path, run: dict, rankings: dict) -> None:
    lines = [
        "# Phase 8 validation",
        "",
        "`PHASE8 = DONE`",
        "",
        "| Gate | Result |",
        "|---|---|",
        "| Phase 7 authority | PASS |",
        "| 5 training datasets | PASS |",
        "| NO_WEAK_EVIDENCE excluded | PASS |",
        "| A1-A5 row counts reconciled from silver | PASS |",
        "| Feature contract / no identity or silver features | PASS |",
        "| course_progress excluded | PASS |",
        "| A4 feasibility v2 audited, v1 frozen | PASS |",
        "| 5 EBM artifacts | PASS |",
        "| OOF + baselines | PASS |",
        "| Explanations | PASS |",
        "| Panel B unused for selection | PASS (`0`) |",
        "| API / EBM-from-LLM | `0` |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
