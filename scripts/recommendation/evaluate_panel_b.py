"""Evaluate frozen models on Panel B automated references. Never retunes models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.evaluation.metrics import clip_score, mae, spearman  # noqa: E402
from src.recommendation.evaluation.ranking_diagnostic import aggregate_case_metrics, evaluate_case_ranking  # noqa: E402
from src.recommendation.labeling.panel_b_reference import (  # noqa: E402
    build_panel_b_reference_table,
    pairwise_reference_agreement,
)
from src.recommendation.models.features import ACTION_TO_KEY, APPROVED_FEATURES, encode_state_features  # noqa: E402
from src.recommendation.ranking.ranker import rank_actions  # noqa: E402
from src.recommendation.weak_supervision.matrix import FINAL_ACTIONS  # noqa: E402
from src.recommendation.weak_supervision.silver import write_json  # noqa: E402
from scripts.recommendation.bootstrap_panel_b import bootstrap_case_metrics, percentile_ci  # noqa: E402


RAW_CANDIDATES = {
    "REF_GEMINI35": ROOT / "artifacts/recommendation/labeling/raw/panel_b_reference_gemini35.jsonl",
    "REF_GEMINI31": ROOT / "artifacts/recommendation/labeling/raw/panel_b_reference_gemini31.jsonl",
}
BASELINE_DIR = ROOT / "artifacts/recommendation/models/baselines"
MODEL_NAMES = ("EBM", "ACTION_STAGE_PRIOR", "RIDGE", "RANDOM_FOREST")


def _write_blocked(output_dir: Path, missing: dict) -> int:
    write_json(output_dir / "phase9_manifest.json", {
        "version": "recommendation.phase9_manifest.v1",
        "phase9_code": "DONE",
        "phase9_data": "BLOCKED_PANEL_B_REFERENCE_API",
        "evaluation_name": "AUTOMATED_REFERENCE_EVALUATION",
        "missing_raw_references": {name: str(path.as_posix()) for name, path in missing.items()},
    })
    print(json.dumps({"status": "BLOCKED_PANEL_B_REFERENCE_API", "missing": list(missing)}, indent=2))
    return 0


def _score_frozen_baselines(panel_b: pd.DataFrame) -> dict[str, pd.DataFrame]:
    features = encode_state_features(panel_b)
    X = features.to_numpy(dtype=float)
    stage = features["stage_code"].to_numpy(dtype=float)
    case_ids = panel_b["case_id"].astype(str).to_numpy()
    scored = {}
    for model_name, suffix in (("ACTION_STAGE_PRIOR", "stage_prior"), ("RIDGE", "ridge"), ("RANDOM_FOREST", "rf")):
        rows = []
        for action_id in FINAL_ACTIONS:
            blob = pd.read_pickle(BASELINE_DIR / f"{ACTION_TO_KEY[action_id]}_{suffix}.pkl")
            if model_name == "ACTION_STAGE_PRIOR":
                raw = np.asarray([blob["priors"].get(float(value), blob["global"]) for value in stage], dtype=float)
            else:
                raw = np.asarray(blob["model"].predict(X), dtype=float)
            clipped = np.asarray(clip_score(raw), dtype=float)
            for case_id, raw_score, relevance in zip(case_ids, raw, clipped):
                rows.append({"case_id": case_id, "action_id": action_id, "raw_score": float(raw_score), "relevance_score": float(relevance)})
        scored[model_name] = pd.DataFrame(rows)
    return scored


def _evaluate_model(scores: pd.DataFrame, reference: pd.DataFrame, feasibility: pd.DataFrame) -> tuple[list[dict], dict]:
    feas_map = {(str(row.case_id), str(row.action_id)): str(row.feasibility_status) for row in feasibility.itertuples(index=False)}
    case_rows = []
    for case_id, group in scores.groupby("case_id"):
        ref_group = reference[(reference["case_id"] == case_id) & reference["reference_status"].isin(["DUAL_SOURCE", "SINGLE_SOURCE"])]
        ref = {str(row.action_id): float(row.reference_relevance) for row in ref_group.itertuples(index=False)}
        score_map = {str(row.action_id): (float(row.raw_score), float(row.relevance_score)) for row in group.itertuples(index=False)}
        feas = {action: feas_map.get((str(case_id), action), "UNKNOWN") for action in score_map}
        metrics = evaluate_case_ranking(scores=score_map, feasibility=feas, reference=ref)
        metrics["case_id"] = str(case_id)
        case_rows.append(metrics)
    return case_rows, aggregate_case_metrics(case_rows)


def _action_diagnostics(scores: pd.DataFrame, reference: pd.DataFrame, feasibility: pd.DataFrame) -> dict:
    merged = scores.merge(reference, on=["case_id", "action_id"], how="left")
    if "feasibility_status" not in merged.columns:
        merged = merged.merge(feasibility[["case_id", "action_id", "feasibility_status"]], on=["case_id", "action_id"], how="left")
    report = {}
    for action_id, group in merged.groupby("action_id"):
        valid = group[group["reference_status"].isin(["DUAL_SOURCE", "SINGLE_SOURCE"])]
        y = valid["reference_relevance"].to_numpy(dtype=float) if len(valid) else np.array([])
        pred = valid["relevance_score"].to_numpy(dtype=float) if len(valid) else np.array([])
        report[action_id] = {
            "n_scored": int(len(group)),
            "n_valid_reference": int(len(valid)),
            "mean_score": float(group["relevance_score"].mean()),
            "mae": None if len(valid) == 0 else mae(y, pred),
            "spearman": None if len(valid) < 3 else spearman(y, pred),
            "feasibility_exclusion_rate": float(group["feasibility_status"].eq("INFEASIBLE").mean()),
            "top1_rate": float(group["rank"].eq(1).mean()) if "rank" in group.columns else None,
            "top3_rate": float(group["in_top_k"].mean()) if "in_top_k" in group.columns else None,
        }
    return report


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_reports(reports_dir: Path, metrics: dict, agreement: dict, bootstrap: dict, a5: dict) -> None:
    lines = [
        "# Phase 9 Panel B evaluation",
        "",
        "`PHASE9_DATA = DONE`",
        "",
        "This is an AUTOMATED_REFERENCE_EVALUATION against Gemini 3.5 + Gemini 3.1 weak references.",
        "It is not expert ground truth and not a causal claim about student outcomes.",
        "Gemini 3.5 and Gemini 3.1 are the same model family.",
        "",
        "## Reference agreement",
        "",
        f"- Overall exact agreement: `{agreement['_overall']['exact_agreement']}/{agreement['_overall']['n']}` ({agreement['_overall']['exact_agreement_rate']:.6f}).",
        "",
        "| Action | Exact | Linear kappa | Quadratic kappa | DUAL | SINGLE | NO_REFERENCE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for action_id in FINAL_ACTIONS:
        item = agreement[action_id]
        counts = item["reference_status_counts"]
        lines.append(
            f"| {action_id} | {_fmt(item['exact_agreement_rate'])} | {_fmt(item['linear_weighted_kappa'])} | {_fmt(item['quadratic_weighted_kappa'])} | {counts.get('DUAL_SOURCE', 0)} | {counts.get('SINGLE_SOURCE', 0)} | {counts.get('NO_REFERENCE', 0)} |"
        )
    lines += [
        "",
        "## Frozen model comparison",
        "",
        "| Model | NDCG@3 | 95% CI | P@1 | Recall@3 | MRR | Pairwise | Invalid | Coverage |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        item = metrics[name]
        ci = bootstrap["ndcg_ci"][name]
        lines.append(
            f"| {name} | {_fmt(item['ndcg@3'])} | [{_fmt(ci['low'])}, {_fmt(ci['high'])}] | {_fmt(item['precision@1'])} | {_fmt(item['recall@3'])} | {_fmt(item['mrr'])} | {_fmt(item['pairwise_accuracy'])} | {_fmt(item['invalid_action_rate'])} | {_fmt(item['coverage'])} |"
        )
    lines += ["", "## Paired NDCG@3 deltas vs EBM", "", "| Contrast | Mean delta | 95% CI |", "|---|---:|---|"]
    for name, item in bootstrap["deltas"].items():
        lines.append(f"| EBM - {name} | {_fmt(item['mean'])} | [{_fmt(item['low'])}, {_fmt(item['high'])}] |")
    lines += [
        "",
        "## A5 REVIEW",
        "",
        f"- A5 top-1 rate: `{_fmt(a5['top1_rate'])}`.",
        f"- A5 top-3 rate: `{_fmt(a5['top3_rate'])}`.",
        f"- REVIEW plan rate: `{_fmt(a5['review_plan_rate'])}`.",
        "A5 remains REVIEW and was not suppressed.",
        "",
        "No model was tuned on Panel B.",
    ]
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "PHASE9_PANEL_B_EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (reports_dir / "PHASE9_VALIDATION.md").write_text(
        "# Phase 9 validation\n\n`PHASE9_CODE = DONE`\n`PHASE9_DATA = DONE`\n\n"
        "Frozen Phase 8 models were applied to Panel B automated references. "
        "NO_REFERENCE actions were excluded from graded metrics and were not mapped to 0. "
        "Panel B was not used for training or hyperparameter selection.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, default=ROOT / "artifacts/recommendation/inference/panel_b_rankings.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/recommendation/evaluation")
    args = parser.parse_args()
    missing = {name: path for name, path in RAW_CANDIDATES.items() if not path.exists()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if missing or not args.rankings.exists():
        return _write_blocked(args.output_dir, missing)
    panel_b = pd.read_parquet(args.panel_b)
    panel_ids = set(panel_b["case_id"].astype(str))
    if len(panel_ids) != 150:
        raise ValueError("Panel B must contain 150 cases")
    reference = build_panel_b_reference_table(RAW_CANDIDATES["REF_GEMINI35"], RAW_CANDIDATES["REF_GEMINI31"], panel_ids)
    reference_path = args.output_dir / "panel_b_reference.parquet"
    reference.to_parquet(reference_path, index=False)
    agreement = pairwise_reference_agreement(reference)
    ebm = pd.read_parquet(args.rankings)
    ebm["case_id"] = ebm["case_id"].astype(str)
    feasibility = ebm[["case_id", "action_id", "feasibility_status"]].drop_duplicates()
    scored = {"EBM": ebm}
    scored.update(_score_frozen_baselines(panel_b))
    metrics = {}
    case_tables = {}
    for name in MODEL_NAMES:
        frame = scored[name].copy()
        if "rank" not in frame.columns:
            ranked_rows = []
            for case_id, group in frame.groupby("case_id"):
                rows = []
                for row in group.itertuples(index=False):
                    feas = feasibility.loc[(feasibility["case_id"] == str(case_id)) & (feasibility["action_id"] == row.action_id), "feasibility_status"]
                    rows.append({
                        "action_id": row.action_id,
                        "raw_score": float(row.raw_score),
                        "relevance_score": float(row.relevance_score),
                        "feasibility_status": str(feas.iloc[0]) if len(feas) else "UNKNOWN",
                    })
                for item in rank_actions(rows, top_k=3):
                    item["case_id"] = str(case_id)
                    ranked_rows.append(item)
            frame = pd.DataFrame(ranked_rows)
        cases, summary = _evaluate_model(frame, reference, feasibility)
        metrics[name] = summary
        case_tables[name] = pd.DataFrame(cases)
        case_tables[name]["model"] = name
    ebm_cases = case_tables["EBM"].copy()
    ebm_cases.to_parquet(args.output_dir / "panel_b_case_metrics.parquet", index=False)
    pd.concat(case_tables.values(), ignore_index=True).to_parquet(args.output_dir / "panel_b_case_metrics_all.parquet", index=False)
    boot_frames = {name: bootstrap_case_metrics(case_tables[name], iterations=2000, seed=2026) for name in MODEL_NAMES}
    ndcg_ci = {name: percentile_ci(boot_frames[name]["ndcg@3"]) for name in MODEL_NAMES}
    deltas = {}
    ebm_boot = boot_frames["EBM"]["ndcg@3"].to_numpy(dtype=float)
    for name in ("ACTION_STAGE_PRIOR", "RIDGE", "RANDOM_FOREST"):
        delta = ebm_boot - boot_frames[name]["ndcg@3"].to_numpy(dtype=float)
        deltas[name] = {"mean": float(np.mean(delta)), "low": float(np.percentile(delta, 2.5)), "high": float(np.percentile(delta, 97.5))}
    bootstrap = {"ndcg_ci": ndcg_ci, "deltas": deltas, "iterations": 2000, "seed": 2026}
    pd.DataFrame([{"model": name, **ndcg_ci[name]} for name in MODEL_NAMES]).to_parquet(args.output_dir / "panel_b_bootstrap.parquet", index=False)
    action_diag = _action_diagnostics(ebm, reference, feasibility)
    a5 = {
        "top1_rate": metrics["EBM"]["a5_top1_rate"],
        "top3_rate": metrics["EBM"]["a5_top3_rate"],
        "review_plan_rate": metrics["EBM"]["review_plan_rate"],
        "warning": "REVIEW",
    }
    payload = {
        "version": "recommendation.phase9_manifest.v1",
        "phase9_code": "DONE",
        "phase9_data": "DONE",
        "evaluation_name": "AUTOMATED_REFERENCE_EVALUATION",
        "panel_b_cases": 150,
        "reference_rows": int(len(reference)),
        "reference_status_counts": reference["reference_status"].value_counts().astype(int).to_dict(),
        "agreement": agreement,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "action_diagnostics": action_diag,
        "a5": a5,
        "panel_b_overlap_with_training": 0,
        "models_tuned_on_panel_b": False,
    }
    write_json(args.output_dir / "panel_b_metrics.json", payload)
    write_json(args.output_dir / "phase9_manifest.json", payload)
    _write_reports(ROOT / "reports/recommendation", metrics, agreement, bootstrap, a5)
    print(json.dumps({
        "status": "DONE",
        "reference_rows": int(len(reference)),
        "ndcg": {name: metrics[name]["ndcg@3"] for name in MODEL_NAMES},
        "ndcg_ci": ndcg_ci,
        "deltas": deltas,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
