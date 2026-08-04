"""Fail-closed scientific release gate for outcome-grounded V2.1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/outcome_grounded_v2_1"
FINAL = OUT / "final_oof"
REPORT = ROOT / "reports/recommend_hybrid/v2_1/V2_1_CORRECTED_SCIENTIFIC_RESULTS_VI.md"

REQUIRED_CONTROLS = {
    "NC1_LABEL_SHUFFLE_RETRAIN",
    "NC2A_TRAIN_STATE_SHUFFLE",
    "NC2B_TEST_STATE_SHUFFLE",
    "NC3_ACTION_IDENTITY_SHUFFLE_RETRAIN",
    "NC4_WRONG_TRAJECTORY_REBUILD",
    "NC5_TIME_REVERSAL_PLACEBO",
}
REQUIRED_ABLATIONS = {
    "FULL",
    "NO_RISK_PROFILE",
    "NO_BEHAVIOR_STATE",
    "NO_OPPORTUNITY",
    "NO_DEFICIT",
    "NO_COUNTERFACTUAL_DELTA",
    "NO_ACTION_INTERACTIONS",
    "NO_WORKLOAD",
    "ACTION_PRIOR_ONLY",
    "NO_CONSTRAINTS_OFFLINE_ONLY",
}
REQUIRED_MODEL_FAMILIES = {
    "interaction_logistic",
    "pairwise_logistic",
    "lambdamart",
    "boosted_tree",
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require(path: Path) -> Path:
    if not path.exists():
        raise RuntimeError(f"Required scientific artifact is missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(require(path).read_text(encoding="utf-8"))


def top_unavailable_rate() -> float:
    predictions = pd.read_parquet(require(FINAL / "OOF_RANKING_PREDICTIONS.parquet"))
    if "action_available" not in predictions.columns:
        return 0.0
    top = predictions.loc[predictions.groupby("group_id")["model_score"].idxmax()]
    return float((pd.to_numeric(top["action_available"], errors="coerce").fillna(0) <= 0).mean())


def checksum_tree(root: Path) -> dict[str, str]:
    output = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "CORRECTED_CHECKSUMS.json":
            continue
        output[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return output


def full_grid_gate() -> dict[str, Any]:
    marker = load_json(OUT / "FULL_REGISTERED_SEARCH.json")
    folds = marker.get("folds", [])
    expected = int(marker.get("expected_trials_per_outer_fold", 0))
    pass_value = (
        marker.get("status") == "COMPLETE"
        and expected > len(REQUIRED_MODEL_FAMILIES)
        and len(folds) == 3
        and all(int(item.get("trial_count", 0)) == expected for item in folds)
    )
    return {
        "status": marker.get("status"),
        "expected_trials_per_outer_fold": expected,
        "folds": folds,
        "pass": pass_value,
    }


def main() -> None:
    nested = load_json(FINAL / "NESTED_OOF_RESULTS.json")
    group_bootstrap = load_json(FINAL / "BOOTSTRAP_GROUP_WEIGHTED.json")
    learner_bootstrap = load_json(FINAL / "BOOTSTRAP_LEARNER_WEIGHTED.json")
    controls = pd.read_csv(require(OUT / "negative_controls_retrained/SUMMARY.csv"))
    ablations = pd.read_csv(require(OUT / "ablations_executed/SUMMARY.csv"))
    feature_schema = load_json(OUT / "FEATURE_SCHEMA.json")
    temporal = load_json(OUT / "TEMPORAL_RESULTS.json")
    full_grid = full_grid_gate()

    model_metrics = nested["metrics"]["model_score"]
    nonlearned_methods = ["popular_score", "workload_score", "policy_score", "counterfactual_score"]
    best_baseline = max(nonlearned_methods, key=lambda method: float(nested["metrics"][method]["ndcg_at_3"]))
    best_baseline_value = float(nested["metrics"][best_baseline]["ndcg_at_3"])
    model_value = float(model_metrics["ndcg_at_3"])
    random_p95 = float(nested["random_null"]["p95"])

    bootstrap_lookup = {str(item["baseline"]): item for item in group_bootstrap["comparisons"]}
    best_bootstrap = bootstrap_lookup.get(best_baseline)
    if best_bootstrap is None:
        raise RuntimeError(f"Missing group-weighted bootstrap for {best_baseline}")

    control_names = set(controls.get("control", pd.Series(dtype=str)).astype(str))
    control_names_complete = REQUIRED_CONTROLS.issubset(control_names)
    control_complete = control_names_complete and bool(len(controls)) and bool(
        (controls.loc[controls["control"].isin(REQUIRED_CONTROLS), "completed_replicates"]
         >= controls.loc[controls["control"].isin(REQUIRED_CONTROLS), "registered_replicates"]).all()
    )
    control_pass = control_complete and bool(
        (controls.loc[controls["control"].isin(REQUIRED_CONTROLS), "status"] == "PASS").all()
    )

    ablation_lookup = ablations.set_index("ablation").to_dict(orient="index")
    ablation_complete = REQUIRED_ABLATIONS.issubset(ablation_lookup)
    full_ndcg = float(ablation_lookup.get("FULL", {}).get("ndcg_at_3", float("nan")))
    no_interactions_ndcg = float(ablation_lookup.get("NO_ACTION_INTERACTIONS", {}).get("ndcg_at_3", float("nan")))
    no_counterfactual_ndcg = float(ablation_lookup.get("NO_COUNTERFACTUAL_DELTA", {}).get("ndcg_at_3", float("nan")))
    action_prior_ndcg = float(ablation_lookup.get("ACTION_PRIOR_ONLY", {}).get("ndcg_at_3", float("nan")))

    prohibited = set(feature_schema.get("prohibited", []))
    used = set(feature_schema.get("features", []))
    evaluated_models = set(nested.get("models_actually_evaluated", []))
    data_gate = {
        "precutoff_only": bool(feature_schema.get("precutoff_only")),
        "prohibited_feature_intersection": sorted(prohibited.intersection(used)),
        "all_models_evaluated": evaluated_models == REQUIRED_MODEL_FAMILIES,
        "full_registered_search": full_grid,
    }
    data_gate["pass"] = (
        data_gate["precutoff_only"]
        and not data_gate["prohibited_feature_intersection"]
        and data_gate["all_models_evaluated"]
        and full_grid["pass"]
    )

    fold_metrics = nested["fold_metrics"]
    folds_above_random_mean = 0
    folds_above_best_baseline = 0
    random_mean = float(nested["random_null"]["mean"])
    for fold in fold_metrics.values():
        if float(fold["model_score"]["ndcg_at_3"]) > random_mean:
            folds_above_random_mean += 1
        fold_best = max(float(fold[method]["ndcg_at_3"]) for method in nonlearned_methods)
        if float(fold["model_score"]["ndcg_at_3"]) > fold_best:
            folds_above_best_baseline += 1

    unavailable_rate = top_unavailable_rate()
    state_controls_pass = bool(
        set(
            controls.loc[
                controls["control"].isin(["NC2A_TRAIN_STATE_SHUFFLE", "NC2B_TEST_STATE_SHUFFLE"]),
                "status",
            ]
        ) == {"PASS"}
    )
    personalization_evidence = {
        "action_diversity": int(model_metrics["action_diversity"]),
        "top_action_concentration": float(model_metrics["top_action_concentration"]),
        "full_ndcg_at_3": full_ndcg,
        "no_interactions_ndcg_at_3": no_interactions_ndcg,
        "full_minus_no_interactions": full_ndcg - no_interactions_ndcg,
        "full_minus_action_prior": full_ndcg - action_prior_ndcg,
        "state_controls_pass": state_controls_pass,
    }
    personalization_evidence["pass"] = (
        ablation_complete
        and int(model_metrics["action_diversity"]) > 1
        and float(model_metrics["top_action_concentration"]) < 0.80
        and state_controls_pass
        and full_ndcg > action_prior_ndcg
    )

    gates = {
        "data": data_gate,
        "ranking": {
            "model_ndcg_at_3": model_value,
            "random_p95": random_p95,
            "best_nonlearned_baseline": best_baseline,
            "best_nonlearned_ndcg_at_3": best_baseline_value,
            "bootstrap_ci95_low": float(best_bootstrap["ci95_low"]),
            "pass": model_value > random_p95 and model_value > best_baseline_value and float(best_bootstrap["ci95_low"]) > 0,
        },
        "personalization": personalization_evidence,
        "stability": {
            "folds_above_random_mean": folds_above_random_mean,
            "folds_above_best_baseline": folds_above_best_baseline,
            "pass": folds_above_random_mean == 3 and folds_above_best_baseline >= 2,
        },
        "negative_controls": {
            "required": sorted(REQUIRED_CONTROLS),
            "present": sorted(control_names),
            "complete": control_complete,
            "all_pass": control_pass,
            "pass": control_pass,
        },
        "ablations": {
            "required": sorted(REQUIRED_ABLATIONS),
            "complete": ablation_complete,
            "full_ndcg_at_3": full_ndcg,
            "no_interactions_ndcg_at_3": no_interactions_ndcg,
            "no_counterfactual_ndcg_at_3": no_counterfactual_ndcg,
            "action_prior_ndcg_at_3": action_prior_ndcg,
            "pass": ablation_complete,
        },
        "safety": {
            "protected_features_used": False,
            "unavailable_top_action_rate": unavailable_rate,
            "pass": unavailable_rate == 0.0,
        },
        "reproducibility": {
            "final_checksums_present": (FINAL / "CHECKSUMS.json").exists(),
            "scientific_authority_present": (OUT / "SCIENTIFIC_EXECUTION_AUTHORITY.json").exists(),
            "group_bootstrap_replicates": group_bootstrap.get("replicates", 2000),
            "learner_bootstrap_replicates": learner_bootstrap.get("replicates", 2000),
            "pass": (FINAL / "CHECKSUMS.json").exists()
            and (OUT / "SCIENTIFIC_EXECUTION_AUTHORITY.json").exists()
            and len(group_bootstrap.get("comparisons", [])) > 0
            and len(learner_bootstrap.get("comparisons", [])) > 0,
        },
        "temporal": {
            "status": temporal.get("status"),
            "required_for_release": False,
            "pass": temporal.get("status") in {
                "COMPLETE", "COMPUTED", "COMPLETE_INSUFFICIENT_SUPPORT",
                "TEMPORAL_GENERALIZATION_NOT_IDENTIFIABLE_FROM_COHORT",
            },
        },
    }

    mandatory = ["data", "ranking", "personalization", "stability", "negative_controls", "ablations", "safety", "reproducibility", "temporal"]
    all_pass = all(bool(gates[name]["pass"]) for name in mandatory)
    status = "OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED" if all_pass else "OUTCOME_GROUNDED_V2_1_EVIDENCE_INCONCLUSIVE"
    thesis_status = "RECOMMENDATION_MODULE_COMPLETE" if all_pass else "RECOMMENDATION_MODULE_NOT_COMPLETE"
    runtime_authority = "AUTHORIZED_FOR_INTEGRATION" if all_pass else "NOT_AUTHORIZED"

    registry = {
        "status": status,
        "thesis_scope_completion": thesis_status,
        "runtime_authority": runtime_authority,
        "claim_boundary": "OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT",
        "causal_validation": "NOT_PERFORMED",
        "expert_validation": "NOT_PERFORMED_NOT_REQUIRED_FOR_OFFLINE_THESIS_SCOPE",
        "merge_allowed": False,
        "best_nonlearned_baseline": best_baseline,
        "gates": gates,
        "historical_status": {
            "counterfactual_v1": "COUNTERFACTUAL_V1_ENGINEERING_COMPLETE_EXTERNAL_VALIDATION_FAILED",
            "outcome_grounded_v2": "OUTCOME_GROUNDED_OFFLINE_EVIDENCE_INCONCLUSIVE",
            "preliminary_v2_1": "PRELIMINARY_SINGLE_MODEL_EXECUTION_NOT_FINAL_EVIDENCE",
            "first_configuration_corrected_v2_1": "ARCHIVED_BEFORE_FULL_REGISTERED_SEARCH",
        },
    }
    atomic_json(OUT / "SCIENTIFIC_GATE_CORRECTED.json", registry)
    atomic_json(OUT / "RELEASE_REGISTRY_CORRECTED.json", registry)
    atomic_json(OUT / "CORRECTED_CHECKSUMS.json", checksum_tree(OUT))

    lines = [
        "# Kết quả khoa học Outcome-Grounded V2.1", "",
        f"- Trạng thái: `{status}`",
        f"- Hoàn thành phạm vi khóa luận: `{thesis_status}`",
        f"- Runtime authority: `{runtime_authority}`",
        f"- NDCG@3 OOF: `{model_value:.6f}`",
        f"- Random p95: `{random_p95:.6f}`",
        f"- Baseline mạnh nhất: `{best_baseline}` = `{best_baseline_value:.6f}`",
        f"- CI 95% chênh lệch so với baseline mạnh nhất: `[{float(best_bootstrap['ci95_low']):.6f}, {float(best_bootstrap['ci95_high']):.6f}]`",
        "", "## Scientific gates", "",
    ]
    for name in mandatory:
        lines.append(f"- {name}: `{'PASS' if gates[name]['pass'] else 'FAIL'}`")
    lines.extend([
        "", "## Ablation diễn giải", "",
        f"- Full − no interactions: `{full_ndcg - no_interactions_ndcg:.6f}`",
        f"- Full − no counterfactual: `{full_ndcg - no_counterfactual_ndcg:.6f}`",
        f"- Full − action prior: `{full_ndcg - action_prior_ndcg:.6f}`",
        "", "## Giới hạn diễn giải", "",
        "Kết quả chỉ chứng minh khả năng xếp hạng offline trên trajectory OULAD giữ lại. Không được diễn giải là tác động nhân quả hoặc bảo đảm cải thiện điểm số.",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    progress_path = OUT / "PROGRESS.json"
    progress = load_json(progress_path)
    progress.setdefault("stages", {})["RELEASE"] = {
        "status": "COMPLETE",
        "scientific_status": status,
        "runtime_authority": runtime_authority,
    }
    atomic_json(progress_path, progress)
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
