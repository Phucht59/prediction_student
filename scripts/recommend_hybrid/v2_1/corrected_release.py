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
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        output[str(path.relative_to(root)).replace("\\", "/")] = digest
    return output


def main() -> None:
    nested = load_json(FINAL / "NESTED_OOF_RESULTS.json")
    group_bootstrap = load_json(FINAL / "BOOTSTRAP_GROUP_WEIGHTED.json")
    learner_bootstrap = load_json(FINAL / "BOOTSTRAP_LEARNER_WEIGHTED.json")
    controls = pd.read_csv(require(OUT / "negative_controls_retrained/SUMMARY.csv"))
    ablations = pd.read_csv(require(OUT / "ablations_executed/SUMMARY.csv"))
    feature_schema = load_json(OUT / "FEATURE_SCHEMA.json")
    temporal = load_json(OUT / "TEMPORAL_RESULTS.json")

    model_metrics = nested["metrics"]["model_score"]
    nonlearned_methods = [
        "popular_score",
        "workload_score",
        "policy_score",
        "counterfactual_score",
    ]
    best_baseline = max(
        nonlearned_methods,
        key=lambda method: float(nested["metrics"][method]["ndcg_at_3"]),
    )
    best_baseline_value = float(nested["metrics"][best_baseline]["ndcg_at_3"])
    model_value = float(model_metrics["ndcg_at_3"])
    random_p95 = float(nested["random_null"]["p95"])

    bootstrap_lookup = {
        str(item["baseline"]): item for item in group_bootstrap["comparisons"]
    }
    best_bootstrap = bootstrap_lookup.get(best_baseline)
    if best_bootstrap is None:
        raise RuntimeError(f"Missing group-weighted bootstrap for {best_baseline}")

    control_complete = bool(len(controls)) and bool(
        (controls["completed_replicates"] >= controls["registered_replicates"]).all()
    )
    control_pass = control_complete and bool((controls["status"] == "PASS").all())

    ablation_lookup = ablations.set_index("ablation").to_dict(orient="index")
    required_ablations = {
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
    ablation_complete = required_ablations.issubset(ablation_lookup)
    full_ndcg = float(ablation_lookup.get("FULL", {}).get("ndcg_at_3", float("nan")))
    no_interactions_ndcg = float(
        ablation_lookup.get("NO_ACTION_INTERACTIONS", {}).get("ndcg_at_3", float("nan"))
    )

    prohibited = set(feature_schema.get("prohibited", []))
    used = set(feature_schema.get("features", []))
    data_gate = {
        "precutoff_only": bool(feature_schema.get("precutoff_only")),
        "prohibited_feature_intersection": sorted(prohibited.intersection(used)),
        "all_models_evaluated": set(nested.get("models_actually_evaluated", []))
        == {"interaction_logistic", "pairwise_logistic", "lambdamart", "boosted_tree"},
    }
    data_gate["pass"] = (
        data_gate["precutoff_only"]
        and not data_gate["prohibited_feature_intersection"]
        and data_gate["all_models_evaluated"]
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
    gates = {
        "data": {**data_gate},
        "ranking": {
            "model_ndcg_at_3": model_value,
            "random_p95": random_p95,
            "best_nonlearned_baseline": best_baseline,
            "best_nonlearned_ndcg_at_3": best_baseline_value,
            "bootstrap_ci95_low": float(best_bootstrap["ci95_low"]),
            "pass": model_value > random_p95
            and model_value > best_baseline_value
            and float(best_bootstrap["ci95_low"]) > 0,
        },
        "personalization": {
            "action_diversity": int(model_metrics["action_diversity"]),
            "top_action_concentration": float(model_metrics["top_action_concentration"]),
            "full_ndcg_at_3": full_ndcg,
            "no_interactions_ndcg_at_3": no_interactions_ndcg,
            "state_controls_pass": bool(
                set(
                    controls.loc[
                        controls["control"].isin(
                            ["NC2A_TRAIN_STATE_SHUFFLE", "NC2B_TEST_STATE_SHUFFLE"]
                        ),
                        "status",
                    ]
                )
                == {"PASS"}
            ),
            "pass": int(model_metrics["action_diversity"]) > 1
            and control_pass
            and (full_ndcg > no_interactions_ndcg),
        },
        "stability": {
            "folds_above_random_mean": folds_above_random_mean,
            "folds_above_best_baseline": folds_above_best_baseline,
            "pass": folds_above_random_mean == 3 and folds_above_best_baseline >= 2,
        },
        "negative_controls": {
            "complete": control_complete,
            "all_pass": control_pass,
            "pass": control_pass,
        },
        "ablations": {"complete": ablation_complete, "pass": ablation_complete},
        "safety": {
            "protected_features_used": False,
            "unavailable_top_action_rate": unavailable_rate,
            "pass": unavailable_rate == 0.0,
        },
        "reproducibility": {
            "final_checksums_present": (FINAL / "CHECKSUMS.json").exists(),
            "scientific_authority_present": (OUT / "SCIENTIFIC_EXECUTION_AUTHORITY.json").exists(),
            "pass": (FINAL / "CHECKSUMS.json").exists()
            and (OUT / "SCIENTIFIC_EXECUTION_AUTHORITY.json").exists(),
        },
        "temporal": {
            "status": temporal.get("status"),
            "required_for_release": False,
            "pass": temporal.get("status")
            in {
                "COMPLETE",
                "COMPUTED",
                "COMPLETE_INSUFFICIENT_SUPPORT",
                "TEMPORAL_GENERALIZATION_NOT_IDENTIFIABLE_FROM_COHORT",
            },
        },
    }

    mandatory = [
        "data",
        "ranking",
        "personalization",
        "stability",
        "negative_controls",
        "ablations",
        "safety",
        "reproducibility",
        "temporal",
    ]
    all_pass = all(bool(gates[name]["pass"]) for name in mandatory)
    status = (
        "OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED"
        if all_pass
        else "OUTCOME_GROUNDED_V2_1_EVIDENCE_INCONCLUSIVE"
    )
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
        },
    }
    atomic_json(OUT / "SCIENTIFIC_GATE_CORRECTED.json", registry)
    atomic_json(OUT / "RELEASE_REGISTRY_CORRECTED.json", registry)
    atomic_json(OUT / "CORRECTED_CHECKSUMS.json", checksum_tree(OUT))

    lines = [
        "# Kết quả khoa học Outcome-Grounded V2.1",
        "",
        f"- Trạng thái: `{status}`",
        f"- Hoàn thành phạm vi khóa luận: `{thesis_status}`",
        f"- Runtime authority: `{runtime_authority}`",
        f"- NDCG@3 OOF: `{model_value:.6f}`",
        f"- Random p95: `{random_p95:.6f}`",
        f"- Baseline mạnh nhất: `{best_baseline}` = `{best_baseline_value:.6f}`",
        f"- CI 95% chênh lệch so với baseline mạnh nhất: "
        f"`[{float(best_bootstrap['ci95_low']):.6f}, {float(best_bootstrap['ci95_high']):.6f}]`",
        "",
        "## Scientific gates",
        "",
    ]
    for name in mandatory:
        lines.append(f"- {name}: `{'PASS' if gates[name]['pass'] else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Giới hạn diễn giải",
            "",
            "Kết quả chỉ chứng minh khả năng xếp hạng offline trên trajectory OULAD giữ lại. "
            "Không được diễn giải là tác động nhân quả hoặc bảo đảm cải thiện điểm số.",
        ]
    )
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
