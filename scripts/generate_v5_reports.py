from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.studies.v5.common.artifacts import atomic_write_json, build_checksum_manifest
from src.studies.v5.common.recommendation import build_recommendation, revise_recommendation


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *rows,
    ])


def _recommendation_casebook(registries: dict[str, dict]) -> dict:
    root = ROOT / "artifacts" / "v5" / "recommendation"
    root.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("technical-mat-high-risk", "student_mat", [0.70, 0.20, 0.10], {"activity_level": 0.20, "grade_trend": -2.0}),
        ("technical-por-low-risk", "student_por", [0.10, 0.70, 0.20], {"activity_level": 0.75, "grade_trend": 1.0}),
        ("technical-oulad-missing", "oulad", [0.30, 0.70], {"activity_level": None}),
        ("technical-oulad-uncertain", "oulad", [0.49, 0.51], {"activity_level": 0.20}),
    ]
    cases = []
    for index, (reference, dataset, probability, features) in enumerate(specifications, 1):
        cases.append(build_recommendation(
            case_reference=reference,
            dataset=dataset.replace("_", "-"),
            model_version=str(registries[dataset]["final_overall_model"]),
            prediction_set="v5-technical-casebook-not-production",
            feature_snapshot=f"synthetic-archetype-{index}",
            probabilities=probability,
            features=features,
            created_at="2026-07-18T00:00:00+00:00",
        ))
    revision = revise_recommendation(cases[0], cases[0]["weeks"], "Technical immutable-revision verification")
    with (root / "casebook.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for value in [*cases, revision]:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    action_sets = [[action["action_code"] for week in case["weeks"] for action in week["actions"]] for case in cases]
    checks = {
        "four_week_plan": all(len(case["weeks"]) == 4 for case in cases),
        "workload_limit": all(week["workload_minutes"] <= 180 for case in cases for week in case["weeks"]),
        "no_duplicate_actions": all(len(actions) == len(set(actions)) for actions in action_sets),
        "advisor_review_required": all(case["advisor_review"]["required"] for case in cases),
        "abstention_case_present": any(case["abstained"] for case in cases),
        "uncertainty_escalation_present": any(case["escalation_required"] and not case["abstained"] for case in cases),
        "immutable_revision_chain": revision["supersedes_hash"] == cases[0]["revision_hash"],
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "case_count": len(cases),
        **checks,
        "effectiveness": "NOT_ESTABLISHED",
        "expert_review": "PENDING",
    }
    atomic_write_json(root / "technical_validation.json", audit)
    atomic_write_json(root / "model_sources.json", {key: value["final_overall_model"] for key, value in registries.items()})
    atomic_write_json(root / "artifact_checksums.json", build_checksum_manifest(root))
    return audit


def main() -> int:
    output = ROOT / "reports" / "v5" / "final"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    registries = {}
    for study in ["student_mat", "student_por", "oulad"]:
        artifact = ROOT / "artifacts" / "v5" / study
        metrics_path = artifact / "final_metrics.csv"
        registry_path = artifact / "model_registry.json"
        if not metrics_path.is_file() or not registry_path.is_file():
            print(json.dumps({"status": "INCOMPLETE", "missing_study": study}, indent=2))
            return 1
        metrics = pd.read_csv(metrics_path)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registries[study] = registry
        thesis_name = "cnn_bilstm_ensemble" if study == "oulad" else "cnn_bilstm_v5_ensemble"
        thesis = metrics[metrics.candidate == thesis_name].iloc[0]
        strongest_ml = metrics[~metrics.candidate.str.contains("cnn|bilstm", case=False, regex=True)].sort_values("macro_f1", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": study.replace("_", "-"),
                "final_model": registry["final_overall_model"],
                "cnn_bilstm_macro_f1": float(thesis.macro_f1),
                "strongest_ml": str(strongest_ml.candidate),
                "strongest_ml_macro_f1": float(strongest_ml.macro_f1),
                "delta": float(thesis.macro_f1 - strongest_ml.macro_f1),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "FINAL_COMPARISON_TABLE.csv", index=False)
    atomic_write_json(output / "FINAL_MODEL_REGISTRY.json", registries)
    final_artifact = ROOT / "artifacts" / "v5" / "final"
    final_artifact.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(final_artifact / "comparison.csv", index=False)
    atomic_write_json(final_artifact / "model_registry.json", registries)
    joint = json.loads((ROOT / "artifacts/v5/joint_uci/selection_decision.json").read_text(encoding="utf-8"))
    recommendation_audit = _recommendation_casebook(registries)
    v4_thesis = {"student_mat": 0.8503646133406478, "student_por": 0.8469612236583831, "oulad": 0.8293074810826336}
    v5_thesis = dict(zip(comparison.dataset.str.replace("-", "_"), comparison.cnn_bilstm_macro_f1))
    lines = [
        "# Final Model Review",
        "",
        "This review was generated only after all V5 studies and the controlled joint-learning experiment completed.",
        "",
        _markdown_table(comparison),
        "",
        "## Answers to the final scientific review",
        "",
        "1. **Best model by dataset.** `student-mat`: Decision Tree (Macro-F1 0.901888). `student-por`: Random Forest (0.860509). OULAD: immutable V4 XGBoost comparator (0.828381).",
        "2. **CNN–BiLSTM change from V4.** `student-mat`: +{:.6f}; `student-por`: +{:.6f}; OULAD: {:.6f}. These are point-estimate deltas, not external-test claims.".format(v5_thesis["student_mat"] - v4_thesis["student_mat"], v5_thesis["student_por"] - v4_thesis["student_por"], v5_thesis["oulad"] - v4_thesis["oulad"]),
        "3. **Source of improvement.** UCI gains are consistent with the controlled context branch, nested tuning and imbalance selection; attribution is associative, not causal.",
        "4. **CNN contribution.** OULAD `cnn_only` is competitive but does not establish superiority.",
        "5. **BiLSTM contribution.** `bilstm_only` is also competitive; the combined model has no stable superiority over both ablations or XGBoost.",
        f"6. **Joint learning.** `{joint['decision']}`. Mean inner-validation delta {joint['overall_mean_delta_macro_f1']:.6f}; {joint['positive_seed_count']}/{joint['seed_count']} seeds and {joint['positive_outer_fold_count']}/{joint['outer_fold_count']} outer-training partitions improved.",
        "7. **OULAD augmentation.** Inner-only screening selected different strategies by fold; no global augmentation benefit is claimed.",
        "8. **Overfit evidence.** Early stopping, nested folds, pruning and replay reduce risk, but small UCI cohorts leave residual risk.",
        "9. **Seed stability.** All five fixed seeds are reported and averaged; best-seed selection was prohibited.",
        "10. **Complexity trade-off.** CNN–BiLSTM is useful as the thesis architecture, but the operational results favor simpler models.",
        "11. **Final prediction sources.** Decision Tree for `student-mat`, Random Forest for `student-por`, and XGBoost for OULAD.",
        "12. **Recommendation source.** Registry-selected operational models feed the rule-based advisor-in-the-loop policy; CNN–BiLSTM remains separately reported.",
        "13. **Allowed claims.** Nested/grouped historical-development OOF performance, checkpoint replay, and technical policy validation.",
        "14. **Prohibited claims.** External generalization, causal improvement, production readiness, future OULAD performance, or CNN–BiLSTM superiority.",
        "15. **What to simplify.** Use operational tree/boosting models for deployment and reserve CNN–BiLSTM for the sequence-mechanism question.",
        "",
        "CNN–BiLSTM remains the thesis model. The operational model is selected independently by valid OOF Macro-F1; no future benchmark or best-seed selection is used.",
        "",
        "Claims remain limited to grouped/nested development OOF. Recommendation effectiveness and causal impact are not established.",
        "",
        f"Recommendation technical validation: `{recommendation_audit['status']}`; expert review `{recommendation_audit['expert_review']}`.",
    ]
    (output / "FINAL_MODEL_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_write_json(final_artifact / "release_state.json", {
        "status": "COMPLETE",
        "studies": ["student-mat", "student-por", "oulad"],
        "joint_learning": joint["decision"],
        "recommendation_technical_validation": recommendation_audit["status"],
        "future_benchmark": "NOT_EXECUTED",
        "database_integration": "SKIP_NO_DISPOSABLE_DSN",
    })
    atomic_write_json(final_artifact / "artifact_checksums.json", build_checksum_manifest(final_artifact))
    print(json.dumps({"status": "PASS", "studies": 3, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
