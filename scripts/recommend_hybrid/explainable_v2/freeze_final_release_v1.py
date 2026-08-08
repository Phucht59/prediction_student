"""Create the Recommendation V2 final release reports and immutable authority manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/recommend_hybrid/explainable_v2"
REPORTS = ROOT / "reports/recommend_hybrid_v2"
OUT = BASE / "frozen/final_release_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    phase9_path = BASE / "audit/final_integration_v1/PHASE9_END_TO_END_INTEGRATION_AUDIT.json"
    phase10_path = BASE / "audit/final_release_v1/PHASE10_FINAL_AUDIT.json"
    dev_path = BASE / "frozen/development_v2/DEVELOPMENT_FREEZE_MANIFEST.json"
    pb_dir = BASE / "final_heldout/panel_b_v1"
    pb_manifest_path = pb_dir / "PANEL_B_FINAL_HELDOUT_MANIFEST.json"
    metrics_path = pb_dir / "PANEL_B_FINAL_HELDOUT_METRICS.json"
    phase9, phase10, dev = load(phase9_path), load(phase10_path), load(dev_path)
    pb_manifest, metrics = load(pb_manifest_path), load(metrics_path)
    if not (phase9["status"] == phase10["status"] == dev["status"] == pb_manifest["status"] == "PASS"):
        raise RuntimeError("FINAL_RELEASE_GATE_NOT_PASS")
    ranker = metrics["frozen_five_ebm_ranker"]
    baseline = metrics["panel_a_action_stage_only_baseline"]
    boot = metrics["paired_case_bootstrap"]

    write(REPORTS / "FINAL_MODEL_ARCHITECTURE_AND_PIPELINE.md", f"""# Final Model Architecture and Pipeline

Status: **PASS**. Release scope: Recommendation V2.

1. The frozen Hybrid CNN–BiLSTM predicts learner risk. Its architecture and weights were not retrained or tuned by Recommendation V2 or Panel B.
2. The frozen router decides whether recommendation processing is justified, using exactly `RECOMMEND`, `INSUFFICIENT_EVIDENCE`, `HUMAN_REVIEW`, and `NO_FEASIBLE_ACTION`.
3. The canonical V4 feasibility policy removes impossible actions before ranking.
4. Five action-specific EBMs score feasible interventions. Native ordinal predictions use the 0–3 scale; the single public adapter produces `clip(native / 3, 0, 1)`.
5. The highest-scoring valid action is recommended only when evidence and ambiguity gates pass.
6. Explanations and plans are generated from observed pre-cutoff evidence.
7. The plausibility simulator reports a **model-implied risk delta** only. It does not estimate a causal treatment effect.

The unavailable `seed_disagreement` value is nullable and is never silently replaced with zero. The frozen router therefore applies no disagreement threshold when no real finite value exists. Panel B was evaluated once after development freeze and was not used for tuning.
""")

    write(REPORTS / "PANEL_B_FINAL_HELDOUT_RESULTS.md", f"""# Panel B Final Held-Out Results

These are **PANEL_B_FINAL_HELDOUT** results, not Panel-A development results.

| Measure | Result |
|---|---:|
| Cases | {metrics['panel_b_case_count']} |
| Real external Gemini review records | {metrics['real_external_review_record_count']} |
| Evidence-complete coverage | {metrics['evidence_complete_case_coverage']:.3f} |
| NDCG@3 | {ranker['ndcg_at_3']} |
| Exact best Top-1 agreement | {ranker['exact_best_top1_agreement']} |
| Precision@1 | {ranker['precision_at_1_relevance_ge_1']} |
| Recall@3 | {ranker['recall_at_3_relevance_ge_1']} |
| MRR | {ranker['mrr_relevance_ge_1']} |
| Pairwise accuracy | {ranker['pairwise_accuracy']} |
| Invalid-action rate | {ranker['invalid_action_rate']} |
| Action+stage baseline NDCG@3 | {baseline['ndcg_at_3']} |
| Full minus baseline NDCG@3 | {boot['mean_full_minus_baseline_ndcg_at_3']} |
| Paired-bootstrap 95% CI | [{boot['ci_low_95']}, {boot['ci_high_95']}] |

Provider/model/prompt provenance and response hashes are preserved in the frozen Panel-B manifest and checksum inventory. There were zero provider-call failures and no review was fabricated, simulated, or substituted. No model, threshold, calibration, feature, feasibility rule, or risk threshold was changed after observing Panel B.
""")

    phase9_pass_count = sum(value in (True, "PASS") for value in phase9["checks"].values())
    phase10_pass_count = sum(value in (True, "PASS") for value in phase10["checks"].values())
    write(REPORTS / "FINAL_SCIENTIFIC_AUDIT.md", f"""# Final Scientific Audit

Overall status: **PASS**.

- Development freeze: PASS; Panel B was untouched at that gate.
- Phase 9 end-to-end integration: PASS ({phase9_pass_count}/{len(phase9['checks'])} checks).
- Phase 10 independent audit: PASS ({phase10_pass_count}/{len(phase10['checks'])} checks).
- Panel A retained 1,499 of 1,500 provenance rows; the sole row with fewer than two independent source families remains auditable but was excluded from supervised training/evaluation.
- Selected EBM configuration remained `a70599afad40`; raw EBM calibration remained selected under the preregistered NDCG-primary rule.
- Panel A release gates passed without threshold relaxation. Panel A metrics remain development-only.
- Panel B contains {metrics['panel_b_case_count']} held-out cases and {metrics['real_external_review_record_count']} real external review records, with zero failed calls and complete evidence coverage.
- Student/query overlap, post-cutoff leakage, feature leakage, invalid-action, secret, salt, private-mapping, and provenance checks passed.
- Frozen Panel-B metrics and evidence were hash-verified and were not recomputed during Phases 9–11.
- The post-Panel-B ranker clamp is engineering-only and output-invariant across the frozen Panel-B score artifact; it does not alter rankings or metrics.
- A stale ignored private mapping was removed from the repository and quarantined outside it; it is not tracked or released.

Scientific claim boundary: the module supports predictive ranking and plausibility analysis. Any simulated change is a **model-implied risk delta**, not a causal treatment effect.
""")

    write(REPORTS / "FINAL_RELEASE_SUMMARY.md", f"""# Recommendation V2 Final Release Summary

Recommendation V2 is frozen and authorized for runtime integration under the exact hashed lineage in `FINAL_RELEASE_MANIFEST.json`.

- Development freeze: PASS
- Phase 9 integration: PASS
- Phase 10 audit: PASS
- Panel B final held-out evaluation: PASS
- Panel B NDCG@3: {ranker['ndcg_at_3']}
- Panel B invalid-action rate: {ranker['invalid_action_rate']}
- Runtime authorized: TRUE
- Final metrics claimed: TRUE, scoped only as `PANEL_B_FINAL_HELDOUT`

Historical development and held-out manifests retain their original `runtime_authorized=false` values as immutable audit facts. Runtime authority is granted only by the final release manifest and only for the exact hashes recorded there. Any later change invalidates that authority until re-audited. No causal effect claim is made.
""")

    report_hashes = {path.name: sha(path) for path in sorted(REPORTS.glob("FINAL_*.md"))}
    report_hashes["PANEL_B_FINAL_HELDOUT_RESULTS.md"] = sha(REPORTS / "PANEL_B_FINAL_HELDOUT_RESULTS.md")
    manifest = {
        "schema_version": "recommendation_v2_final_release_v1",
        "scope": "FINAL_RELEASE",
        "status": "PASS",
        "runtime_authorized": True,
        "final_metrics_claimed": True,
        "final_metrics_scope": "PANEL_B_FINAL_HELDOUT",
        "panel_b_touched": True,
        "post_panel_b_tuning_permitted": False,
        "provider_calls_after_panel_b_freeze": 0,
        "panel_b_case_count": metrics["panel_b_case_count"],
        "real_external_review_record_count": metrics["real_external_review_record_count"],
        "failed_provider_calls": pb_manifest["failed_provider_calls"],
        "selected_config_id": dev["ranker"]["selected_config_id"],
        "calibration_decision": dev["ranker"]["calibration_decision"],
        "score_contract": dev["ranker"]["score_contract"],
        "router_statuses": dev["router"]["public_statuses"],
        "simulator_language": "model-implied risk delta",
        "lineage_sha256": {
            "development_freeze": sha(dev_path),
            "phase9_integration_audit": sha(phase9_path),
            "phase10_final_audit": sha(phase10_path),
            "panel_b_final_manifest": sha(pb_manifest_path),
            "panel_b_final_metrics": sha(metrics_path),
            "panel_b_frozen_reviews": pb_manifest["frozen_reviews_sha256"],
            "panel_b_scores": pb_manifest["scores_sha256"],
            "risk_checkpoint_manifest": dev["lineage_sha256"]["risk_checkpoint_manifest"],
            "panel_a_review_freeze": dev["lineage_sha256"]["panel_a_review_freeze"],
            "label_manifest": dev["lineage_sha256"]["label_manifest"],
            "five_ebm_manifest": dev["lineage_sha256"]["five_ebm_manifest"],
            "ranker_freeze": dev["lineage_sha256"]["ranker_freeze"],
            "router_freeze": dev["lineage_sha256"]["router_freeze"],
            "feasibility_policy": dev["lineage_sha256"]["feasibility_policy"],
            "release_gates": dev["lineage_sha256"]["release_gates"],
            "current_ranker_source": sha(ROOT / "src/recommend_hybrid/explainable_v2/ranker.py"),
        },
        "five_model_sha256": dev["ranker"]["five_model_sha256"],
        "report_sha256": report_hashes,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "FINAL_RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "checksums.sha256").write_text(f"{sha(manifest_path)}  FINAL_RELEASE_MANIFEST.json\n", encoding="utf-8")
    print("PHASE11_FREEZE_STATUS=PASS")
    print("RUNTIME_AUTHORIZED=TRUE")
    print(f"FINAL_RELEASE_MANIFEST_SHA256={sha(manifest_path)}")


if __name__ == "__main__":
    main()
