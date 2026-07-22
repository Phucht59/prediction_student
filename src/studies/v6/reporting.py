from __future__ import annotations

import json
import subprocess
from typing import Any

from .contract import ARTIFACT_ROOT, REPORT_ROOT, ROOT, atomic_json, atomic_text


def _read(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def generate_final_report() -> dict[str, Any]:
    audit = _read("artifacts/v6/audit/knowledge_audit.json")
    reproduction = _read("artifacts/v6/prediction/v5_1_reproduction.json")
    final = _read("artifacts/v6/prediction/final/run_state.json")
    calibration = _read("artifacts/v6/prediction/calibration.json")
    domain = _read("artifacts/v6/prediction/domain_generalization/run_state.json")
    profiles = _read("artifacts/v6/prediction/risk_profile_state.json")
    recommendation = _read("artifacts/v6/recommendation/run_state.json")
    expert = _read("artifacts/v6/recommendation/expert_evaluation/metrics.json")
    linkage = _read("artifacts/v6/linkage/analysis.json")
    database = _read("artifacts/v6/database/audit.json")
    validation = _read("artifacts/v6/validation_report.json")
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    verdict = (
        "INTEGRATED_SYSTEM_PASS"
        if validation["status"] == "PASS" and expert["status"] == "COMPLETE"
        else "INTEGRATED_SYSTEM_PASS_EXPERT_PENDING"
        if validation["status"] == "PASS"
        and recommendation["status"] == "PASS"
        and expert["status"] == "PENDING_EXPERT_LABELS"
        else "PREDICTION_PASS_RECOMMENDATION_FAIL"
        if final["status"] == "COMPLETE" and recommendation["status"] != "PASS"
        else "INTEGRATED_SYSTEM_NOT_READY"
    )
    metrics = final["ensemble_metrics"]
    report = f"""# Final V6 integrated system review

System verdict: **{verdict}**

## Git and immutable bases

- Branch: `{branch}`
- Repository integration base: `24cca2b7f0904504e6f1c937af04589938e1a73f`
- Scientific V5.1 source: `308370cf6c6f16e65cc0f0aaa3f38393ae141e16`
- Recommendation V5.2 source: `b9087ceb1600582ad1351b134a2f4c4d9af77d89`
- Report-generation HEAD: `{head}`
- Future OULAD: `LOCKED_NOT_EXECUTED`

## Knowledge audit

- Order destruction: `{audit['order_destruction']['verdict']}`
- Residual signal: `{audit['residual_ceiling']['verdict']}`
- Oracle complementarity gain: {audit['oracle_complementarity']['oracle_gain_over_best']:.6f}
- Survival feasibility: `{audit['survival']['verdict']}`
- Graph feasibility: `{audit['graph']['verdict']}`; graph was skipped because Candidate D failed its guardrail.

## Prediction evidence

- V5.1 reproduction: **{reproduction['status']}**
- Selected: `C_TEMPORAL_MULTITASK_W0` (P1 pretraining + withdrawal/outcome heads)
- Parameters: {final['selected_model']['parameter_count']:,}
- Outer matrix: 3 folds x 5 fixed seeds ({final['checkpoint_count']} checkpoints)
- Macro-F1: {metrics['macro_f1']:.6f}
- At-risk F1: {metrics['at_risk_f1']:.6f}
- PR-AUC: {metrics['pr_auc']:.6f}
- Brier: {metrics['brier']:.6f}
- ECE before outer reporting calibration: {metrics['ece']:.6f}
- Recall@10%: {metrics['recall_at_10_percent']:.6f}
- Survival C-index: {metrics['survival_concordance']:.6f}
- Withdrawal recall: {metrics['withdrawal_recall']:.6f}
- Outcome Macro-F1: {metrics['outcome_macro_f1']:.6f}
- Total recorded training runtime: {final['runtime_seconds']:.1f} seconds
- Peak CUDA allocation: {final['max_gpu_memory_bytes'] / 1024**2:.1f} MiB
- Calibration temperature: {calibration['temperature']:.6f}; fit on inner OOF only
- Domain conclusion: `{domain['conclusion']}`

CNN-BiLSTM is retained as the temporal thesis model; XGBoost is retained as an
operational cross-check. The integration value lies in progression modelling,
risk prioritization, calibrated uncertainty and governed recommendations.

## Risk profiles

- Schema: `{profiles['profile_schema']}`
- Records / coverage: {profiles['records']} / {profiles['coverage']:.3%}
- Abstention: {profiles['abstention_rate']:.3%}
- Confidence: `{profiles['confidence_distribution']}`
- Top-k distribution: `{profiles['top_k_distribution']}`
- Mean deep-ML disagreement: {profiles['mean_deep_ml_disagreement']:.6f}
- Sensitive demographics in payload: {profiles['sensitive_demographics_in_payload']}

## Recommendation and governance

- Plans / coverage: {recommendation['plans_generated']} / {recommendation['coverage']:.3%}
- Escalation: {recommendation['escalation_rate']:.3%}
- Conflicts / duplicates / workload violations: {recommendation['conflicts']} / {recommendation['duplicate_plans']} / {recommendation['workload_violations']}
- Missing lineage: {recommendation['missing_lineage']}
- Deterministic replay: {recommendation['deterministic_replay']}
- Linkage/stability: **{linkage['status']}**
- Database: `{database['status']}`; production write = {database['production_write']}

## Expert evidence and claims

- Status: **{expert['status']}**
- Experts / cases scored: {expert['experts']} / {expert['cases_scored']}
- Action F1 / Top-3 recall / approval / escalation F1 / agreement: pending real labels
- A blinded 60-case, two-expert package exists; no labels were fabricated.
- Recommendation technical validation is not evidence of causal student-outcome improvement.

## Validation

- V6 validation: **{validation['status']}** ({validation['passed']}/{validation['check_count']})
- Protected V4-V5.4 hashes: **{'PASS' if validation['checks']['protected_hashes'] else 'FAIL'}**
- Future lock: **{'PASS' if validation['checks']['future_locked'] else 'FAIL'}**
"""
    atomic_text(REPORT_ROOT / "final/FINAL_V6_INTEGRATED_SYSTEM_REVIEW.md", report)
    result = {
        "schema_version": "v6_final_review_v1",
        "status": "COMPLETE",
        "system_verdict": verdict,
        "branch": branch,
        "report_generation_head": head,
        "expert_status": expert["status"],
    }
    atomic_json(ARTIFACT_ROOT / "final_review.json", result)
    return result


__all__ = ["generate_final_report"]
