"""Phase 2/3: refreeze development + complete Panel C protocol. No Gemini."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel_c_common import (
    MODEL_NAME,
    PROMPT_PATH,
    PROMPT_VERSION,
    PROVIDER,
    ROOT,
    V3,
    assert_payload_blinded,
    build_case_payload,
    forbidden_hits,
    load_panel_c_feature_rows,
    prompt_sha256,
    prompt_text,
    sha256_file,
)

AUDIT = V3 / "audit"
FREEZE = V3 / "freeze"
PANEL = V3 / "panel_c"
REPORTS = ROOT / "reports" / "recommend_hybrid" / "v3"


def tracked_paths() -> list[Path]:
    return [
        V3 / "data" / "C0_PREDICTION_PROVENANCE.json",
        V3 / "data" / "FEATURE_MANIFEST.json",
        V3 / "data" / "FEATURE_LINEAGE.csv",
        V3 / "data" / "learner_stage_features.parquet",
        V3 / "labels" / "LABEL_PORTABILITY_SUMMARY.json",
        V3 / "labels" / "WEAK_LABEL_MANIFEST.json",
        V3 / "labels" / "v3_action_rows.parquet",
        V3 / "ranker" / "FIVE_EBM_MANIFEST.json",
        V3 / "ranker" / "BASELINE_RESULTS.csv",
        V3 / "ranker" / "BASELINE_RESULTS_RUNTIME_EQUIVALENT.csv",
        V3 / "ranker" / "B0_ACTION_STAGE_PRIOR.json",
        V3 / "ranker" / "oof_predictions.parquet",
        V3 / "ranker" / "final_models" / "ASSESSMENT_COMPLETION.joblib",
        V3 / "ranker" / "final_models" / "QUIZ_RETRIEVAL_PRACTICE.joblib",
        V3 / "ranker" / "final_models" / "RECOVER_ENGAGEMENT.joblib",
        V3 / "ranker" / "final_models" / "STUDY_REGULARITY.joblib",
        V3 / "ranker" / "final_models" / "TARGETED_CONTENT_REVIEW.joblib",
        V3 / "router" / "ROUTER_CONFIG.json",
        V3 / "challenger" / "SELECTION.json",
        V3 / "audit" / "PRE_PANEL_C_AUDIT.json",
        V3 / "audit" / "INVALID_ACTION_CASES.csv",
        V3 / "audit" / "LABEL_PROVENANCE_METRICS.csv",
        PANEL / "PANEL_C_SAMPLED_CASES.parquet",
        PROMPT_PATH,
        ROOT / "src" / "recommend_hybrid" / "v3" / "metrics.py",
        ROOT / "src" / "recommend_hybrid" / "v3" / "pipeline.py",
        ROOT / "src" / "recommend_hybrid" / "v3" / "feasibility.py",
        ROOT / "src" / "recommend_hybrid" / "v3" / "risk_router.py",
        ROOT / "src" / "recommend_hybrid" / "v3" / "safety_router.py",
        ROOT / "src" / "recommend_hybrid" / "v3" / "plan_builder.py",
        ROOT / "src" / "recommend_hybrid" / "v3" / "ranker.py",
    ]


def main() -> None:
    audit = json.loads((AUDIT / "PRE_PANEL_C_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("PRE_PANEL_C_AUDIT") != "PASS":
        raise SystemExit("REFREEZE blocked: PRE_PANEL_C_AUDIT is not PASS")

    prompt = prompt_text()
    prompt_hash = prompt_sha256()
    prompt_hits = forbidden_hits(prompt)
    if prompt_hits:
        raise SystemExit(f"Panel C prompt contains forbidden tokens: {prompt_hits}")

    rows = load_panel_c_feature_rows()
    labels = pd.read_parquet(V3 / "labels" / "v3_action_rows.parquet")
    portable_students = set(
        labels.loc[labels.portability_status.eq("CONDITIONALLY_PORTABLE"), "student_key"].astype(str)
    )
    students = set(rows.student_key.astype(str))
    overlap = sorted(students & portable_students)
    if overlap:
        raise SystemExit(f"Panel C not disjoint from portable Panel A: {len(overlap)} students")
    if rows.query_id.nunique() != len(rows):
        raise SystemExit("duplicate Panel C cases")
    if not set(rows.stage.astype(str)).issubset({"EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75"}):
        raise SystemExit("non-intervention stage in Panel C sample")

    case_records = []
    n_candidates = 0
    n_empty = 0
    for _, row in rows.iterrows():
        case_id, payload, evaluations = build_case_payload(row)
        assert_payload_blinded(payload, prompt)
        n_cand = len(payload["candidate_actions"])
        n_candidates += n_cand
        n_empty += int(n_cand == 0)
        case_records.append(
            {
                "case_id": case_id,
                "query_id": str(row["query_id"]),
                "student_key": str(row["student_key"]),
                "course_key": str(row["course_key"]),
                "stage": str(row["stage"]),
                "cutoff_day": int(row["cutoff_day"]),
                "n_eligible_actions": n_cand,
                "eligible_actions": [item["action_id"] for item in payload["candidate_actions"]],
                "evaluations": evaluations,
                "payload_sha256": hashlib_sha(payload),
            }
        )

    protocol = {
        "name": "PANEL_C",
        "schema_version": "panel_c_v3_protocol_v1",
        "heldout": True,
        "n_students_requested": 150,
        "n_students_sampled": int(rows.student_key.nunique()),
        "n_cases": int(len(rows)),
        "student_overlap_with_portable_panel_a": 0,
        "panel_b_reuse": False,
        "duplicate_case_count": 0,
        "only_intervention_stages": True,
        "candidate_policy": "feasible_actions_only",
        "one_request_per_case": True,
        "complete_coverage_required": True,
        "gemini_required": True,
        "gemini_executed": False,
        "provider": PROVIDER,
        "requested_model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "prompt_path": str(PROMPT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "prompt_sha256": prompt_hash,
        "prompt_must_not_include": list(FORBIDDEN := [
            "risk_probability",
            "predicted_risk",
            "risk_band",
            "risk_margin",
            "risk_threshold",
            "uncertainty",
            "model_id",
            "final_result",
            "Five-EBM scores",
            "rank positions",
        ]),
        "score_contract": "ordinal_0_3_or_ABSTAIN",
        "primary_metric": "NDCG@3",
        "secondary_metrics": [
            "precision_at_1",
            "mrr",
            "recall_at_3",
            "pairwise_accuracy",
            "exact_best_top1_agreement",
        ],
        "baselines_frozen_before_open": ["B0_action_stage", "B1_rule_score"],
        "abstention_policy": (
            "Preserve every authentic review record. Primary ranking metrics use only "
            "cases with a non-abstained review for every feasible candidate action."
        ),
        "bootstrap": {
            "iterations": 2000,
            "seed": 2026,
            "unit": "query",
            "estimand": "Five-EBM-C0 minus baseline NDCG@3",
        },
        "retry_policy": {
            "max_attempts": 3,
            "retry_on": ["timeout", "http_429", "http_5xx", "transport"],
            "do_not_change_prompt": True,
            "do_not_substitute_model": True,
        },
        "historical_panel_b_is_not_v3_heldout": True,
        "status": "PROTOCOL_FROZEN_AWAITING_GEMINI",
    }
    case_manifest = {
        "schema_version": "panel_c_case_manifest_v1",
        "n_students": int(rows.student_key.nunique()),
        "n_cases": int(len(case_records)),
        "n_eligible_action_slots": int(n_candidates),
        "n_cases_with_no_eligible_action": int(n_empty),
        "student_overlap_with_portable_panel_a": 0,
        "cases": case_records,
    }
    provider_manifest = {
        "schema_version": "panel_c_provider_manifest_v1",
        "status": "NOT_STARTED",
        "provider": PROVIDER,
        "requested_model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "n_cases_planned": int(len(case_records)),
        "n_cases_reviewed": 0,
        "n_provider_failures": 0,
        "model_substitution": False,
    }

    PANEL.mkdir(parents=True, exist_ok=True)
    (PANEL / "PANEL_C_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (PANEL / "PANEL_C_CASE_MANIFEST.json").write_text(json.dumps(case_manifest, indent=2) + "\n", encoding="utf-8")
    (PANEL / "PANEL_C_PROVIDER_MANIFEST.json").write_text(json.dumps(provider_manifest, indent=2) + "\n", encoding="utf-8")

    checksums = {}
    for path in tracked_paths():
        if not path.exists():
            raise SystemExit(f"missing freeze artifact: {path}")
        checksums[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
    checksums["artifacts/recommend_hybrid/v3/panel_c/PANEL_C_PROTOCOL.json"] = sha256_file(PANEL / "PANEL_C_PROTOCOL.json")
    checksums["artifacts/recommend_hybrid/v3/panel_c/PANEL_C_CASE_MANIFEST.json"] = sha256_file(PANEL / "PANEL_C_CASE_MANIFEST.json")
    checksums["artifacts/recommend_hybrid/v3/panel_c/PANEL_C_PROVIDER_MANIFEST.json"] = sha256_file(
        PANEL / "PANEL_C_PROVIDER_MANIFEST.json"
    )

    freeze = {
        "status": "FROZEN_DEVELOPMENT",
        "schema_version": "development_freeze_manifest_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_authority": "Phase4 Hybrid C0",
        "recommendation_authority": "Five-EBM-C0",
        "ranker": "Five-EBM-C0",
        "five_ebm_refit": False,
        "evaluation_semantics": "runtime_equivalent_eligible_only",
        "panel_b_used_for_tuning": False,
        "panel_c_used_for_tuning": False,
        "gemini_full_relabel": False,
        "post_freeze_tuning_permitted": False,
        "DEVELOPMENT_FROZEN": True,
        "POST_FREEZE_TUNING_ALLOWED": False,
        "pre_panel_c_audit": "PASS",
        "invalid_action_root_cause": audit.get("invalid_action_root_cause"),
        "runtime_equivalent_invalid_action_rate": audit.get("runtime_equivalent_invalid_action_rate"),
        "checksums": checksums,
    }
    FREEZE.mkdir(parents=True, exist_ok=True)
    (FREEZE / "DEVELOPMENT_FREEZE_MANIFEST_V2.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    checksum_lines = [f"{digest}  {rel}" for rel, digest in sorted(checksums.items())]
    checksum_lines.append(f"{sha256_file(FREEZE / 'DEVELOPMENT_FREEZE_MANIFEST_V2.json')}  artifacts/recommend_hybrid/v3/freeze/DEVELOPMENT_FREEZE_MANIFEST_V2.json")
    (FREEZE / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    report = f"""# 08 — Development refreeze

**STATUS: FROZEN**

`DEVELOPMENT_FROZEN = true`  
`POST_FREEZE_TUNING_ALLOWED = false`

Reason for V2 freeze: Phase 1 corrected ranking evaluation to runtime-equivalent
eligible-only semantics (evaluator scope bug). Five-EBM-C0 artifacts were not refit.
Risk-router / pipeline wiring were already correct; regression tests were added.

Panel C protocol completed before any provider call:

- students: {protocol['n_students_sampled']}
- cases: {protocol['n_cases']}
- eligible action slots: {n_candidates}
- cases with zero eligible actions: {n_empty}
- portable Panel A student overlap: 0
- model: `{MODEL_NAME}`
- prompt: `{PROMPT_VERSION}`
- prompt_sha256: `{prompt_hash}`

Historical Panel B remains closed and unused.
"""
    (REPORTS / "08_DEVELOPMENT_REFREEZE.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "DEVELOPMENT_FROZEN": True,
                "POST_FREEZE_TUNING_ALLOWED": False,
                "PANEL_C_PROTOCOL": "PASS",
                "n_cases": protocol["n_cases"],
                "prompt_sha256": prompt_hash,
                "freeze_sha256": sha256_file(FREEZE / "DEVELOPMENT_FREEZE_MANIFEST_V2.json"),
            },
            indent=2,
        )
    )


def hashlib_sha(payload: dict) -> str:
    from panel_c_common import canonical_json_bytes, sha256_bytes

    return sha256_bytes(canonical_json_bytes(payload))


if __name__ == "__main__":
    main()
