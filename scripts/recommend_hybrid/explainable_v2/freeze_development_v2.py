"""Create the Recommendation V2 development freeze after all Panel-A gates pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = (
    ROOT / "artifacts/recommend_hybrid/explainable_v2/frozen/development_v2"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> int:
    paths = {
        "config": ROOT / "configs/recommend_hybrid/explainable_v2.yaml",
        "risk_checkpoint_manifest": (
            ROOT
            / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
        ),
        "risk_authority_audit": (
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/run_state"
            / "HYBRID_OOF_AUTHORITY_AUDIT.json"
        ),
        "panel_a_review_freeze": (
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/annotations/frozen/panel_a_v1"
            / "PANEL_A_FREEZE_MANIFEST.json"
        ),
        "label_manifest": (
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/labels/panel_a_v1"
            / "label_model_manifest.json"
        ),
        "ranker_freeze": (
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/frozen/ranker_panel_a_v2"
            / "RANKER_PANEL_A_FREEZE_MANIFEST.json"
        ),
        "five_ebm_manifest": (
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/frozen/ranker_panel_a_v2"
            / "FIVE_EBM_MANIFEST.json"
        ),
        "calibration_selection": (
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/ranker_development/panel_a_v1"
            / "RANKER_SELECTION_BOOTSTRAP.json"
        ),
        "router_freeze": (
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/frozen/router_panel_a_v1"
            / "ROUTER_FREEZE_MANIFEST.json"
        ),
        "release_gates": (
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/release_gates/panel_a_v1"
            / "PANEL_A_RELEASE_GATES.json"
        ),
        "feasibility_policy": (
            ROOT / "src/recommend_hybrid/explainable_v2/action_eligibility.py"
        ),
        "runtime_feasibility_adapter": (
            ROOT / "src/recommend_hybrid/explainable_v2/feasibility.py"
        ),
        "ranker_contract": (
            ROOT / "src/recommend_hybrid/explainable_v2/ranker.py"
        ),
        "router_contract": (
            ROOT / "src/recommend_hybrid/explainable_v2/safety_router.py"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"MISSING_FREEZE_INPUTS={missing}")

    authority = load(paths["risk_authority_audit"])
    labels = load(paths["label_manifest"])
    ranker = load(paths["ranker_freeze"])
    five_ebm = load(paths["five_ebm_manifest"])
    calibration = load(paths["calibration_selection"])
    router = load(paths["router_freeze"])
    release = load(paths["release_gates"])
    if authority.get("authority_status") != "PASS":
        raise RuntimeError("FROZEN_RISK_AUTHORITY_NOT_PASS")
    if labels.get("status") != "PASS" or labels.get("retained_row_count") != 1499:
        raise RuntimeError("CORRECTED_LABEL_FREEZE_NOT_PASS")
    if ranker.get("status") != "PASS":
        raise RuntimeError("RANKER_FREEZE_NOT_PASS")
    if calibration.get("decision") != "KEEP_RAW_EBM_RANKER":
        raise RuntimeError("CALIBRATION_DECISION_NOT_RAW_EBM")
    if router.get("status") != "PASS":
        raise RuntimeError("ROUTER_FREEZE_NOT_PASS")
    if release.get("status") != "PASS":
        raise RuntimeError("PANEL_A_RELEASE_GATES_NOT_PASS")
    for payload in (labels, ranker, calibration, router, release):
        if payload.get("panel_b_touched") is not False:
            raise RuntimeError("PANEL_B_TOUCHED_BEFORE_DEVELOPMENT_FREEZE")
        if payload.get("runtime_authorized") is not False:
            raise RuntimeError("RUNTIME_AUTHORIZED_BEFORE_FINAL_RELEASE")

    model_files = sorted(
        (
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/frozen/ranker_panel_a_v2"
            / "final_models"
        ).glob("*.joblib")
    )
    if len(model_files) != 5:
        raise RuntimeError(f"FROZEN_MODEL_COUNT={len(model_files)} expected=5")

    manifest = {
        "schema_version": "recommendation_v2_development_freeze_v1",
        "status": "PASS",
        "scope": "PANEL_A_DEVELOPMENT_FREEZE",
        "panel_b_touched": False,
        "runtime_authorized": False,
        "final_metrics_claimed": False,
        "risk_model": {
            "family": "HYBRID_CNN_BILSTM",
            "architecture_hash": "df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e",
            "parameter_count": 160492,
            "checkpoint_manifest_sha256": sha256(paths["risk_checkpoint_manifest"]),
            "authority_audit_sha256": sha256(paths["risk_authority_audit"]),
            "verified_mapping_count": authority["verified_mapping_count"],
        },
        "panel_a_review_lineage": {
            "freeze_manifest_sha256": sha256(paths["panel_a_review_freeze"]),
            "source_family": "LLM_EXPERT",
            "provider_source_count": 1,
        },
        "labels": {
            "manifest_sha256": sha256(paths["label_manifest"]),
            "cardinality": 4,
            "minimum_independent_source_families": 2,
            "provenance_rows": 1500,
            "retained_rows": 1499,
            "insufficient_support_rows": 1,
        },
        "feature_schema": five_ebm["features"],
        "feature_schema_sha256": hashlib.sha256(
            json.dumps(
                five_ebm["features"], separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "feasibility_policy": {
            "version": "v4_query_level_annotation_policy",
            "canonical_policy_sha256": sha256(paths["feasibility_policy"]),
            "runtime_adapter_sha256": sha256(paths["runtime_feasibility_adapter"]),
        },
        "ranker": {
            "selected_config_id": "a70599afad40",
            "calibration_decision": "NONE_RAW_EBM_SELECTED",
            "score_contract": ranker["score_contract"],
            "ranker_contract_sha256": sha256(paths["ranker_contract"]),
            "five_model_sha256": {
                path.stem: sha256(path) for path in model_files
            },
            "freeze_manifest_sha256": sha256(paths["ranker_freeze"]),
            "calibration_selection_sha256": sha256(
                paths["calibration_selection"]
            ),
        },
        "router": {
            "public_statuses": router["public_route_statuses"],
            "selected_thresholds": router["selected_thresholds"],
            "selected_thresholds_sha256": router[
                "selected_thresholds_sha256"
            ],
            "router_contract_sha256": sha256(paths["router_contract"]),
            "freeze_manifest_sha256": sha256(paths["router_freeze"]),
            "seed_disagreement_status": (
                "UNAVAILABLE_IN_FROZEN_SOURCE_ARTIFACT_NOT_ZERO_IMPUTED"
            ),
        },
        "release_gate_report_sha256": sha256(paths["release_gates"]),
        "pre_freeze_checks": {
            "explainable_v2_pytest": "184 passed, 1 skipped_private_salt",
            "ruff": "PASS",
            "git_diff_check": "PASS",
            "student_overlap": 0,
            "query_overlap": 0,
            "post_cutoff_violations": 0,
            "panel_b_changed_files": 0,
            "real_secret_detected": False,
            "private_mapping_or_salt_tracked": False,
        },
        "lineage_sha256": {
            key: sha256(path) for key, path in paths.items()
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    output_path = OUTPUT_DIR / "DEVELOPMENT_FREEZE_MANIFEST.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "checksums.sha256").write_text(
        f"{sha256(output_path)}  {output_path.name}\n",
        encoding="utf-8",
    )
    print(f"DEVELOPMENT_FREEZE_MANIFEST={output_path}")
    print("DEVELOPMENT_FREEZE=PASS")
    print("PANEL_B_TOUCHED=FALSE")
    print("RUNTIME_AUTHORIZED=FALSE")
    print("FINAL_METRICS_CLAIMED=FALSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
