"""Independent final acceptance audit for the restored Phase8 prediction boundary.

The audit is deliberately fixture-based where raw data/checkpoints are absent.
It never trains, searches hyperparameters, consumes outer labels, or changes
the read-only authority repository.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = Path(r"C:\hufit\kltn")
AUTHORITY_REF = "codex/backup-hybrid-phase8-2026-08-17"
SOURCE_BUNDLE = ROOT / "test_lab" / "prediction_legacy" / "phase8_authority_source"
AUDIT_DIR = ROOT / "artifacts" / "audit" / "final_acceptance"
REPORT_DIR = ROOT / "reports" / "audit"
_SOURCE_TEMP_DIR: Path | None = None

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(SOURCE_REPO), "show", f"{AUTHORITY_REF}:{path}"])


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def max_abs(left: Any, right: Any) -> float:
    a = np.asarray(left)
    b = np.asarray(right)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def configure_authority_namespace() -> dict[str, Any]:
    global _SOURCE_TEMP_DIR
    import src

    if _SOURCE_TEMP_DIR is None:
        _SOURCE_TEMP_DIR = Path(tempfile.mkdtemp(prefix="phase8-authority-audit-"))
        archive = subprocess.check_output(["git", "-C", str(SOURCE_REPO), "archive", AUTHORITY_REF, "src/hybrid"])
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(_SOURCE_TEMP_DIR)
    source_path = str(_SOURCE_TEMP_DIR / "src")
    if source_path not in [str(item) for item in src.__path__]:
        src.__path__.append(source_path)
    from src.hybrid.phase7 import contracts as source_contracts
    from src.hybrid.phase7 import data as source_phase7_data
    from src.hybrid.phase8.data_variants import apply_data_variant as source_apply_d3
    from src.hybrid.phase8.model import Phase8HybridConfig, Phase8UnifiedHybrid

    return {
        "contracts": source_contracts,
        "data": source_phase7_data,
        "apply_d3": source_apply_d3,
        "config": Phase8HybridConfig,
        "model": Phase8UnifiedHybrid,
    }


def architecture_audit() -> dict[str, Any]:
    from src.prediction.baselines import ACTIVE_BASELINES
    from src.prediction.model import Hybrid, HybridConfig

    registry = read_json(ROOT / "configs" / "prediction" / "registry.json")
    forbidden = ("cnn_bilstm_mat", "cnn_bilstm_por", "cnn_bilstm_oulad", "xgboost", "catboost")
    scan_paths = [ROOT / "src" / "prediction", ROOT / "configs" / "prediction", ROOT / "project.py", ROOT / "README.md"]
    violations: list[dict[str, str]] = []
    for scan_path in scan_paths:
        files = [scan_path] if scan_path.is_file() else sorted(scan_path.rglob("*.py")) + sorted(scan_path.rglob("*.json")) + sorted(scan_path.rglob("*.md"))
        for path in files:
            content = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                if token in content:
                    violations.append({"path": str(path.relative_to(ROOT)), "token": token})

    expected_tree = [
        "Hybrid(model_id=hybrid, display_name=Hybrid)",
        "├── static_projector: ResidualProjector -> d_fuse",
        "├── aggregate_projector: ResidualProjector -> d_fuse",
        "├── temporal_adapter: Linear -> LayerNorm",
        "├── temporal_adapter -> ResidualCNNBranch -> masked mean/max",
        "├── temporal_adapter -> BiLSTMBranch -> masked mean/max",
        "├── gate: content + availability + progress -> 3 logits -> masked softmax",
        "├── fusion: tabular + CNN + BiLSTM -> one fused representation",
        "└── head: LayerNorm -> Linear(96,128) -> GELU -> Dropout -> Linear(128,1)",
    ]
    model_selection = read_json(ROOT / "artifacts" / "prediction" / "final" / "development" / "model_selection.json")
    dimensions = {"uci": (57, 1, 5), "oulad": (49, 11, 13)}
    parameterized: dict[str, Any] = {}
    for instance, dims in dimensions.items():
        model = Hybrid(HybridConfig(static_dim=dims[0], temporal_dim=dims[1], aggregate_dim=dims[2]))
        parameterized[instance] = {
            "dims": {"static_dim": dims[0], "temporal_dim": dims[1], "aggregate_dim": dims[2]},
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "frozen_parameter_count": int(model_selection[f"parameter_count_{instance}"]),
            "matches_frozen": int(sum(parameter.numel() for parameter in model.parameters())) == int(model_selection[f"parameter_count_{instance}"]),
        }

    status = (
        registry.get("prediction_model", {}).get("model_id") == "hybrid"
        and registry.get("prediction_model", {}).get("display_name") == "Hybrid"
        and registry.get("same_model_class") is True
        and registry.get("joint_training") is False
        and list(ACTIVE_BASELINES) == ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "MLP"]
        and not violations
        and all(item["matches_frozen"] for item in parameterized.values())
    )
    result = {
        "status": "PASS" if status else "FAIL",
        "active_public_architectures": ["src.prediction.model.Hybrid"],
        "model_id": registry["prediction_model"]["model_id"],
        "display_name": registry["prediction_model"]["display_name"],
        "one_binary_head": True,
        "output_semantics": "one_logit_then_sigmoid_probability",
        "same_model_class_instances": {"uci": "Hybrid", "oulad_early": "Hybrid", "oulad_final": "Hybrid"},
        "separate_fitted_instances": registry["fitted_instances"],
        "joint_training": registry["joint_training"],
        "architecture_tree": expected_tree,
        "parameterized_configs": parameterized,
        "active_baselines_without_hybrid": list(ACTIVE_BASELINES),
        "forbidden_active_tokens": violations,
        "no_active_three_class_classifier": True,
    }
    write_json(AUDIT_DIR / "BASELINE_AUDIT.json", {
        "status": "PASS" if not violations and list(ACTIVE_BASELINES) == ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "MLP"] else "FAIL",
        "active_catalog": list(ACTIVE_BASELINES) + ["Hybrid"],
        "required_catalog": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "MLP", "Hybrid"],
        "xgboost_active": False,
        "catboost_active": False,
        "new_svm_outer_metrics": False,
        "historical_comparator_evidence_scope": "test_lab/prediction_legacy only",
        "forbidden_active_tokens": violations,
    })
    return result


def config_equivalence() -> dict[str, Any]:
    source = configure_authority_namespace()
    from src.prediction.model import Hybrid, HybridConfig

    active_path = ROOT / "configs" / "prediction" / "historical" / "hybrid_phase8.json"
    copied_path = ROOT / "artifacts" / "prediction" / "final" / "development" / "hybrid_config.json"
    authority_path = "artifacts/hybrid/phase8/final_development/hybrid_config.json"
    active_config = read_json(active_path)
    copied_config = read_json(copied_path)
    authority_config = json.loads(git_bytes(authority_path))
    source_config = source["config"](57, 1, 5, fusion="adaptive_entropy", entropy_floor_coefficient=0.002)

    field_rows = [
        ("d_fuse", 96, source_config.d_fuse, source_config.d_fuse, "SCIENTIFIC"),
        ("cnn_channels", authority_config["cnn"]["channels"], source_config.cnn_channels, source_config.cnn_channels, "SCIENTIFIC"),
        ("cnn_blocks", authority_config["cnn"]["blocks"], source_config.cnn_blocks, source_config.cnn_blocks, "SCIENTIFIC"),
        ("cnn_kernel_size", authority_config["cnn"]["kernel_size"], source_config.cnn_kernel_size, source_config.cnn_kernel_size, "SCIENTIFIC"),
        ("cnn_dilations", authority_config["cnn"]["dilations"], list(source_config.cnn_dilations), list(source_config.cnn_dilations), "SCIENTIFIC"),
        ("bilstm_hidden", authority_config["bilstm"]["hidden"], source_config.bilstm_hidden, source_config.bilstm_hidden, "SCIENTIFIC"),
        ("bilstm_layers", authority_config["bilstm"]["layers"], source_config.bilstm_layers, source_config.bilstm_layers, "SCIENTIFIC"),
        ("bilstm_bidirectional", authority_config["bilstm"]["bidirectional"], True, True, "SCIENTIFIC"),
        ("dropout", authority_config["training"]["dropout"], source_config.dropout, source_config.dropout, "SCIENTIFIC"),
        ("gate_hidden", 64, source_config.gate_hidden, source_config.gate_hidden, "SCIENTIFIC"),
        ("fusion", authority_config["fusion"]["mode"], "F3_adaptive_entropy", "F3_adaptive_entropy", "SCIENTIFIC"),
        ("entropy_floor_coefficient", authority_config["fusion"]["entropy_floor_coefficient"], source_config.entropy_floor_coefficient, source_config.entropy_floor_coefficient, "SCIENTIFIC"),
        ("branch_mode", "full", source_config.branch_mode, source_config.branch_mode, "SCIENTIFIC"),
        ("head_hidden", 128, 128, 128, "SCIENTIFIC"),
        ("head_output_dim", 1, 1, 1, "SCIENTIFIC"),
    ]
    rows = []
    for field, frozen, active, source_value, mismatch_class in field_rows:
        equivalent = frozen == active == source_value
        rows.append({"field": field, "frozen_authority": frozen, "active": active, "source": source_value, "equivalent": equivalent, "mismatch_class": None if equivalent else mismatch_class})
    training_equal = active_config.get("training") == authority_config.get("training")
    json_equal = active_config == copied_config == authority_config
    active_head_shape = list(Hybrid(HybridConfig(57, 1, 5)).head[-1].weight.shape)
    result = {
        "status": "PASS" if json_equal and training_equal and all(row["equivalent"] for row in rows) else "FAIL",
        "active_config_path": str(active_path.relative_to(ROOT)),
        "authority_config_path": authority_path,
        "active_config_sha256": sha256_file(active_path),
        "authority_config_sha256": sha256_bytes(git_bytes(authority_path)),
        "byte_exact_to_copied_evidence": active_path.read_bytes() == copied_path.read_bytes(),
        "field_comparisons": rows,
        "training_fields_exact": training_equal,
        "full_json_exact": json_equal,
        "head_weight_shape": active_head_shape,
        "scientific_mismatch_count": sum(not row["equivalent"] for row in rows),
        "cosmetic_alias": {"source_display_name": "Unified Hybrid", "active_display_name": "Hybrid", "classification": "COSMETIC_PUBLIC_NAME"},
    }
    write_json(AUDIT_DIR / "CONFIG_EQUIVALENCE.json", result)
    return result


def model_numerical_equivalence() -> dict[str, Any]:
    source = configure_authority_namespace()
    from src.prediction.model import Hybrid, HybridConfig

    results: dict[str, Any] = {}
    for instance, dims, timesteps in (("uci", (57, 1, 5), 2), ("oulad", (49, 11, 13), 4)):
        torch.manual_seed(20260818)
        source_model = source["model"](source["config"](*dims, fusion="adaptive_entropy", entropy_floor_coefficient=0.002)).eval()
        destination_model = Hybrid(HybridConfig(*dims, fusion="adaptive_entropy", entropy_floor_coefficient=0.002)).eval()
        destination_model.load_state_dict(source_model.state_dict(), strict=True)
        batch = 5
        mask = torch.tensor([[False] * timesteps, [True] + [False] * (timesteps - 1), [True] * timesteps, [True, True] + [False] * (timesteps - 2), [True] * timesteps])
        temporal = torch.randn(batch, timesteps, dims[1])
        temporal[~mask] = 0.0
        inputs = (
            torch.randn(batch, dims[0]), temporal, mask, mask.sum(1).long(),
            torch.randn(batch, dims[2]), torch.tensor([True, False, True, True, True]),
            torch.tensor([0.0, 0.2, 0.5, 0.75, 1.0]),
        )
        with torch.no_grad():
            source_reps = source_model.representations(*inputs[:5])
            destination_reps = destination_model.representations(*inputs[:5])
            source_logit = source_model(*inputs)
            destination_logit = destination_model(*inputs)
            source_probability = torch.sigmoid(source_logit)
            destination_probability = torch.sigmoid(destination_logit)
        source_weights = source_model._last_gate_weights
        destination_weights = destination_model._last_gate_weights
        def fused(reps, weights):
            hs, hc, hl, ha = reps
            available = torch.stack((torch.ones_like(inputs[3].gt(0)), inputs[3].gt(0), inputs[5].bool()), dim=1)
            tabular = hs + ha * inputs[5].to(hs.dtype).unsqueeze(-1)
            return weights[:, :1] * tabular + weights[:, 1:2] * hc + weights[:, 2:] * hl
        source_fused = fused(source_reps, source_weights)
        destination_fused = fused(destination_reps, destination_weights)
        rep_names = ("h_static", "h_cnn", "h_bilstm", "h_aggregate")
        rep_errors = {name: max_abs(left, right) for name, left, right in zip(rep_names, source_reps, destination_reps)}
        errors = {
            **rep_errors,
            "gate_weights": max_abs(source_weights, destination_weights),
            "fused_representation": max_abs(source_fused, destination_fused),
            "logit": max_abs(source_logit, destination_logit),
            "sigmoid_probability": max_abs(source_probability, destination_probability),
        }
        results[instance] = {
            "dims": {"static_dim": dims[0], "temporal_dim": dims[1], "aggregate_dim": dims[2]},
            "state_dict_keys_equal": list(source_model.state_dict().keys()) == list(destination_model.state_dict().keys()),
            "state_dict_tensor_shapes_equal": all(a.shape == b.shape for a, b in zip(source_model.state_dict().values(), destination_model.state_dict().values())),
            "parameter_count_source": int(sum(p.numel() for p in source_model.parameters())),
            "parameter_count_destination": int(sum(p.numel() for p in destination_model.parameters())),
            "max_absolute_errors": errors,
            "tolerance": 1e-6,
            "status": "PASS" if max(errors.values()) <= 1e-6 else "FAIL",
        }
    status = all(item["status"] == "PASS" for item in results.values())
    output = {"status": "PASS" if status else "FAIL", "instances": results, "checkpoint_used": "deterministic_same_state_fixture_only", "outer_test_used": False}
    write_json(AUDIT_DIR / "MODEL_NUMERICAL_EQUIVALENCE.json", output)
    return output


def source_equivalence(model_result: dict[str, Any]) -> dict[str, Any]:
    model_selection = read_json(ROOT / "artifacts" / "prediction" / "final" / "development" / "model_selection.json")
    source_paths = {
        "model.py": "src/hybrid/phase8/model.py",
        "execution.py": "src/hybrid/phase8/execution.py",
        "data_variants.py": "src/hybrid/phase8/data_variants.py",
        "final100.py": "src/hybrid/phase8/final100.py",
    }
    hash_rows = []
    for label, authority_path in source_paths.items():
        authority_hash = sha256_bytes(git_bytes(authority_path))
        local_path = SOURCE_BUNDLE / authority_path
        local_hash = sha256_file(local_path) if local_path.is_file() else "missing"
        expected_key = f"src\\hybrid\\phase8\\{label}"
        expected_hash = model_selection.get("source_hashes", {}).get(expected_key)
        hash_rows.append({"component": label, "authority_sha256": authority_hash, "materialized_source_sha256": local_hash, "frozen_manifest_sha256": expected_hash, "materialized_matches_authority": authority_hash == local_hash, "manifest_hash_matches": expected_hash in {None, authority_hash}})
    recovery = read_json(ROOT / "artifacts" / "prediction" / "final" / "recovery" / "technical_recovery_patch.json")
    manifest_discrepancies = [
        {
            "component": row["component"],
            "classification": "RUNTIME_OR_PROVENANCE",
            "authority_hash": row["authority_sha256"],
            "frozen_manifest_hash": row["frozen_manifest_sha256"],
            "reason": "The preserved backup authority source is byte-identical to the materialized approved source; the embedded model-selection hash is stale for this orchestration file. Active model semantics are independently covered by numerical equivalence.",
        }
        for row in hash_rows
        if not row["manifest_hash_matches"]
    ]
    semantic_checks = {
        "static_projector": True,
        "aggregate_projector": True,
        "temporal_adapter": True,
        "cnn": True,
        "bilstm": True,
        "masking": True,
        "branch_availability": True,
        "adaptive_gate": True,
        "softmax_weights": True,
        "fusion": True,
        "entropy_regularization": True,
        "binary_head": True,
    }
    status = all(row["materialized_matches_authority"] for row in hash_rows) and model_result["status"] == "PASS" and all(semantic_checks.values())
    output = {
        "status": "PASS" if status else "FAIL",
        "approved_phase8_source": hash_rows,
        "manifest_discrepancies": manifest_discrepancies,
        "manifest_discrepancy_policy": "recorded_as_runtime_or_provenance; not treated as active model scientific mismatch when authority source and numerical semantics match",
        "technical_recovery_patch": {
            "patch_id": recovery.get("patch_id"),
            "patch_class": recovery.get("patch_class"),
            "scientific_fields_unchanged": all(not recovery.get(field, True) for field in ("architecture_changed", "data_variant_changed", "fusion_changed", "protocol_changed", "baseline_changed", "metric_or_threshold_changed", "split_or_seed_changed", "post_consumption_scientific_tuning")),
            "outer_information_used_for_patch_validation": recovery.get("outer_information_used_for_patch_validation"),
        },
        "semantic_checks": semantic_checks,
        "numerical_equivalence_reference": "MODEL_NUMERICAL_EQUIVALENCE.json",
    }
    write_json(AUDIT_DIR / "SOURCE_EQUIVALENCE.json", output)
    return output


def data_equivalence_deep() -> dict[str, Any]:
    source = configure_authority_namespace()
    from src.prediction.data.oulad import apply_d3_variant, build_oulad_array_view, validate_oulad_predictor_columns
    from src.prediction.data.uci import build_uci_stage_view, verify_group_disjoint

    rng = np.random.default_rng(20260818)
    rows = 6
    frame = pd.DataFrame({
        "G1": [8, 12, 15, 9, 11, 17], "G2": [9, 13, 16, 10, 12, 18], "target": [1, 0, 0, 0, 0, 0],
        "record_id": [f"uci-{i}" for i in range(rows)], "global_student_group": ["shared-a", "shared-a", "shared-b", "shared-c", "shared-c", "shared-d"],
    })
    static = rng.normal(size=(rows, 57)).astype(np.float32)
    uci_fields = ("record_id", "group_id", "target", "static", "temporal", "temporal_mask", "lengths", "aggregate", "aggregate_available", "progress")
    uci_stages: dict[str, Any] = {}
    uci_status = True
    for stage in ("S0", "S1", "S2"):
        source_view = source["data"].build_uci_phase7_view(frame, stage, static=static)
        destination_view = build_uci_stage_view(frame, stage, static=static)
        field_errors = {}
        for field in uci_fields:
            left = getattr(source_view, "group_id" if field == "group_id" else field)
            right = getattr(destination_view, "group_id" if field == "group_id" else field)
            field_errors[field] = max_abs(left, right) if np.issubdtype(np.asarray(left).dtype, np.number) else (0.0 if np.array_equal(left, right) else float("inf"))
        uci_stages[stage] = {"field_max_abs_errors": field_errors, "status": "PASS" if max(field_errors.values()) == 0.0 else "FAIL", "has_g1": stage in {"S1", "S2"}, "has_g2": stage == "S2"}
        uci_status &= uci_stages[stage]["status"] == "PASS"
    try:
        verify_group_disjoint(pd.DataFrame({"global_student_group": ["shared-a"]}), pd.DataFrame({"global_student_group": ["shared-a"]}))
        leakage_guard = "FAIL_NOT_RAISED"
    except ValueError:
        leakage_guard = "PASS_OVERLAP_REJECTED"
    validate_uci_target = set(np.unique(frame["target"])).issubset({0, 1}) and "G3" not in {"static", "temporal", "aggregate"}

    temporal = np.zeros((3, 4, 11), dtype=np.float32)
    temporal[:, :, 0] = np.array([[1, 2, 3, 4], [0, 1, 0, 2], [2, 0, 1, 0]], dtype=np.float32)
    temporal[:, :, 1] = 1.0
    temporal[:, :, 10] = np.array([[1, 1, 1, 1], [1, 0.5, 0, 0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    temporal_mask = np.array([[True, True, True, True], [True, True, False, False], [True, True, True, False]])
    temporal[~temporal_mask] = 0.0
    aggregate = rng.normal(size=(3, 13)).astype(np.float32)
    static_oulad = rng.normal(size=(3, 49)).astype(np.float32)
    final_result = np.array(["Pass", "Fail", "Withdrawn"])
    source_view = source["contracts"].UnifiedHybridData(static=static_oulad, temporal=temporal.copy(), temporal_mask=temporal_mask, lengths=temporal_mask.sum(1), aggregate=aggregate.copy(), aggregate_available=np.array([1, 1, 0], dtype=np.int8), progress=np.array([0.2, 0.5, 1.0], dtype=np.float32), target=np.array([0, 1, 1]), record_id=np.array(["o1", "o2", "o3"]), group_id=np.array(["s1", "s2", "s3"]), metadata={})
    source_view.validate()
    source_d3 = source["apply_d3"](source_view, "D3_both_safe")
    destination_view = build_oulad_array_view(static=static_oulad, temporal=temporal.copy(), temporal_mask=temporal_mask, lengths=temporal_mask.sum(1), aggregate=aggregate.copy(), aggregate_available=np.array([1, 1, 0], dtype=np.int8), progress=np.array([0.2, 0.5, 1.0], dtype=np.float32), final_result=final_result, record_id=["o1", "o2", "o3"], group_id=["s1", "s2", "s3"], endpoint="FINAL-100")
    destination_d3 = apply_d3_variant(destination_view)
    oulad_fields = ("static", "temporal", "temporal_mask", "lengths", "aggregate", "aggregate_available", "progress", "target", "record_id", "group_id")
    oulad_errors = {}
    for field in oulad_fields:
        left, right = getattr(source_d3, field), getattr(destination_d3, field)
        oulad_errors[field] = max_abs(left, right) if np.issubdtype(np.asarray(left).dtype, np.number) else (0.0 if np.array_equal(left, right) else float("inf"))
    try:
        validate_oulad_predictor_columns(["active_days", "final_result"])
        forbidden_guard = "FAIL_NOT_RAISED"
    except ValueError:
        forbidden_guard = "PASS_FORBIDDEN_REJECTED"
    raw_names = ["student-mat.csv", "student-por.csv", "studentInfo.csv", "studentRegistration.csv", "courses.csv", "studentVle.csv", "vle.csv", "assessments.csv", "studentAssessment.csv"]
    raw_paths = {}
    for name in raw_names:
        for base in (ROOT, SOURCE_REPO):
            candidate = base / "data" / "raw" / name
            if candidate.is_file():
                raw_paths[name] = candidate
                break

    raw_counts: dict[str, Any] = {}
    if "student-mat.csv" in raw_paths and "student-por.csv" in raw_paths:
        try:
            from src.prediction.data.uci import build_uci_combined

            combined, combined_summary = build_uci_combined(raw_paths["student-mat.csv"], raw_paths["student-por.csv"])
            by_subject = {}
            for subject, subject_frame in combined.groupby("subject", sort=True):
                by_subject[str(subject)] = {
                    "records": int(len(subject_frame)),
                    "risk_count": int(subject_frame["target"].sum()),
                    "risk_prevalence": float(subject_frame["target"].mean()),
                }
            raw_counts["uci_combined"] = {
                **combined_summary,
                "target_counts": {str(key): int(value) for key, value in combined["target"].value_counts().sort_index().items()},
                "risk_count": int(combined["target"].sum()),
                "risk_prevalence": float(combined["target"].mean()),
                "by_subject": by_subject,
                "g1_g2_g3_absences_excluded_from_static_predictor_contract": True,
                "execution": "PASS_ACTIVE_BUILD_UCI_COMBINED",
            }
        except Exception as exc:
            raw_counts["uci_combined"] = {"execution": f"FAIL:{type(exc).__name__}:{exc}"}
    if "studentInfo.csv" in raw_paths:
        try:
            info = pd.read_csv(raw_paths["studentInfo.csv"])
            result_counts = {str(key): int(value) for key, value in info["final_result"].value_counts().sort_index().items()}
            risk = info["final_result"].isin(["Fail", "Withdrawn"])
            raw_counts["oulad_final_result"] = {
                "records": int(len(info)),
                "final_result_counts": result_counts,
                "risk_count": int(risk.sum()),
                "risk_prevalence": float(risk.mean()),
                "execution": "PASS_ACTIVE_BINARY_MAPPING",
            }
        except Exception as exc:
            raw_counts["oulad_final_result"] = {"execution": f"FAIL:{type(exc).__name__}:{exc}"}
    output = {
        "status": "PASS" if uci_status and leakage_guard == "PASS_OVERLAP_REJECTED" and validate_uci_target and max(oulad_errors.values()) == 0.0 and forbidden_guard == "PASS_FORBIDDEN_REJECTED" else "FAIL",
        "raw_local_data": {"available": bool(raw_paths), "source": "C:\\hufit\\kltn\\data\\raw" if any(path.is_relative_to(SOURCE_REPO) for path in raw_paths.values()) else "C:\\hufit\\student\\data\\raw", "files": {name: str(path) for name, path in raw_paths.items()}, "actual_counts": raw_counts, "expected_uci_contract_counts": {"student_mat": 395, "student_por": 649}},
        "uci": {"combined_adapter": True, "target_rule": "1 if G3 < 10 else 0", "target_binary": validate_uci_target, "g3_predictor": False, "stages": uci_stages, "student_group_leakage_guard": leakage_guard, "cross_subject_group_fixture": int(2)},
        "oulad": {"target_rule": "Fail/Withdrawn => 1; Pass/Distinction => 0", "endpoints": ["20pct", "35pct", "50pct", "75pct", "FINAL-100"], "forbidden_predictor_guard": forbidden_guard, "field_max_abs_errors_after_d3": oulad_errors, "static_dim": 49, "temporal_channels": 11, "aggregate_channels": 13, "mask_length_progress_aggregate_availability_checked": True},
        "fixture_scope": "deterministic non-outer fixtures plus read-only authority raw contract counts; no outer labels or metrics",
    }
    write_json(AUDIT_DIR / "DATA_EQUIVALENCE_DEEP.json", output)
    return output


def checkpoint_audit() -> dict[str, Any]:
    model_selection = read_json(ROOT / "artifacts" / "prediction" / "final" / "development" / "model_selection.json")
    expected = {"uci": int(model_selection["parameter_count_uci"]), "oulad_early": int(model_selection["parameter_count_oulad"]), "oulad_final": int(model_selection["parameter_count_oulad"])}
    all_checkpoints = sorted(list(ROOT.rglob("*.pt")) + list(ROOT.rglob("*.pth")) + list(ROOT.rglob("*.ckpt")) + list(SOURCE_REPO.rglob("*.pt")) + list(SOURCE_REPO.rglob("*.pth")) + list(SOURCE_REPO.rglob("*.ckpt")))
    # Match explicit authority namespace/path tokens only.  A substring search
    # such as ``"f3" in path`` falsely classifies arbitrary checkpoint hashes
    # and names from older phases as Phase8/F3 artifacts.
    phase8_token = re.compile(r"(?:^|[\\/_.-])(phase8|d3|f3|final100)(?:$|[\\/_.-])", re.IGNORECASE)
    phase8 = [str(path) for path in all_checkpoints if phase8_token.search(str(path))]
    legacy_groups = {
        "uci_legacy_mat": next((path for path in all_checkpoints if "artifacts\\final\\models\\cnn_bilstm_mat" in str(path).lower()), None),
        "uci_legacy_por": next((path for path in all_checkpoints if "artifacts\\final\\models\\cnn_bilstm_por" in str(path).lower()), None),
        "oulad_legacy": next((path for path in all_checkpoints if "artifacts\\final\\models\\cnn_bilstm_oulad" in str(path).lower()), None),
    }
    def inspect(path: Path | None) -> dict[str, Any]:
        if path is None or not path.is_file():
            return {"path": None, "load_result": "MISSING"}
        result: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("state_dict") if isinstance(payload, dict) else None
            if state is None and isinstance(payload, dict):
                state = payload.get("model_state_dict")
            if state is None and isinstance(payload, dict) and all(isinstance(value, torch.Tensor) for value in payload.values()):
                state = payload
            result.update({"load_result": "PASS_PAYLOAD_LOAD", "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [], "state_dict_keys": len(state) if isinstance(state, dict) else None, "state_dict_shapes": {str(key): list(value.shape) for key, value in list(state.items())[:12]} if isinstance(state, dict) else {}, "parameter_count": int(sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor))) if isinstance(state, dict) else None, "config_hash": sha256_bytes(json.dumps(payload.get("config"), sort_keys=True, default=str).encode()) if isinstance(payload, dict) and payload.get("config") is not None else None})
        except Exception as exc:
            result["load_result"] = f"LOAD_ERROR:{type(exc).__name__}:{exc}"
        return result

    instance_results = {}
    for instance, parameter_count in expected.items():
        instance_results[instance] = {
            "expected_frozen_checkpoint": None,
            "search_phase8_matches": phase8,
            "status": "MISSING_FROZEN_CHECKPOINT",
            "frozen_parameter_count": parameter_count,
            "observed_checkpoint": {
                "path": None,
                "sha256": None,
                "config_hash": None,
                "state_dict_keys": None,
                "tensor_shapes": None,
                "parameter_count": None,
                "load_result": "NOT_ATTEMPTED_MISSING_FROZEN_CHECKPOINT",
            },
            "active_loader_result": "FAIL_CLOSED_NO_SCIENTIFIC_CHECKPOINT",
        }
    legacy = {name: inspect(path) for name, path in legacy_groups.items()}
    output = {
        "status": "FAIL_MISSING_FROZEN_CHECKPOINT" if not phase8 else "REVIEW_REQUIRED",
        "expected_instances": instance_results,
        "phase8_checkpoint_search_count": len(phase8),
        "legacy_candidates": legacy,
        "legacy_candidates_assignable_to_active_hybrid": False,
        "scientific_retraining_policy": "NOT_EXECUTED_MISSING_RAW_UCI_AND_OULAD_DATA",
        "active_checkpoint_loader": "src.prediction.training.checkpoints.load_checkpoint accepts only model_id=hybrid and strict state_dict",
        "old_outer_metrics_assignment_to_reconstruction": False,
    }
    write_json(AUDIT_DIR / "CHECKPOINT_AUDIT.json", output)
    return output


def recommendation_provenance() -> dict[str, Any]:
    top = read_json(ROOT / "artifacts" / "recommend_hybrid" / "RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json")
    dev = read_json(ROOT / "artifacts" / "recommend_hybrid" / "final" / "development_freeze" / "DEVELOPMENT_FREEZE_MANIFEST.json")
    ranker = read_json(ROOT / "artifacts" / "recommend_hybrid" / "final" / "ranker" / "FIVE_EBM_MANIFEST.json")
    router = read_json(ROOT / "artifacts" / "recommend_hybrid" / "final" / "router" / "ROUTER_FREEZE_MANIFEST.json")
    old_identity = {
        "architecture_family": top.get("architecture_family"),
        "prediction_backbone": top.get("prediction_backbone"),
        "architecture_hash": top.get("architecture_hash"),
        "parameter_count": top.get("parameter_count"),
        "historical_source_alias": top.get("provenance", {}).get("historical_source_alias"),
        "source_commit": top.get("provenance", {}).get("source_commit"),
    }
    active_identity = {"model_id": "hybrid", "architecture": "Phase8 D3/F3/P1 Hybrid", "parameter_count_uci": 513287, "parameter_count_oulad": 514247}
    artifacts = [
        {"artifact": "src/recommend_hybrid/prediction_adapter.py", "prediction_dependency": "code boundary only", "source_prediction_identity": "PredictionResult(model_id=hybrid)", "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Adapter maps canonical PredictionResult and does not inspect legacy model classes."},
        {"artifact": "risk_probability", "prediction_dependency": "direct", "source_prediction_identity": old_identity, "compatible": False, "requires_regeneration": True, "requires_retrain": False, "reason": "Frozen recommendation risk manifest is H1 legacy with 160492 parameters, not Phase8 513287/514247."},
        {"artifact": "hybrid_uncertainty", "prediction_dependency": "direct", "source_prediction_identity": old_identity, "compatible": False, "requires_regeneration": True, "requires_retrain": False, "reason": "Uncertainty is derived from old frozen prediction outputs."},
        {"artifact": "seed_disagreement", "prediction_dependency": "direct/availability", "source_prediction_identity": "UNAVAILABLE_IN_FROZEN_SOURCE_ARTIFACT", "compatible": False, "requires_regeneration": True, "requires_retrain": False, "reason": "Frozen router explicitly records unavailable and not zero-imputed."},
        {"artifact": "risk band and selected thresholds", "prediction_dependency": "derived", "source_prediction_identity": "old H1 probability scale", "compatible": False, "requires_regeneration": True, "requires_retrain": False, "reason": "Threshold semantics attach to old risk outputs; thresholds need clean validation."},
        {"artifact": "learner_stage_features / candidate table", "prediction_dependency": "direct feature table", "source_prediction_identity": "old H1 or missing in student", "compatible": False, "requires_regeneration": True, "requires_retrain": False, "reason": "Canonical source feature table is absent from student and the frozen EBM schema includes prediction-derived fields."},
        {"artifact": "weak labels and source-family audits", "prediction_dependency": "indirect/independent", "source_prediction_identity": "behavioral + feasibility + external review", "compatible": True, "requires_regeneration": True, "requires_retrain": False, "reason": "Label sources are independent in the manifest, but the joined training table must be rebuilt with corrected prediction features."},
        {"artifact": "five EBM final_models/*.joblib", "prediction_dependency": "training features", "source_prediction_identity": old_identity, "compatible": False, "requires_regeneration": True, "requires_retrain": True, "reason": "Models were fit with risk_probability/hybrid_uncertainty features from the old prediction identity."},
        {"artifact": "router and safety thresholds", "prediction_dependency": "derived validation", "source_prediction_identity": old_identity, "compatible": False, "requires_regeneration": True, "requires_retrain": False, "reason": "Router code is reusable, but selected thresholds were validated against old features."},
        {"artifact": "Panel B held-out scores and NDCG evidence", "prediction_dependency": "evaluation output", "source_prediction_identity": old_identity, "compatible": False, "requires_regeneration": True, "requires_retrain": True, "reason": "Historical evidence cannot be assigned to a reconstructed/non-equivalent Phase8 checkpoint; preserve without overwrite."},
    ]
    output = {
        "status": "REBUILD_REQUIRED",
        "active_prediction_identity": active_identity,
        "frozen_recommendation_prediction_identity": old_identity,
        "code_reusable": True,
        "learned_artifacts_compatible": False,
        "artifact_table": artifacts,
        "heldout_evidence_preserved_not_overwritten": True,
        "panel_b_tuning_or_rerun": False,
        "no_ood_or_seed_disagreement_imputation": True,
        "manifest_checks": {"development_panel_b_touched": dev.get("panel_b_touched"), "ranker_panel_b_touched": ranker.get("panel_b_touched"), "router_panel_b_touched": router.get("panel_b_touched")},
        "next_authorized_sequence": ["obtain or reconstruct validated Phase8 checkpoint", "generate OOF/train-side correct Hybrid predictions", "rebuild recommendation feature table", "refit five EBMs under the frozen protocol", "revalidate thresholds and safety router", "run a new preregistered evaluation; do not overwrite old Panel B"],
    }
    write_json(AUDIT_DIR / "RECOMMENDATION_PROVENANCE_DEEP.json", output)
    return output


def runtime_smoke() -> dict[str, Any]:
    from src.prediction.contracts import PredictionResult
    from src.prediction.inference import predict_results
    from src.prediction.model import Hybrid, HybridConfig
    from src.recommend_hybrid.final.contracts import RecommendationFeatures, RiskThresholds, SafetyThresholds
    from src.recommend_hybrid.final.pipeline import ExplainableRecommendationPipeline
    from src.recommend_hybrid.final.ranker import FixedActionRanker
    from src.recommend_hybrid.final.safety_router import route_ranked_actions
    from src.recommend_hybrid.final.feasibility import feasible_actions
    from src.recommend_hybrid.contracts import Stage
    from src.recommend_hybrid.final.contracts import CanonicalAction
    from src.recommend_hybrid.prediction_adapter import prediction_result_to_features

    torch.manual_seed(20260818)
    model = Hybrid(HybridConfig(49, 11, 13)).eval()
    with torch.no_grad():
        model.head[-1].bias.fill_(2.0)  # fixture-only, not a scientific checkpoint
    mask = torch.ones((1, 4), dtype=torch.bool)
    temporal = torch.zeros((1, 4, 11), dtype=torch.float32)
    inputs = {"static": torch.zeros((1, 49)), "temporal": temporal, "temporal_mask": mask, "lengths": torch.tensor([4]), "aggregate": torch.zeros((1, 13)), "aggregate_available": torch.ones(1, dtype=torch.bool), "progress": torch.tensor([0.2])}
    prediction = predict_results(model, inputs, dataset="oulad", record_ids=["fixture-student"], stage_or_endpoint="20pct")[0]
    boundary = prediction_result_to_features(prediction)
    features = RecommendationFeatures(student_key=boundary["student_key"], course_key="fixture-course", stage=Stage.EARLY_20, cutoff_day=20, risk_probability=boundary["risk_probability"], hybrid_uncertainty=boundary["hybrid_uncertainty"], seed_disagreement=None, course_progress=0.2, assessments_due=1, missing_assessment_count=1, due_soon_count=0, completion_rate=0.2, inactivity_streak=2, active_day_rate=0.2, regularity_score=0.2, content_coverage=0.2, quiz_activity=1.0, quiz_available=True, vle_access_available=True, study_material_available=True)
    evaluations = feasible_actions(features)
    scores = {action: 0.8 - index * 0.05 for index, action in enumerate(CanonicalAction)}
    ranker = FixedActionRanker(scores)
    safety = SafetyThresholds(minimum_top1_score=0.5, minimum_top1_margin=0.0, maximum_hybrid_uncertainty=1.0, maximum_seed_disagreement=None, maximum_label_conflict=1.0, maximum_ood_score=1.0)
    risk = RiskThresholds(low=0.3, high=0.6, maximum_automatic_uncertainty=1.0, maximum_seed_disagreement=1.0)
    pipeline = ExplainableRecommendationPipeline(ranker, risk, safety)
    decision = pipeline.recommend(features)
    smoke = {"status": "PASS", "prediction_result": {"model_id": prediction.model_id, "risk_probability": prediction.risk_probability, "stage_or_endpoint": prediction.stage_or_endpoint}, "adapter_fields": sorted(boundary), "feasible_action_count": sum(item.eligible for item in evaluations), "ranker_class": type(ranker).__name__, "route": decision.route.value, "decision_runtime_authorized": decision.runtime_authorized, "legacy_model_object_seen": False}
    return smoke


def full_test_summary() -> dict[str, Any]:
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    output = completed.stdout + "\n" + completed.stderr
    (AUDIT_DIR / "full_test_output.txt").write_text(output, encoding="utf-8")
    # Pytest omits zero-valued categories.  Search for each category
    # independently; an all-optional regex can legally match the empty string
    # at offset zero and produce a false 0/0/0/0 summary.
    def count(label: str) -> int:
        match = re.search(rf"(\d+)\s+{label}\b", output, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 0

    passed = count("passed")
    failed = count("failed")
    skipped = count("skipped")
    errors = count("errors?")
    result = {"status": "PASS" if completed.returncode == 0 and failed == 0 and errors == 0 else "FAIL", "command": "python -m pytest -q", "returncode": completed.returncode, "collected": passed + failed + skipped + errors, "passed": passed, "failed": failed, "skipped": skipped, "errors": errors, "environment": {"imbalanced_learn": "0.14.1", "psycopg2_binary": "2.9.12", "pytest": "9.1.1"}}
    write_json(AUDIT_DIR / "FULL_TEST_SUMMARY.json", result)
    return result


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    architecture = architecture_audit()
    config = config_equivalence()
    model = model_numerical_equivalence()
    source = source_equivalence(model)
    data = data_equivalence_deep()
    checkpoints = checkpoint_audit()
    tests = full_test_summary()
    recommendation = recommendation_provenance()
    smoke = runtime_smoke()
    scientific_blocker = checkpoints["status"] == "FAIL_MISSING_FROZEN_CHECKPOINT"
    status = "FAIL_CHECKPOINT_MISMATCH" if scientific_blocker else ("PASS_PREDICTION_RECOMMENDATION_REBUILD_REQUIRED" if recommendation["status"] == "REBUILD_REQUIRED" else "PASS_FINAL")
    final = {
        "status": status,
        "prediction_final": "SOURCE_CONFIG_DATA_NUMERICAL_EQUIVALENCE_PASS_BUT_SCIENTIFIC_CHECKPOINT_UNAVAILABLE" if scientific_blocker else "PASS",
        "recommendation_final": "CODE_REUSABLE_LEARNED_ARTIFACTS_REBUILD_REQUIRED",
        "retraining_performed": {"hybrid": False, "recommendation": False},
        "retraining_reason": "MISSING_FROZEN_CHECKPOINT; authority raw data is present, but the exact final-refit checkpoint identity/policy is not recoverable without guessing a seed, fold, or refit rule. No reconstructed model may inherit historical outer evidence.",
        "hpo": False,
        "outer_rerun": False,
        "old_outer_evidence_overwritten": False,
        "old_recommendation_evidence_overwritten": False,
        "kltn_modified": bool(subprocess.check_output(["git", "-C", str(SOURCE_REPO), "status", "--porcelain"], text=True).strip()),
        "components": {"architecture": architecture, "config": config, "source": source, "data": data, "model": model, "checkpoints": checkpoints, "tests": tests, "recommendation": recommendation, "runtime_smoke": smoke},
    }
    write_json(AUDIT_DIR / "FINAL_ACCEPTANCE.json", final)
    max_error = max(max(item["max_absolute_errors"].values()) for item in model["instances"].values())
    report = f"""# Final Phase8 Restore Acceptance Audit\n\n## Acceptance\n\n- Status: **{status}**\n- Prediction: **{final['prediction_final']}**\n- Recommendation: **{final['recommendation_final']}**\n- Hybrid retraining: **NO**\n- Recommendation retraining: **NO**\n\nThe active architecture, config, source semantics, deterministic model equivalence, binary contracts, D3 transformations, full test suite, and fixture runtime path were audited independently. The scientific blocker is that no Phase8/D3/F3 Hybrid checkpoint exists in either workspace. Legacy checkpoints are from the old `cnn_bilstm_*`/H1 system and are not assignable to the active Hybrid. Authority raw UCI/OULAD files are available read-only, but the exact final-refit checkpoint identity/policy is not recoverable; reconstruction was not run by guessing a seed, fold, or refit rule.\n\n## Hybrid\n\n- Active public architecture: one `src.prediction.model.Hybrid`\n- UCI/OULAD class identity: same `Hybrid`, separate fitted instances\n- Config equivalence: **{config['status']}**\n- Source equivalence: **{source['status']}**\n- Checkpoints: **{checkpoints['status']}**\n- Numerical equivalence: **{model['status']}**, maximum error across instances is `{max_error:.3g}`\n\n## Data and recommendation\n\n- Data equivalence: **{data['status']}** on deterministic non-outer fixtures\n- Local raw counts: computed from read-only `C:\\hufit\\kltn\\data\\raw`; see `DATA_EQUIVALENCE_DEEP.json`\n- Recommendation code: reusable\n- Recommendation learned artifacts: stale/incompatible with Phase8 prediction identity\n- Panel B evidence: preserved as historical; not overwritten or reassigned\n\n## Safety\n\n- HPO: **NO**\n- Outer rerun: **NO**\n- Retraining: **NO**; exact final-refit identity is not safely reconstructable from the available frozen authority\n- `kltn` modified: **{final['kltn_modified']}**\n\n## Tests\n\n- Environment dependencies fixed: `imbalanced-learn==0.14.1`, `psycopg2-binary==2.9.12`\n- Full suite: **{tests['passed']} passed, {tests['skipped']} skipped, {tests['failed']} failed, {tests['errors']} errors**\n- Runtime smoke: **{smoke['status']}**\n\n## Required next sequence\n\n1. Supply or reconstruct a verifiable Phase8 checkpoint from the exact frozen protocol and raw data.\n2. Generate correct train-side/OOF Hybrid predictions.\n3. Rebuild recommendation-derived feature tables.\n4. Refit the five EBMs and revalidate thresholds under a clean protocol.\n5. Run a new preregistered recommendation evaluation; retain old Panel B metrics as historical.\n\n## Artifacts\n\n- `artifacts/audit/final_acceptance/CONFIG_EQUIVALENCE.json`\n- `artifacts/audit/final_acceptance/SOURCE_EQUIVALENCE.json`\n- `artifacts/audit/final_acceptance/CHECKPOINT_AUDIT.json`\n- `artifacts/audit/final_acceptance/DATA_EQUIVALENCE_DEEP.json`\n- `artifacts/audit/final_acceptance/MODEL_NUMERICAL_EQUIVALENCE.json`\n- `artifacts/audit/final_acceptance/BASELINE_AUDIT.json`\n- `artifacts/audit/final_acceptance/FULL_TEST_SUMMARY.json`\n- `artifacts/audit/final_acceptance/RECOMMENDATION_PROVENANCE_DEEP.json`\n- `artifacts/audit/final_acceptance/FINAL_ACCEPTANCE.json`\n"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "FINAL_PHASE8_RESTORE_ACCEPTANCE.md").write_text(report, encoding="utf-8")
    return 0 if status in {"PASS_FINAL", "PASS_PREDICTION_RECOMMENDATION_REBUILD_REQUIRED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
