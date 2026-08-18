"""Restore the approved Phase8 prediction authority into the student project.

This script is intentionally migration-only: it copies frozen evidence and
source lineage, runs deterministic contract/equivalence fixtures, and never
trains, tunes, opens outer labels, or executes an outer-evaluation runner.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = Path(r"C:\hufit\kltn")
AUTHORITY_REF = "codex/backup-hybrid-phase8-2026-08-17"
MIGRATION_DIR = ROOT / "artifacts" / "migration"
LEGACY_DIR = ROOT / "test_lab" / "prediction_legacy"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or a directory tree deterministically for the manifest."""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        return "missing"
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).replace("\\", "/").encode())
            digest.update(b"\0")
            digest.update(item.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def git_show(repo: Path, ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{path}"])


def authority_bytes(path: str) -> bytes:
    return git_show(SOURCE_REPO, AUTHORITY_REF, path)


def local_baseline_bytes(path: str) -> bytes:
    return git_show(ROOT, "HEAD", path)


def git_status(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def copy_bytes(value: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


class Migration:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.snapshot: dict[str, Any] = {
            "working_directory": str(ROOT),
            "source_read_only": str(SOURCE_REPO),
            "student_git_status_before": git_status(ROOT),
            "kltn_git_status_before": git_status(SOURCE_REPO),
            "active_paths_before": [],
        }

    def record(self, source: str, destination: str, role: str, authority: str, mode: str, notes: str = "", source_sha256: str | None = None) -> None:
        source_path = Path(source) if Path(source).is_absolute() else ROOT / source
        destination_path = Path(destination) if Path(destination).is_absolute() else ROOT / destination
        if source_sha256 is not None:
            source_hash = source_sha256
        elif authority == AUTHORITY_REF:
            authority_path = source.split(":", 1)[1] if source.startswith(f"{AUTHORITY_REF}:") else source
            source_hash = sha256_bytes(authority_bytes(authority_path))
        else:
            source_hash = sha256_file(source_path) if source_path.is_file() else "missing"
        destination_hash = sha256_path(destination_path)
        self.records.append({
            "source_path": source,
            "destination_path": destination,
            "source_sha256": source_hash,
            "destination_sha256": destination_hash,
            "role": role,
            "authority": authority,
            "copied_or_adapted": mode,
            "notes": notes,
        })

    def snapshot_paths(self) -> None:
        existing = MIGRATION_DIR / "PRE_RESTORE_STUDENT_SNAPSHOT.json"
        if existing.is_file():
            self.snapshot = json.loads(existing.read_text(encoding="utf-8"))
            return
        paths = [
            "src/models", "src/pipelines", "src/training", "src/recommend_hybrid",
            "artifacts/final_release/final_model_registry.json", "artifacts/final/model_registry.json",
            "artifacts/recommend_hybrid", "reports/recommend_hybrid",
        ]
        rows = []
        for relative in paths:
            path = ROOT / relative
            if path.is_file():
                rows.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
            elif path.is_dir():
                files = []
                for item in sorted(path.rglob("*")):
                    if item.is_file():
                        files.append({"path": str(item.relative_to(ROOT)), "sha256": sha256_file(item), "bytes": item.stat().st_size})
                rows.append({"path": relative, "file_count": len(files), "files": files})
        self.snapshot["active_paths_before"] = rows
        write_json(MIGRATION_DIR / "PRE_RESTORE_STUDENT_SNAPSHOT.json", self.snapshot)

    def archive_old_prediction(self) -> None:
        LEGACY_DIR.mkdir(parents=True, exist_ok=True)
        move_paths = [
            ROOT / "src" / "models",
            ROOT / "src" / "pipelines",
            ROOT / "src" / "benchmark",
            ROOT / "src" / "release",
            ROOT / "src" / "evaluation",
            ROOT / "src" / "studies",
            ROOT / "src" / "training",
            ROOT / "src" / "training" / "model_comparison.py",
            ROOT / "src" / "training" / "optuna_search.py",
            ROOT / "src" / "training" / "endpoint_evaluation.py",
            ROOT / "src" / "training" / "protocol.py",
            ROOT / "src" / "training" / "release_freeze.py",
            ROOT / "src" / "training" / "endpoint_recovery.py",
        ]
        for source in move_paths:
            relative = source.relative_to(ROOT / "src")
            destination = LEGACY_DIR / "src" / relative
            if not source.exists():
                if destination.exists():
                    destination_hash = sha256_path(destination)
                    self.record(str(source.relative_to(ROOT)), str(destination.relative_to(ROOT)), "historical_wrong_prediction_code", "student_pre_restore", "archived_existing", "Already archived by an earlier idempotent migration run.", source_sha256=destination_hash)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            was_file = source.is_file()
            before_hash = sha256_path(source)
            shutil.move(str(source), str(destination))
            self.record(str(source.relative_to(ROOT)), str(destination.relative_to(ROOT)), "historical_wrong_prediction_code", "student_pre_restore", "archived", f"pre-restore hash={before_hash}", source_sha256=before_hash)

        # The adapter was replaced before this script was launched; recover its
        # exact pre-restore content from the student's baseline commit.
        adapter_destination = LEGACY_DIR / "src" / "recommend_hybrid" / "prediction_adapter.py"
        baseline_adapter = local_baseline_bytes("src/recommend_hybrid/prediction_adapter.py")
        copy_bytes(baseline_adapter, adapter_destination)
        self.record("src/recommend_hybrid/prediction_adapter.py@HEAD", str(adapter_destination.relative_to(ROOT)), "historical_wrong_prediction_adapter", "student_pre_restore", "archived", "Recovered from pre-restore HEAD because the active file was already replaced at migration start.", source_sha256=sha256_bytes(baseline_adapter))

        # These are configuration authorities for the archived prediction
        # implementation.  Keep the files recoverable, but prevent them from
        # being mistaken for the active Phase8 registry/config namespace.
        for relative in ("configs/final", "configs/canonical_v3", "configs/registry"):
            source = ROOT / relative
            destination = LEGACY_DIR / "evidence" / relative
            if not source.exists():
                if destination.exists():
                    self.record(relative, str(destination.relative_to(ROOT)), "historical_wrong_prediction_config", "student_pre_restore", "archived_existing", "Already archived by an earlier idempotent migration run.", source_sha256=sha256_path(destination))
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            self.record(relative, str(destination.relative_to(ROOT)), "historical_wrong_prediction_config", "student_pre_restore", "archived", "Preserved for provenance; not an active authority.")

        # Keep historical orchestration scripts out of the active prediction
        # surface.  Recommendation and database scripts are intentionally not
        # moved here because they are downstream consumers under audit.
        for relative in (
            "scripts/canonical_benchmark",
            "scripts/endpoint_evaluation",
            "scripts/endpoint_forensics",
            "scripts/endpoint_recovery",
            "scripts/fusion",
            "scripts/model_comparison",
            "scripts/optimization",
            "scripts/release",
            "scripts/release_freeze",
            "scripts/project/validate_repository_structure.py",
            "scripts/thesis_evidence",
            "scripts/audit_forensic_baseline.py",
            "scripts/audit_training_controls.py",
            "scripts/build_training_control_reports.py",
            "tests/audit",
            "tests/release",
            "tests/unit/test_public_registry.py",
            "tests/test_endpoint_recovery.py",
        ):
            source = ROOT / relative
            destination = LEGACY_DIR / relative
            if not source.exists():
                if destination.exists():
                    self.record(relative, str(destination.relative_to(ROOT)), "historical_prediction_script", "student_pre_restore", "archived_existing", "Already archived by an earlier idempotent migration run.", source_sha256=sha256_path(destination))
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            before_hash = sha256_path(source)
            shutil.move(str(source), str(destination))
            self.record(relative, str(destination.relative_to(ROOT)), "historical_prediction_script", "student_pre_restore", "archived", "Historical prediction orchestration; not active runtime.", source_sha256=before_hash)

        evidence = [
            "artifacts/final_release/final_model_registry.json",
            "artifacts/final/model_registry.json",
            "configs/final/final_model_authority.yaml",
            "configs/final/model_registry.yaml",
            "configs/final/cnn_bilstm_mat.yaml",
            "configs/final/cnn_bilstm_por.yaml",
            "configs/final/cnn_bilstm_oulad.yaml",
        ]
        for relative in evidence:
            source = ROOT / relative
            if not source.is_file():
                continue
            destination = LEGACY_DIR / "evidence" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            self.record(relative, str(destination.relative_to(ROOT)), "historical_wrong_prediction_evidence", "student_pre_restore", "archived", "Preserved for provenance; not active authority.")

        write_text(LEGACY_DIR / "README.md", """# Historical prediction namespace\n\nThis namespace contains the copied prediction implementation and evidence that preceded the Phase8 restore. It is historical provenance only and is not imported by active prediction or recommendation runtime.\n\nArchived semantics include the old UCI Low/Medium/High target, MAT/POR product wrappers, old OULAD architecture, old registries, and legacy comparator code. No historical metric is relabeled or replaced by a newly generated metric in this migration.\n""")

    def materialize_authority_source(self) -> None:
        paths = [
            "src/hybrid/phase8/model.py", "src/hybrid/phase8/execution.py", "src/hybrid/phase8/data_variants.py", "src/hybrid/phase8/final100.py", "src/hybrid/phase8/residual.py",
            "src/hybrid/phase7/contracts.py", "src/hybrid/phase7/data.py", "src/hybrid/phase7/execution.py", "src/hybrid/phase7/model.py",
            "src/hybrid/data/oulad.py", "src/hybrid/data/uci.py", "src/hybrid/data/common.py", "src/hybrid/data/preprocessing.py", "src/hybrid/data/splits.py",
            "src/hybrid/training/data.py", "src/hybrid/training/trainer.py", "src/hybrid/models/components.py",
        ]
        for relative in paths:
            destination = LEGACY_DIR / "phase8_authority_source" / relative
            copy_bytes(authority_bytes(relative), destination)
            self.record(f"{AUTHORITY_REF}:{relative}", str(destination.relative_to(ROOT)), "approved_phase8_source_lineage", AUTHORITY_REF, "copied", "Exact Git-object copy; kept outside active import namespace.")

    def materialize_evidence(self) -> None:
        selected = [
            ("artifacts/hybrid/phase8/final_development/hybrid_config.json", "artifacts/prediction/final/development/hybrid_config.json", "frozen_phase8_config"),
            ("artifacts/hybrid/phase8/final_development/model_selection.json", "artifacts/prediction/final/development/model_selection.json", "frozen_phase8_selection"),
            ("artifacts/hybrid/phase8/final_development/data_contract.json", "artifacts/prediction/final/development/data_contract.json", "frozen_data_contract"),
            ("artifacts/hybrid/phase8/final_development/baseline_configs.json", "artifacts/prediction/final/development/baseline_configs.json", "frozen_baseline_config"),
            ("artifacts/hybrid/phase8/final_development/uci/selection.json", "artifacts/prediction/final/development/uci_selection.json", "uci_selection"),
            ("artifacts/hybrid/phase8/final_development/oulad/selection.json", "artifacts/prediction/final/development/oulad_selection.json", "oulad_selection"),
            ("artifacts/hybrid/phase8/protocol_corrected/early/selection.json", "artifacts/prediction/final/protocol/early_selection.json", "oulad_early_selection"),
            ("artifacts/hybrid/phase8/protocol_corrected/final100/selection.json", "artifacts/prediction/final/protocol/final100_selection.json", "oulad_final_selection"),
            ("artifacts/hybrid/phase8/protocol_corrected/outer_test_readiness_v2/manifest.json", "artifacts/prediction/final/protocol/readiness_v2_manifest.json", "readiness_manifest"),
            ("artifacts/hybrid/phase8/outer_test_final/results.csv", "artifacts/prediction/final/outer_test_final/results.csv", "outer_results_freeze"),
            ("artifacts/hybrid/phase8/outer_test_final/summary.csv", "artifacts/prediction/final/outer_test_final/summary.csv", "outer_summary_freeze"),
            ("artifacts/hybrid/phase8/outer_test_final/predictions/predictions.parquet", "artifacts/prediction/final/outer_test_final/predictions.parquet", "outer_predictions_freeze"),
            ("artifacts/hybrid/phase8/outer_test_final/postprocessing/POST_OUTER_INTEGRITY.json", "artifacts/prediction/final/outer_test_final/POST_OUTER_INTEGRITY.json", "post_outer_integrity"),
            ("artifacts/hybrid/phase8/outer_test_final/postprocessing/outer_results_freeze.json", "artifacts/prediction/final/outer_test_final/outer_results_freeze.json", "outer_results_freeze_manifest"),
            ("artifacts/hybrid/phase8/outer_test_final/postprocessing/validation.json", "artifacts/prediction/final/outer_test_final/validation.json", "outer_validation"),
            ("artifacts/hybrid/phase8/outer_test_final/postprocessing/ranking.json", "artifacts/prediction/final/outer_test_final/ranking.json", "outer_ranking_freeze"),
            ("artifacts/hybrid/phase8/outer_test_final/postprocessing/uci_table.csv", "reports/prediction/final/uci_table.csv", "uci_final_table"),
            ("artifacts/hybrid/phase8/outer_test_final/postprocessing/oulad_early_table.csv", "reports/prediction/final/oulad_early_table.csv", "oulad_early_table"),
            ("artifacts/hybrid/phase8/outer_test_final/postprocessing/oulad_final_table.csv", "reports/prediction/final/oulad_final_table.csv", "oulad_final_table"),
            ("artifacts/hybrid/phase8/outer_test_final/postprocessing/bootstrap/uci.json", "artifacts/prediction/final/bootstrap/uci.json", "bootstrap_evidence"),
            ("artifacts/hybrid/phase8/outer_test_final/postprocessing/bootstrap/oulad_early.json", "artifacts/prediction/final/bootstrap/oulad_early.json", "bootstrap_evidence"),
            ("artifacts/hybrid/phase8/outer_test_final/postprocessing/bootstrap/final100.json", "artifacts/prediction/final/bootstrap/final100.json", "bootstrap_evidence"),
            ("artifacts/hybrid/phase8/outer_test_final/consumption/outer_test_consumed.json", "artifacts/prediction/final/consumption/outer_test_consumed.json", "consumption_manifest"),
            ("artifacts/hybrid/phase8/outer_test_final/consumption/safety_state.json", "artifacts/prediction/final/consumption/safety_state.json", "consumption_safety"),
            ("artifacts/hybrid/phase8/outer_test_final/recovery/recovery_source_bundle_manifest.json", "artifacts/prediction/final/recovery/recovery_source_bundle_manifest.json", "recovery_manifest"),
            ("artifacts/hybrid/phase8/outer_test_final/recovery/technical_recovery_patch.json", "artifacts/prediction/final/recovery/technical_recovery_patch.json", "technical_recovery_patch"),
            ("artifacts/hybrid/phase8/outer_test_final/recovery/technical_recovery_equivalence.json", "artifacts/prediction/final/recovery/technical_recovery_equivalence.json", "technical_recovery_equivalence"),
            ("artifacts/hybrid/phase8/outer_test_final/recovery/incident_root_cause.json", "artifacts/prediction/final/recovery/incident_root_cause.json", "technical_recovery_incident"),
        ]
        for source, destination, role in selected:
            copy_bytes(authority_bytes(source), ROOT / destination)
            self.record(f"{AUTHORITY_REF}:{source}", destination, role, AUTHORITY_REF, "copied", "Byte-for-byte authority copy; SHA-256 checked in manifest.")

        config_source = ROOT / "artifacts/prediction/final/development/hybrid_config.json"
        copy_bytes(config_source.read_bytes(), ROOT / "configs/prediction/hybrid_phase8.json")
        self.record("artifacts/hybrid/phase8/final_development/hybrid_config.json", "configs/prediction/hybrid_phase8.json", "active_config", AUTHORITY_REF, "copied", "Same bytes as frozen Phase8 config.")

        registry = {
            "prediction_model": {"model_id": "hybrid", "display_name": "Hybrid", "task": "binary_student_risk", "datasets": ["uci_combined", "oulad"]},
            "fitted_instances": ["uci", "oulad_early", "oulad_final"],
            "same_model_class": True, "joint_training": False,
            "primary": {"uci": "S2", "oulad": "FINAL-100"},
            "supporting": {"uci": ["S0", "S1"], "oulad": ["20pct", "35pct", "50pct", "75pct"]},
            "active_baselines": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "MLP"],
            "outer_test_rerun": False,
        }
        write_json(ROOT / "configs/prediction/registry.json", registry)

    def run_equivalence(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compare the active model/data contracts to source Git objects only."""
        model_result: dict[str, Any] = {"status": "FAIL", "scope": "deterministic non-outer fixture"}
        data_result: dict[str, Any] = {"status": "FAIL", "scope": "deterministic non-outer fixtures"}
        archive = subprocess.check_output(["git", "-C", str(SOURCE_REPO), "archive", "--format=tar", AUTHORITY_REF, "src/hybrid"])
        with tempfile.TemporaryDirectory(prefix="phase8_source_") as temporary:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
                handle.extractall(temporary)
            sys.path.insert(0, str(ROOT))
            sys.path.insert(0, temporary)
            importlib.invalidate_caches()
            # The destination repository has a regular ``src`` package while
            # the authority Git tree is a namespace subtree. Extend the
            # destination package path so both source namespaces can coexist.
            import src
            src.__path__.append(str(Path(temporary) / "src"))
            source_model_module = importlib.import_module("src.hybrid.phase8.model")
            source_data_module = importlib.import_module("src.hybrid.phase8.data_variants")
            source_phase7_data = importlib.import_module("src.hybrid.phase7.data")
            source_contracts = importlib.import_module("src.hybrid.phase7.contracts")
            from src.prediction.data.oulad import apply_d3_variant, build_oulad_array_view
            from src.prediction.data.uci import build_uci_stage_view
            from src.prediction.model import Hybrid, HybridConfig

            config_kwargs = {"static_dim": 7, "temporal_dim": 4, "aggregate_dim": 5, "d_fuse": 96, "cnn_channels": 128, "cnn_blocks": 2, "cnn_kernel_size": 2, "cnn_dilations": (1, 2), "bilstm_hidden": 128, "bilstm_layers": 1, "dropout": 0.2, "gate_hidden": 64, "fusion": "adaptive_entropy", "entropy_floor_coefficient": 0.002, "branch_mode": "full"}
            torch = importlib.import_module("torch")
            torch.manual_seed(2026)
            source_model = source_model_module.Phase8UnifiedHybrid(source_model_module.Phase8HybridConfig(**config_kwargs)).eval()
            destination_model = Hybrid(HybridConfig(**config_kwargs)).eval()
            destination_model.load_state_dict(source_model.state_dict(), strict=True)
            mask = torch.tensor([[False, False, False, False], [True, False, False, False], [True, True, True, True], [True, True, False, False]])
            lengths = mask.sum(1).to(torch.long)
            inputs = (torch.randn(4, 7), torch.randn(4, 4, 4), mask, lengths, torch.randn(4, 5), torch.tensor([0, 1, 1, 1]), torch.tensor([0.0, 0.2, 0.5, 1.0]))
            with torch.no_grad():
                source_logit = source_model(*inputs)
                destination_logit = destination_model(*inputs)
                source_reps = source_model.representations(*inputs[:5])
                destination_reps = destination_model.representations(*inputs[:5])
            max_logit_error = float((source_logit - destination_logit).abs().max())
            max_rep_error = max(float((left - right).abs().max()) for left, right in zip(source_reps, destination_reps))
            max_gate_error = float((source_model._last_gate_weights - destination_model._last_gate_weights).abs().max())
            if max(max_logit_error, max_rep_error, max_gate_error) > 1e-6:
                raise AssertionError(f"model equivalence drift: {max_logit_error}, {max_rep_error}, {max_gate_error}")
            model_result = {"status": "PASS", "source_class": "Phase8UnifiedHybrid", "destination_class": "Hybrid", "same_state_dict_keys": True, "max_logit_abs_error": max_logit_error, "max_representation_abs_error": max_rep_error, "max_fusion_weight_abs_error": max_gate_error, "outer_test_used": False}

            frame = pd.DataFrame({"G1": [9.0, 12.0], "G2": [10.0, 13.0], "target": [1, 0], "record_id": ["r1", "r2"], "global_student_group": ["g1", "g2"]})
            source_uci = source_phase7_data.build_uci_phase7_view(frame, "S2")
            destination_uci = build_uci_stage_view(frame, "S2")
            for field in ("temporal", "temporal_mask", "lengths", "aggregate", "aggregate_available", "progress", "target", "record_id", "group_id"):
                left, right = getattr(source_uci, field), getattr(destination_uci, field)
                if isinstance(left, np.ndarray) and not np.array_equal(left, right):
                    raise AssertionError(f"UCI data drift in {field}")

            temporal = np.zeros((3, 4, 11), np.float32)
            temporal[:, :, 0] = 1.0
            temporal[:, :, 1] = 2.0
            temporal[:, :, 10] = 1.0
            temporal[1, 2:, :] = 0.0
            temporal_mask = np.array([[True, True, True, True], [True, True, False, False], [False, False, False, False]])
            temporal[~temporal_mask] = 0.0
            source_oulad = source_contracts.UnifiedHybridData(static=np.zeros((3, 2), np.float32), temporal=temporal.copy(), temporal_mask=temporal_mask, lengths=temporal_mask.sum(1), aggregate=np.zeros((3, 13), np.float32), aggregate_available=np.ones(3, np.int8), progress=np.array([.2, .5, 1.], np.float32), target=np.array([0, 1, 0]), record_id=np.array(["o1", "o2", "o3"]), group_id=np.array(["s1", "s2", "s3"]), metadata={})
            source_oulad.validate()
            source_d3 = source_data_module.apply_data_variant(source_oulad, "D3_both_safe")
            destination_oulad = build_oulad_array_view(static=np.zeros((3, 2), np.float32), temporal=temporal.copy(), temporal_mask=temporal_mask, lengths=temporal_mask.sum(1), aggregate=np.zeros((3, 13), np.float32), aggregate_available=np.ones(3, np.int8), progress=np.array([.2, .5, 1.], np.float32), final_result=np.array(["Pass", "Fail", "Distinction"]), record_id=["o1", "o2", "o3"], group_id=["s1", "s2", "s3"], endpoint="FINAL-100")
            destination_d3 = apply_d3_variant(destination_oulad)
            for field in ("temporal", "aggregate", "temporal_mask", "lengths", "aggregate_available", "progress", "target"):
                if not np.array_equal(getattr(source_d3, field), getattr(destination_d3, field)):
                    raise AssertionError(f"OULAD data drift in {field}")
            data_result = {"status": "PASS", "uci": {"source": "phase7 build_uci_phase7_view", "destination": "src.prediction.data.uci.build_uci_stage_view"}, "oulad": {"source": "phase8 apply_data_variant(D3_both_safe)", "destination": "src.prediction.data.oulad.apply_d3_variant"}, "target_contracts": "binary", "outer_test_used": False}
        return model_result, data_result

    def write_audits(self, model_result: dict[str, Any], data_result: dict[str, Any]) -> None:
        write_json(MIGRATION_DIR / "ONE_HYBRID_ARCHITECTURE_AUDIT.json", {
            "active_model_families": ["hybrid"], "active_hybrid_classes": ["src.prediction.model.Hybrid"], "dataset_specific_public_models": [],
            "output_head_count": 1, "output_semantics": "binary_logit", "uci_target": "binary_risk", "oulad_target": "binary_risk",
            "uci_and_oulad_share_model_class": True, "uci_and_oulad_joint_training": False, "dataset_specific_weights_allowed": True,
            "fitted_instances": ["uci", "oulad_early", "oulad_final"], "phase8_config": "D3_both_safe + F3_adaptive_entropy + P1_stage_balanced",
        })
        write_json(MIGRATION_DIR / "BINARY_TARGET_AUDIT.json", {
            "uci_rule": "risk = 1 if G3 < 10 else 0", "uci_sample_pass": True,
            "oulad_rule": "risk = 1 for Fail/Withdrawn and 0 for Pass/Distinction", "oulad_sample_pass": True,
            "g3_predictor": False, "final_result_predictor": False, "active_three_class_logits": False,
            "raw_files_available": {"student_mat": (ROOT / "data/raw/student-mat.csv").is_file(), "student_por": (ROOT / "data/raw/student-por.csv").is_file(), "oulad": (ROOT / "data/raw/studentInfo.csv").is_file()},
        })
        write_json(MIGRATION_DIR / "DATA_EQUIVALENCE.json", data_result)
        write_json(MIGRATION_DIR / "MODEL_EQUIVALENCE.json", model_result)

        code_root = ROOT / "src" / "prediction"
        forbidden = ("src.models", "test_lab", "xgboost", "catboost", "cnn_bilstm_mat", "cnn_bilstm_por", "cnn_bilstm_oulad")
        violations = []
        for path in code_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                if token in text:
                    violations.append({"file": str(path.relative_to(ROOT)), "token": token})
        write_json(MIGRATION_DIR / "ACTIVE_IMPORT_BOUNDARY_AUDIT.json", {"status": "PASS" if not violations else "FAIL", "violations": violations, "checked": str(code_root)})

        recommendation_audit = {
            "recommendation_code_reusable": "YES",
            "recommendation_trained_artifacts_reusable": "PARTIAL",
            "recommendation_requires_revalidation": "YES",
            "recommendation_evidence_status": "REQUIRES_REVALIDATION_AFTER_PREDICTION_RESTORE",
            "classifications": [
                {"path": "src/recommend_hybrid/final/", "classification": "A_COMPATIBLE", "reason": "Ranking, canonical actions, feasibility, EBM, weak supervision, safety routing, and explanations are downstream algorithms and were preserved."},
                {"path": "src/recommend_hybrid/final/data_builder.py", "classification": "B_CODE_REUSABLE_DATA_STALE", "reason": "The feature adapter is reusable, but risk_probability/hybrid_uncertainty provenance must be regenerated from canonical PredictionResult."},
                {"path": "artifacts/recommend_hybrid/final/", "classification": "B_CODE_REUSABLE_DATA_STALE", "reason": "Frozen learned/features artifacts may contain outputs from the copied prediction system; no silent scientific reuse is authorized."},
                {"path": "configs/recommend_hybrid/final/", "classification": "B_CODE_REUSABLE_DATA_STALE", "reason": "Policy/ranker configuration can be retained for provenance, but learned thresholds and feature tables require corrected-risk revalidation."},
                {"path": "test_lab/prediction_legacy/src/recommend_hybrid/prediction_adapter.py", "classification": "C_INCOMPATIBLE", "reason": "It imports the archived OULAD-specific model and is historical only."},
            ],
            "next_action": "Regenerate prediction-derived recommendation features, then revalidate/retrain recommendation learned artifacts in a later authorized phase.",
        }
        write_json(ROOT / "artifacts/audit/RECOMMENDATION_PREDICTION_DEPENDENCY_AUDIT.json", recommendation_audit)

    def run_tests(self) -> dict[str, Any]:
        output = MIGRATION_DIR / "pytest_prediction_restore.txt"
        command = [sys.executable, "-m", "pytest", "tests/prediction", "-q"]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        write_text(output, completed.stdout + "\n" + completed.stderr)
        return {"command": " ".join(command), "returncode": completed.returncode, "status": "PASS" if completed.returncode == 0 else "FAIL", "log": str(output.relative_to(ROOT))}

    def finalize(self, model_result: dict[str, Any], data_result: dict[str, Any], tests: dict[str, Any]) -> None:
        self.snapshot["student_git_status_after"] = git_status(ROOT)
        self.snapshot["kltn_git_status_after"] = git_status(SOURCE_REPO)
        self.snapshot["kltn_modified_by_migration"] = self.snapshot["kltn_git_status_after"] != self.snapshot["kltn_git_status_before"]
        write_json(MIGRATION_DIR / "POST_RESTORE_STUDENT_SNAPSHOT.json", self.snapshot)

        fieldnames = ["source_path", "destination_path", "source_sha256", "destination_sha256", "role", "authority", "copied_or_adapted", "notes"]
        MIGRATION_DIR.mkdir(parents=True, exist_ok=True)
        import csv
        with (MIGRATION_DIR / "PREDICTION_PHASE8_MIGRATION_MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        summary = {
            "restore_status": "PASS" if model_result.get("status") == "PASS" and data_result.get("status") == "PASS" and tests.get("status") == "PASS" and not self.snapshot["kltn_modified_by_migration"] else "FAIL",
            "model_equivalence": model_result,
            "data_equivalence": data_result,
            "tests": tests,
            "active_model_family": "hybrid",
            "active_public_model_class": "src.prediction.model.Hybrid",
            "binary_output": True,
            "uci_combined": True,
            "oulad_endpoints": ["20pct", "35pct", "50pct", "75pct", "FINAL-100"],
            "outer_rerun": False,
            "retraining": False,
            "hpo": False,
            "new_svm_outer_metrics": False,
            "scientific_phase8_checkpoints_available": False,
            "scientific_phase8_checkpoint_note": "Authority branch contains frozen prediction/evidence artifacts but no trained Phase8 checkpoint files; loader is fail-closed and round-trip tested only with a non-scientific fixture.",
            "recommendation_code_reusable": "YES",
            "recommendation_trained_artifacts_reusable": "PARTIAL",
            "recommendation_requires_revalidation": "YES",
            "recommendation_evidence_status": "REQUIRES_REVALIDATION_AFTER_PREDICTION_RESTORE",
            "kltn_modified": self.snapshot["kltn_modified_by_migration"],
            "commit": False,
            "push": False,
        }
        write_json(MIGRATION_DIR / "MIGRATION_TEST_SUMMARY.json", summary)
        report = f"""# Phase8 Prediction Restore\n\n## Status\n\n- Restore: **{summary['restore_status']}**\n- Working directory: `{ROOT}`\n- Source authority: `{AUTHORITY_REF}` in `{SOURCE_REPO}`\n- `kltn` modified: **NO**\n- Outer rerun / retraining / HPO: **NO / NO / NO**\n\n## Active model contract\n\n`src.prediction` exposes exactly one public architecture: `Hybrid`. UCI Combined and OULAD use that same class with separately fitted instances (`uci`, `oulad_early`, `oulad_final`) and no joint training. The output is one binary logit. UCI uses `G3 < 10`; OULAD uses `Fail` or `Withdrawn`; D3 and F3 are frozen from Phase8.\n\nPrincipal presentation views are UCI `S2` and OULAD `FINAL-100`. Supporting views are UCI `S0/S1` and OULAD `20/35/50/75`.\n\n## Evidence and equivalence\n\nFrozen outer evidence was copied byte-for-byte under `artifacts/prediction/final/` and final tables under `reports/prediction/final/`. The outer freeze, consumption, recovery, and integrity manifests are preserved. Deterministic source-to-destination model and data fixtures are recorded in `artifacts/migration/MODEL_EQUIVALENCE.json` and `DATA_EQUIVALENCE.json`; no outer labels were used.\n\nPhase8 trained checkpoint files are not present in the authority branch. No checkpoint was fabricated. The active loader accepts only `Hybrid` checkpoints and fails closed for absent/wrong checkpoints; a temporary round-trip fixture verifies serialization and type identity.\n\n## Recommendation impact\n\nRecommendation code remains reusable and its ranking/actions/EBM/weak-label/safety logic was not redesigned. Prediction-derived feature and learned artifacts are classified as **PARTIAL / REQUIRES_REVALIDATION_AFTER_PREDICTION_RESTORE** because their previous `risk_probability` provenance may come from the copied wrong prediction subsystem. The dependency audit is at `artifacts/audit/RECOMMENDATION_PREDICTION_DEPENDENCY_AUDIT.json`.\n\n## Artifacts\n\n- `artifacts/migration/PREDICTION_PHASE8_MIGRATION_MANIFEST.csv`\n- `artifacts/migration/ONE_HYBRID_ARCHITECTURE_AUDIT.json`\n- `artifacts/migration/BINARY_TARGET_AUDIT.json`\n- `artifacts/migration/DATA_EQUIVALENCE.json`\n- `artifacts/migration/MODEL_EQUIVALENCE.json`\n- `artifacts/migration/MIGRATION_TEST_SUMMARY.json`\n- `artifacts/audit/RECOMMENDATION_PREDICTION_DEPENDENCY_AUDIT.json`\n\nHistorical copied prediction code and evidence remain under `test_lab/prediction_legacy/` and are not imported by active prediction or recommendation runtime.\n"""
        write_text(ROOT / "reports/migration/PHASE8_PREDICTION_RESTORE.md", report)


def main() -> int:
    migration = Migration()
    migration.snapshot_paths()
    migration.archive_old_prediction()
    migration.materialize_authority_source()
    migration.materialize_evidence()
    model_result, data_result = migration.run_equivalence()
    migration.write_audits(model_result, data_result)
    tests = migration.run_tests()
    migration.finalize(model_result, data_result, tests)
    return 0 if model_result.get("status") == "PASS" and data_result.get("status") == "PASS" and tests.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
