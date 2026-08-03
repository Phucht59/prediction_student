"""Structural validation for the frozen OULAD prediction checkpoint authority.

This module deliberately validates the public LFS release independently from the
recommendation manifest.  A checkpoint is never accepted merely because it can
be deserialized: the model topology, tensor schema, preprocessing state,
stage namespace, and registered checksums must agree with the authority.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml

from .prediction_adapter import _preprocessor_sha256

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
REQUIRED_STAGES = (
    "E1_EARLY_20PCT",
    "E2_EARLY_35PCT",
    "M1_MIDDLE_FROZEN",
    "L1_LATE_75PCT",
)
MODEL_CLASS = "src.models.oulad_tabular_residual.CNNBiLSTMTabularResidualOULAD"
RELEASE_MODEL_CLASS = "src.models.oulad_multitask.CNNBiLSTMOULAD"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_fingerprint(state_dict: Mapping[str, Any]) -> str:
    """Fingerprint names, shapes, and dtypes, excluding parameter values."""

    rows = []
    for name, value in sorted(state_dict.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        rows.append(
            {
                "name": str(name),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _gate(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(160)
    except OSError:
        return False
    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1")


def resolve_checkpoint_path(
    local_path: Path,
    release_path: Path,
    *,
    expected_sha256: str,
    structural_validator: Callable[[Path], bool],
) -> dict[str, Any]:
    """Resolve an authority path without silently substituting a checkpoint.

    The release fallback is permitted only when the declared local path is
    absent.  A present-but-invalid local checkpoint is an authority failure,
    not a reason to try another file.
    """

    if local_path.exists():
        source = "local_authority"
        candidate = local_path
    else:
        source = "release_lfs_fallback"
        candidate = release_path
    if not candidate.is_file() or _is_lfs_pointer(candidate):
        raise FileNotFoundError(f"checkpoint is missing or an LFS pointer: {candidate}")
    actual = sha256_file(candidate)
    if actual != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch for {candidate}: "
            f"expected={expected_sha256} actual={actual}"
        )
    if not structural_validator(candidate):
        raise ValueError(f"checkpoint structural authority mismatch: {candidate}")
    return {
        "checkpoint_path": str(candidate),
        "resolved_checkpoint_source": source,
        "sha256": actual,
    }


def _load_class(path: str) -> type[torch.nn.Module]:
    module_name, class_name = path.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(value, type) or not issubclass(value, torch.nn.Module):
        raise TypeError(f"{path} is not a torch.nn.Module class")
    return value


def _instantiate_model(
    model_class: str, payload: Mapping[str, Any]
) -> torch.nn.Module:
    cls = _load_class(model_class)
    return cls(
        int(payload["sequence_dim"])
        if "sequence_dim" in payload
        else 47,
        int(payload["aggregate_dim"]),
        int(payload["static_dim"]),
        dict(payload["config"]),
    )


def _payload_dimensions(payload: Mapping[str, Any]) -> dict[str, int]:
    state = payload["state_dict"]
    sequence = state["backbone.temporal.input_projection.weight"]
    aggregate = state["backbone.aggregate.network.0.weight"]
    static = state["backbone.static.network.0.weight"]
    risk = state["backbone.head.4.weight"]
    return {
        "sequence": int(sequence.shape[1]),
        "aggregate": int(aggregate.shape[1]),
        "static": int(static.shape[1]),
        "risk_output": int(risk.shape[0]),
    }


def _validate_preprocessor(preprocessor: Any) -> tuple[bool, str | None, dict[str, Any]]:
    if not isinstance(preprocessor, Mapping):
        return False, None, {}
    try:
        mean = np.asarray(preprocessor["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(preprocessor["scale"], dtype=np.float64).reshape(-1)
        num_mean = np.asarray(preprocessor["num_mean"], dtype=np.float64).reshape(-1)
        num_scale = np.asarray(preprocessor["num_scale"], dtype=np.float64).reshape(-1)
        num_cols = tuple(str(value) for value in preprocessor["num_cols"])
        categories = {
            str(key): tuple(str(value) for value in values)
            for key, values in dict(preprocessor["categories"]).items()
        }
    except (KeyError, TypeError, ValueError):
        return False, None, {}
    valid = (
        mean.shape == (165,)
        and scale.shape == (165,)
        and num_mean.shape == num_scale.shape == (4,)
        and len(num_cols) == 4
        and len(set(num_cols)) == 4
        and bool(categories)
        and sum(len(values) for values in categories.values()) + 4 == 13
        and np.isfinite(mean).all()
        and np.isfinite(scale).all()
        and np.isfinite(num_mean).all()
        and np.isfinite(num_scale).all()
        and np.all(scale > 0)
        and np.all(num_scale > 0)
    )
    if not valid:
        return False, None, {}
    return (
        True,
        _preprocessor_sha256(preprocessor),
        {
            "aggregate_dimension": int(mean.size),
            "static_dimension": int(num_mean.size + sum(map(len, categories.values()))),
            "numeric_columns": list(num_cols),
            "categorical_columns": sorted(categories),
        },
    )


def _release_rows(root: Path, mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in mapping.get("rows", [])
        if row.get("model_id") == "cnn_bilstm_oulad"
        and row.get("prediction_stage") in REQUIRED_STAGES
    ]


def validate_checkpoint_authority(
    root: Path = ROOT_DEFAULT,
    *,
    release_mapping_path: Path = Path(
        "artifacts/final/unified_stage_aware_oulad/checkpoint_stage_mapping.json"
    ),
    recommendation_manifest_path: Path = Path(
        "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
    ),
) -> dict[str, Any]:
    """Validate the release checkpoint set and compare it to recommendation authority."""

    mapping = json.loads((root / release_mapping_path).read_text(encoding="utf-8"))
    recommendation = json.loads(
        (root / recommendation_manifest_path).read_text(encoding="utf-8")
    )
    authority = yaml.safe_load(
        (root / "configs/recommend_hybrid/model_authority.yaml").read_text(
            encoding="utf-8"
        )
    )
    rows = _release_rows(root, mapping)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["checkpoint"]), row)
    gates: list[dict[str, Any]] = []
    gates.append(_gate("stage_mapping_namespace", set(REQUIRED_STAGES) == {
        str(row["prediction_stage"]) for row in rows
    }, f"found={sorted({str(row['prediction_stage']) for row in rows})}"))
    gates.append(_gate("release_rows_complete", len(rows) == 60, f"found={len(rows)}"))

    checkpoint_records: list[dict[str, Any]] = []
    architecture_fingerprints: dict[int, set[str]] = {}
    preprocessor_fingerprints: dict[int, set[str]] = {}
    for source, row in sorted(unique.items()):
        path = root / source
        checks: list[dict[str, Any]] = []
        exists = path.is_file()
        checks.append(_gate("file_exists", exists, str(path)))
        checks.append(_gate("not_lfs_pointer", exists and not _is_lfs_pointer(path)))
        actual_sha = sha256_file(path) if exists and not _is_lfs_pointer(path) else None
        checks.append(_gate("sha256_matches_release_mapping", actual_sha == row.get("checkpoint_sha256")))
        payload: Mapping[str, Any] | None = None
        load_error = ""
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(loaded, Mapping):
                raise TypeError("payload is not a mapping")
            payload = loaded
        except Exception as exc:  # checkpoint authority must report all gates
            load_error = f"{type(exc).__name__}: {exc}"
        checks.append(_gate("torch_load_cpu", payload is not None, load_error))

        model_class = None
        dimensions: dict[str, int] = {}
        state_fingerprint = None
        preprocessor_hash = None
        parameter_count = None
        strict_error = ""
        if payload is not None:
            model_class = RELEASE_MODEL_CLASS if payload.get("model_id") == "cnn_bilstm_oulad" else None
            try:
                dimensions = _payload_dimensions(payload)
                state_fingerprint = state_dict_fingerprint(payload["state_dict"])
                parameter_count = int(sum(value.numel() for value in payload["state_dict"].values()))
            except Exception as exc:
                strict_error = f"structure: {type(exc).__name__}: {exc}"
            pre_ok, preprocessor_hash, preprocessor_meta = _validate_preprocessor(
                payload.get("preprocessor")
            )
            checks.append(_gate("frozen_preprocessor_valid", pre_ok))
            release_strict_error = ""
            try:
                release_model = _instantiate_model(RELEASE_MODEL_CLASS, payload)
                release_model.load_state_dict(payload["state_dict"], strict=True)
            except Exception as exc:
                release_strict_error = f"{type(exc).__name__}: {exc}"
            checks.append(
                _gate("release_model_strict_load", not release_strict_error, release_strict_error)
            )
            try:
                model = _instantiate_model(MODEL_CLASS, payload)
                model.load_state_dict(payload["state_dict"], strict=True)
                strict_error = ""
            except Exception as exc:
                strict_error = f"{type(exc).__name__}: {exc}"
            checks.append(_gate("authority_model_strict_load", not strict_error, strict_error))
            checks.append(_gate("release_model_class", model_class == RELEASE_MODEL_CLASS, str(payload.get("model_id"))))
            checks.append(_gate("sequence_dimension", dimensions.get("sequence") == 47, str(dimensions.get("sequence"))))
            checks.append(_gate("aggregate_dimension", dimensions.get("aggregate") == 165, str(dimensions.get("aggregate"))))
            checks.append(_gate("static_dimension", dimensions.get("static") == 13, str(dimensions.get("static"))))
            checks.append(_gate("risk_output_dimension", dimensions.get("risk_output") == 1, str(dimensions.get("risk_output"))))
            checks.append(_gate("parameter_count", parameter_count == int(authority["parameter_count"]), f"actual={parameter_count} expected={authority['parameter_count']}"))
            if state_fingerprint is not None:
                architecture_fingerprints.setdefault(int(row["outer_fold"]), set()).add(state_fingerprint)
            if preprocessor_hash is not None:
                preprocessor_fingerprints.setdefault(int(row["outer_fold"]), set()).add(preprocessor_hash)
        checkpoint_records.append({
            "checkpoint_path": source,
            "sha256": actual_sha,
            "registered_sha256": row.get("checkpoint_sha256"),
            "fold": int(row["outer_fold"]),
            "seed": int(row["seed"]),
            "stage": row.get("prediction_stage"),
            "model_class": model_class,
            "dimensions": dimensions,
            "parameter_count": parameter_count,
            "state_dict_fingerprint": state_fingerprint,
            "preprocessor_fingerprint": preprocessor_hash,
            "gates": checks,
        })
        gates.extend(
            {**check, "checkpoint_path": source}
            for check in checks
        )

    for fold in sorted(architecture_fingerprints):
        values = architecture_fingerprints[fold]
        gates.append(_gate("same_fold_architecture_fingerprint", len(values) == 1, f"fold={fold} unique={len(values)}"))
    for fold in sorted(preprocessor_fingerprints):
        values = preprocessor_fingerprints[fold]
        gates.append(_gate("same_fold_preprocessor_fingerprint", len(values) == 1, f"fold={fold} unique={len(values)}"))
    release_classes = {row["model_class"] for row in checkpoint_records}
    release_parameter_counts = {row["parameter_count"] for row in checkpoint_records}
    authority_matches_release = (
        str(recommendation.get("architecture_hash")) == str(authority.get("architecture_hash"))
        and int(recommendation.get("parameter_count", -1)) == int(authority["parameter_count"])
        and release_classes == {MODEL_CLASS}
        and release_parameter_counts == {int(authority["parameter_count"])}
    )
    gates.append(_gate(
        "recommendation_authority_matches_release",
        authority_matches_release,
        "release_classes="
        f"{sorted(release_classes)} release_parameter_counts={sorted(release_parameter_counts)} "
        f"expected_class={MODEL_CLASS} expected_parameter_count={authority['parameter_count']}",
    ))
    failed = [gate for gate in gates if gate["status"] != "PASS"]
    return {
        "schema_version": "checkpoint_authority_validation_v1",
        "status": "PASS" if not failed else "FAIL",
        "claim_boundary": "TECHNICAL_AUTHORITY_VALIDATION_ONLY_NOT_CAUSAL_EFFECT",
        "authority_model_class": MODEL_CLASS,
        "release_model_class": RELEASE_MODEL_CLASS,
        "authority_architecture_hash": authority.get("architecture_hash"),
        "authority_parameter_count": int(authority["parameter_count"]),
        "release_mapping": str(release_mapping_path),
        "recommendation_manifest": str(recommendation_manifest_path),
        "stage_mapping": list(REQUIRED_STAGES),
        "gates": gates,
        "checkpoints": checkpoint_records,
        "failed_gate_count": len(failed),
    }


__all__ = [
    "MODEL_CLASS",
    "RELEASE_MODEL_CLASS",
    "REQUIRED_STAGES",
    "resolve_checkpoint_path",
    "sha256_file",
    "state_dict_fingerprint",
    "validate_checkpoint_authority",
]
