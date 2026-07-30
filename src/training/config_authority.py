"""Authoritative configuration loading and architecture fingerprinting."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import yaml
from torch import nn

from src.training.control import stable_hash


def load_config_authority(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "config_version",
        "protocol_id",
        "stage_policy_version",
        "authority_scope",
        "dataset_schema",
        "architecture",
        "training",
        "loss",
        "thresholds",
        "pretraining",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"configuration authority missing fields: {sorted(missing)}")
    return value


def resolved_deep_config(authority: dict[str, Any]) -> dict[str, Any]:
    """Flatten only model/training fields consumed by the OULAD trainer."""
    return {**authority["architecture"], **authority["training"], **authority["loss"]}


def architecture_metadata(
    model: nn.Module,
    *,
    authority: dict[str, Any],
    aggregate_dim: int,
    static_dim: int,
) -> dict[str, Any]:
    parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    identity = {
        "config_version": authority["config_version"],
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "architecture": authority["architecture"],
        "sequence_channels": int(authority["architecture"]["sequence_channels"]),
        "aggregate_dim": int(aggregate_dim),
        "static_dim": int(static_dim),
        "representation_dim": int(getattr(model, "representation_dim", 0)),
        "auxiliary_heads": list(authority["architecture"]["auxiliary_heads"]),
        "parameter_count": parameter_count,
        "loss": authority["loss"],
        "pretraining": authority["pretraining"],
    }
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "UNKNOWN"
    return {
        **identity,
        "dataset_schema": authority["dataset_schema"],
        "source_commit": source_commit,
        "config_hash": stable_hash(
            {
                "config_version": authority["config_version"],
                "dataset_schema": authority["dataset_schema"],
                "architecture": authority["architecture"],
                "training": authority["training"],
                "loss": authority["loss"],
                "thresholds": authority["thresholds"],
                "pretraining": authority["pretraining"],
            }
        ),
        "architecture_hash": stable_hash(identity),
    }
