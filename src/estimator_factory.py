"""Canonical estimator configuration and construction for Strategy B.

The factory is deliberately shared by inner-CV, outer-fold evaluation and
full-development final refit.  It rejects incomplete configurations instead
of silently changing the loss, resampling or training estimator.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import STUDENT_G3_3CLASS_BINS
from src.data_pipeline import DataPreprocessor, FeatureSelector, get_sequence_columns
from src.models import FocalLoss, create_model


RESOLVED_CONFIG_SCHEMA_VERSION = "strategy_b_resolved_config_v1"
SUPPORTED_SCHEDULERS = {"fixed_lr", "legacy_reduce_on_plateau"}
SUPPORTED_LOSSES = {"cross_entropy", "focal"}
SUPPORTED_CLASS_WEIGHT_MODES = {"none", "balanced"}
SUPPORTED_OVERSAMPLING = {"none", "random", "smote", "smotenc"}

REQUIRED_RESOLVED_CONFIG_KEYS = {
    "schema_version",
    "architecture_variant",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "oversample_method",
    "class_weight_mode",
    "loss",
    "smote_ratio",
    "resampling_k_neighbors",
    "cnn_channels",
    "cnn_kernel_size",
    "lstm_hidden_dim",
    "dropout",
    "sequence_dropout",
    "max_epochs",
    "patience",
    "scheduler",
    "swa",
    "drop_last_train",
    "preprocessing",
    "feature_contract",
    "target_contract",
    "suggested_parameters",
    "fixed_constants",
}


class ResolvedConfigError(ValueError):
    """Raised when an estimator configuration is incomplete or contradictory."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def resolved_config_hash(config: Mapping[str, Any]) -> str:
    validate_resolved_config(config)
    return hashlib.sha256(canonical_json(dict(config)).encode("utf-8")).hexdigest()


def resolved_config_schema() -> dict[str, Any]:
    """Return the machine-readable schema contract written into every run."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": RESOLVED_CONFIG_SCHEMA_VERSION,
        "title": "Strategy B canonical resolved estimator configuration",
        "type": "object",
        "required": sorted(REQUIRED_RESOLVED_CONFIG_KEYS),
        "additionalProperties": True,
        "properties": {
            "schema_version": {"const": RESOLVED_CONFIG_SCHEMA_VERSION},
            "architecture_variant": {"type": "string"},
            "learning_rate": {"type": "number", "exclusiveMinimum": 0},
            "weight_decay": {"type": "number", "minimum": 0},
            "batch_size": {"type": "integer", "minimum": 1},
            "oversample_method": {"enum": sorted(SUPPORTED_OVERSAMPLING)},
            "class_weight_mode": {"enum": sorted(SUPPORTED_CLASS_WEIGHT_MODES)},
            "loss": {"enum": sorted(SUPPORTED_LOSSES)},
            "scheduler": {
                "type": "object",
                "required": ["type", "parameters", "replayable"],
            },
            "swa": {
                "type": "object",
                "required": ["enabled", "batch_norm_statistics_updated", "replayable"],
            },
            "drop_last_train": {"type": "boolean"},
            "preprocessing": {"type": "object"},
            "feature_contract": {"type": "object"},
            "target_contract": {"type": "object"},
            "suggested_parameters": {"type": "object"},
            "fixed_constants": {"type": "object"},
        },
    }


def _normalized_loss(value: Any) -> str:
    name = str(value)
    if name in {"weighted_ce", "ce", "cross_entropy"}:
        return "cross_entropy"
    if name == "focal":
        return "focal"
    raise ResolvedConfigError(f"Unsupported loss: {name!r}")


def resolve_student_config(
    parameters: Mapping[str, Any],
    *,
    architecture_variant: str = "cnn_bilstm",
    suggested_parameters: Mapping[str, Any] | None = None,
    scheduler_type: str = "fixed_lr",
    swa_enabled: bool = False,
    drop_last_train: bool = False,
    evidence_role: str = "phase_b_corrected",
) -> dict[str, Any]:
    """Resolve a complete student estimator configuration.

    ``parameters`` may be an Optuna parameter dictionary or a historical flat
    configuration.  Every training-affecting default is made explicit in the
    returned object.  Downstream fit functions accept only this resolved form.
    """

    values = dict(parameters)
    if scheduler_type not in SUPPORTED_SCHEDULERS:
        raise ResolvedConfigError(f"Unsupported scheduler type: {scheduler_type!r}")
    if scheduler_type == "legacy_reduce_on_plateau":
        scheduler = {
            "type": scheduler_type,
            "parameters": {
                "factor": float(values.get("scheduler_factor", 0.5)),
                "patience": int(values.get("scheduler_patience", 5)),
            },
            "replayable": False,
        }
    else:
        scheduler = {"type": "fixed_lr", "parameters": {}, "replayable": True}

    swa = {
        "enabled": bool(swa_enabled),
        "batch_norm_statistics_updated": False,
        "replayable": False if swa_enabled else True,
    }
    resolved = {
        "schema_version": RESOLVED_CONFIG_SCHEMA_VERSION,
        "architecture_variant": str(values.get("architecture_variant", architecture_variant)),
        "architecture": "cnn_bilstm_classifier",
        "context_mlp_enabled": False,
        "classifier_head": "linear",
        "learning_rate": float(values["learning_rate"]),
        "weight_decay": float(values["weight_decay"]),
        "batch_size": int(values["batch_size"]),
        "oversample_method": str(values.get("oversample_method", "none")).lower(),
        "class_weight_mode": str(values.get("class_weight_mode", "none")).lower(),
        "loss": _normalized_loss(values.get("loss", "cross_entropy")),
        "smote_ratio": float(values.get("smote_ratio", 1.0)),
        "resampling_k_neighbors": int(values.get("resampling_k_neighbors", 5)),
        "cnn_channels": int(values["cnn_channels"]),
        "cnn_kernel_size": int(values["cnn_kernel_size"]),
        "lstm_hidden_dim": int(values["lstm_hidden_dim"]),
        "dropout": float(values["dropout"]),
        "sequence_dropout": float(values["sequence_dropout"]),
        "max_epochs": int(values["max_epochs"]),
        "patience": int(values["patience"]),
        "scheduler": scheduler,
        "swa": swa,
        "drop_last_train": bool(drop_last_train),
        "preprocessing": {
            "scaler": "minmax",
            "fit_scope": "gradient_training_partition_then_full_refit_partition",
            "feature_selection": "train_partition_only",
            "oversampling_scope": "gradient_training_partition_only",
            "unknown_category_policy": "map_to_zero",
        },
        "feature_contract": {
            "scenario": "late_stage",
            "sequence_columns": ["G1", "G2"],
            "context_columns": [],
            "target_or_derived_features_forbidden": True,
        },
        "target_contract": {
            "target_column": "G3",
            "target_mode": "3class",
            "bins": list(STUDENT_G3_3CLASS_BINS),
            "labels": ["Low", "Medium", "High"],
        },
        "suggested_parameters": dict(suggested_parameters or {}),
        "fixed_constants": {
            "architecture_variant": str(values.get("architecture_variant", architecture_variant)),
            "scheduler": deepcopy(scheduler),
            "swa": deepcopy(swa),
            "drop_last_train": bool(drop_last_train),
            "preprocessing": "fold_train_only",
            "feature_contract": "late_stage_g1_g2",
            "target_contract": "student_g3_3class_v1",
        },
        "evidence_role": str(evidence_role),
    }
    if resolved["loss"] == "focal":
        resolved["focal_gamma"] = float(values["focal_gamma"])
    validate_resolved_config(resolved)
    return resolved


def with_training_policy(
    config: Mapping[str, Any],
    *,
    scheduler_type: str,
    swa_enabled: bool,
    drop_last_train: bool,
    evidence_role: str,
) -> dict[str, Any]:
    """Return a newly validated config with only the declared policy changed."""

    validate_resolved_config(config)
    flat = {
        key: deepcopy(value)
        for key, value in dict(config).items()
        if key not in {
            "schema_version", "scheduler", "swa", "drop_last_train",
            "preprocessing", "feature_contract", "target_contract",
            "suggested_parameters", "fixed_constants", "evidence_role",
        }
    }
    scheduler_parameters = dict(config["scheduler"].get("parameters", {}))
    flat["scheduler_factor"] = scheduler_parameters.get("factor", 0.5)
    flat["scheduler_patience"] = scheduler_parameters.get("patience", 5)
    return resolve_student_config(
        flat,
        architecture_variant=str(config["architecture_variant"]),
        suggested_parameters=dict(config["suggested_parameters"]),
        scheduler_type=scheduler_type,
        swa_enabled=swa_enabled,
        drop_last_train=drop_last_train,
        evidence_role=evidence_role,
    )


def validate_resolved_config(config: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_RESOLVED_CONFIG_KEYS - set(config))
    if missing:
        raise ResolvedConfigError(f"Resolved config is missing required keys: {missing}")
    if config["schema_version"] != RESOLVED_CONFIG_SCHEMA_VERSION:
        raise ResolvedConfigError("Unknown resolved config schema_version.")
    if str(config["loss"]) not in SUPPORTED_LOSSES:
        raise ResolvedConfigError("Resolved config has an unsupported loss.")
    if str(config["class_weight_mode"]) not in SUPPORTED_CLASS_WEIGHT_MODES:
        raise ResolvedConfigError("Resolved config has an unsupported class_weight_mode.")
    if str(config["oversample_method"]) not in SUPPORTED_OVERSAMPLING:
        raise ResolvedConfigError("Resolved config has an unsupported oversample_method.")
    if float(config["learning_rate"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ResolvedConfigError("Learning rate/weight decay are outside their valid domains.")
    for key in ("batch_size", "cnn_channels", "cnn_kernel_size", "lstm_hidden_dim", "max_epochs", "patience"):
        if int(config[key]) < 1:
            raise ResolvedConfigError(f"{key} must be positive.")
    scheduler = config["scheduler"]
    if not isinstance(scheduler, Mapping) or set(("type", "parameters", "replayable")) - set(scheduler):
        raise ResolvedConfigError("scheduler must contain type, parameters and replayable.")
    if scheduler["type"] not in SUPPORTED_SCHEDULERS:
        raise ResolvedConfigError("Unsupported scheduler policy.")
    swa = config["swa"]
    if not isinstance(swa, Mapping) or set(("enabled", "batch_norm_statistics_updated", "replayable")) - set(swa):
        raise ResolvedConfigError("swa must contain enabled, batch_norm_statistics_updated and replayable.")
    if scheduler["type"] == "fixed_lr" and bool(swa["enabled"]):
        raise ResolvedConfigError("Strategy B fixed-LR policy requires SWA disabled.")
    if config["feature_contract"].get("sequence_columns") != ["G1", "G2"]:
        raise ResolvedConfigError("Primary feature contract must be exactly G1/G2.")
    if config["feature_contract"].get("context_columns") != []:
        raise ResolvedConfigError("Context feature track is closed for Strategy B Phase A-B.")
    if config["target_contract"].get("bins") != list(STUDENT_G3_3CLASS_BINS):
        raise ResolvedConfigError("Target contract bins do not match the frozen project contract.")
    if not isinstance(config["drop_last_train"], bool):
        raise ResolvedConfigError("drop_last_train must be a boolean.")
    if not isinstance(config["suggested_parameters"], Mapping) or not isinstance(config["fixed_constants"], Mapping):
        raise ResolvedConfigError("suggested_parameters and fixed_constants must be objects.")
    if config["loss"] == "focal" and "focal_gamma" not in config:
        raise ResolvedConfigError("Focal loss requires focal_gamma.")


class StudentEstimatorFactory:
    """Construct every training component from one validated configuration."""

    def __init__(self, spec: Any, resolved_config: Mapping[str, Any]):
        validate_resolved_config(resolved_config)
        expected = get_sequence_columns(spec.kind)
        if list(resolved_config["feature_contract"]["sequence_columns"]) != expected:
            raise ResolvedConfigError("Dataset and resolved feature contracts disagree.")
        if str(resolved_config["target_contract"]["target_column"]) != str(spec.target_col):
            raise ResolvedConfigError("Dataset and resolved target contracts disagree.")
        self.spec = spec
        self.config = deepcopy(dict(resolved_config))

    @property
    def config_hash(self) -> str:
        return resolved_config_hash(self.config)

    def create_preprocessor(self) -> DataPreprocessor:
        return DataPreprocessor(
            target_col=self.spec.target_col,
            oversample_method=str(self.config["oversample_method"]),
            smote_ratio=float(self.config["smote_ratio"]),
            resampling_k_neighbors=int(self.config["resampling_k_neighbors"]),
            oversampling_feature_columns=list(self.config["feature_contract"]["sequence_columns"]),
        )

    def create_selector(self) -> FeatureSelector:
        return FeatureSelector(
            target_col=self.spec.target_col,
            use_feature_selection=True,
            required_features=list(self.config["feature_contract"]["sequence_columns"]),
        )

    def create_model(self, num_numerical: int, cat_cardinalities: list[int], device: torch.device) -> nn.Module:
        return create_model(
            self.spec.kind,
            self.config,
            num_numerical,
            cat_cardinalities,
        ).to(device)

    def create_criterion(self, labels: np.ndarray, device: torch.device) -> nn.Module:
        if self.spec.kind == "xapi":
            return nn.BCEWithLogitsLoss().to(device)
        effective_weights = None
        if self.config["class_weight_mode"] == "balanced":
            encoded = np.asarray(labels, dtype=int)
            counts = np.bincount(encoded, minlength=3)
            weights = len(encoded) / (3 * np.maximum(counts, 1))
            effective_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        if self.config["loss"] == "focal":
            return FocalLoss(weight=effective_weights, gamma=float(self.config["focal_gamma"])).to(device)
        return nn.CrossEntropyLoss(weight=effective_weights).to(device)

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(
            model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )

    def criterion_signature(self) -> dict[str, Any]:
        return {
            "loss": str(self.config["loss"]),
            "class_weight_mode": str(self.config["class_weight_mode"]),
            "focal_gamma": self.config.get("focal_gamma"),
        }

    def resampling_signature(self) -> dict[str, Any]:
        return {
            "oversample_method": str(self.config["oversample_method"]),
            "smote_ratio": float(self.config["smote_ratio"]),
            "resampling_k_neighbors": int(self.config["resampling_k_neighbors"]),
            "scope": str(self.config["preprocessing"]["oversampling_scope"]),
        }

    def estimator_signature(self) -> dict[str, Any]:
        return {
            "factory": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "resolved_config_hash": self.config_hash,
            "criterion": self.criterion_signature(),
            "resampling": self.resampling_signature(),
            "scheduler": deepcopy(self.config["scheduler"]),
            "swa": deepcopy(self.config["swa"]),
            "drop_last_train": bool(self.config["drop_last_train"]),
            "preprocessing": deepcopy(self.config["preprocessing"]),
            "feature_contract": deepcopy(self.config["feature_contract"]),
            "target_contract": deepcopy(self.config["target_contract"]),
        }
