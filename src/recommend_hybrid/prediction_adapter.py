"""Read-only adapter for the frozen Hybrid CNN-BiLSTM prediction authority."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.models.oulad_tabular_residual import CNNBiLSTMTabularResidualOULAD

from .contracts import CheckpointReference, Stage
from .exceptions import AuthorityValidationError

ARCHITECTURE_HASH = "df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e"
PARAMETER_COUNT = 160_492
AGGREGATE_DIMENSION = 165
STATIC_DIMENSION = 13


@dataclass(frozen=True)
class HybridPredictionOutput:
    """Typed tensors emitted by one frozen checkpoint or a locked seed ensemble."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: torch.Tensor
    decision_threshold: float
    confidence: torch.Tensor
    uncertainty: torch.Tensor
    seed_disagreement: torch.Tensor
    student_state_embedding: torch.Tensor
    tabular_expert_embedding: torch.Tensor
    stage: Stage
    fold: int
    seeds: tuple[int, ...]
    checkpoint_references: tuple[CheckpointReference, ...]
    architecture_hash: str
    confidence_source: str = "RAW_MAX_CLASS_PROBABILITY"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array, dtype=np.float64).reshape(-1)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _preprocessor_sha256(preprocessor: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("mean", "scale", "num_mean", "num_scale"):
        value = np.asarray(preprocessor.get(name), dtype=np.float64).reshape(-1)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    digest.update(
        json.dumps(
            {
                "num_cols": [str(item) for item in preprocessor.get("num_cols", ())],
                "categories": {
                    str(key): [str(item) for item in values]
                    for key, values in dict(preprocessor.get("categories", {})).items()
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _to_numeric(values: Sequence[Any]) -> np.ndarray:
    converted: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if not math.isfinite(number):
            number = 0.0
        converted.append(number)
    return np.asarray(converted, dtype=np.float64)


class HybridPredictionAdapter:
    """Expose existing model outputs without changing model state or prediction path."""

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        checkpoint_references: Sequence[CheckpointReference],
        *,
        stage: Stage,
        fold: int,
        decision_threshold: float = 0.5,
        aggregate_mean: np.ndarray | None = None,
        aggregate_scale: np.ndarray | None = None,
        static_num_cols: Sequence[str] | None = None,
        static_num_mean: np.ndarray | None = None,
        static_num_scale: np.ndarray | None = None,
        static_categories: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if not models or len(models) != len(checkpoint_references):
            raise AuthorityValidationError("models and checkpoint references must align")
        if len({ref.seed for ref in checkpoint_references}) != len(checkpoint_references):
            raise AuthorityValidationError("seed references must be unique")
        if any(ref.fold != fold for ref in checkpoint_references):
            raise AuthorityValidationError("checkpoint fold does not match adapter fold")
        self._models = tuple(models)
        self._references = tuple(checkpoint_references)
        self.stage = stage
        self.fold = fold
        if not 0.0 < decision_threshold < 1.0:
            raise AuthorityValidationError("decision threshold must be in (0, 1)")
        self.decision_threshold = float(decision_threshold)
        for model in self._models:
            if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
                raise AuthorityValidationError("model parameter count is not frozen authority")

        if (aggregate_mean is None) != (aggregate_scale is None):
            raise AuthorityValidationError(
                "aggregate mean and scale must be supplied together"
            )
        self._aggregate_mean: np.ndarray | None = None
        self._aggregate_scale: np.ndarray | None = None
        if aggregate_mean is not None and aggregate_scale is not None:
            mean = np.asarray(aggregate_mean, dtype=np.float64).reshape(-1)
            scale = np.asarray(aggregate_scale, dtype=np.float64).reshape(-1)
            if mean.shape != (AGGREGATE_DIMENSION,) or scale.shape != (
                AGGREGATE_DIMENSION,
            ):
                raise AuthorityValidationError(
                    "frozen aggregate preprocessor must have 165 features"
                )
            if not np.isfinite(mean).all() or not np.isfinite(scale).all():
                raise AuthorityValidationError(
                    "aggregate preprocessor contains non-finite values"
                )
            if np.any(scale <= 0.0):
                raise AuthorityValidationError(
                    "aggregate preprocessor scale must be positive"
                )
            self._aggregate_mean = mean.copy()
            self._aggregate_scale = scale.copy()

        static_values = (
            static_num_cols,
            static_num_mean,
            static_num_scale,
            static_categories,
        )
        if any(value is not None for value in static_values) and not all(
            value is not None for value in static_values
        ):
            raise AuthorityValidationError(
                "static preprocessor state must be supplied together"
            )
        self._static_num_cols: tuple[str, ...] | None = None
        self._static_num_mean: np.ndarray | None = None
        self._static_num_scale: np.ndarray | None = None
        self._static_categories: dict[str, tuple[str, ...]] | None = None
        if all(value is not None for value in static_values):
            assert static_num_cols is not None
            assert static_num_mean is not None
            assert static_num_scale is not None
            assert static_categories is not None
            columns = tuple(str(item) for item in static_num_cols)
            num_mean = np.asarray(static_num_mean, dtype=np.float64).reshape(-1)
            num_scale = np.asarray(static_num_scale, dtype=np.float64).reshape(-1)
            categories = {
                str(key): tuple(str(item) for item in values)
                for key, values in static_categories.items()
            }
            if not columns or len(columns) != len(set(columns)):
                raise AuthorityValidationError(
                    "static numeric columns must be non-empty and unique"
                )
            if num_mean.shape != (len(columns),) or num_scale.shape != (
                len(columns),
            ):
                raise AuthorityValidationError(
                    "static numeric preprocessor dimensions are invalid"
                )
            if not np.isfinite(num_mean).all() or not np.isfinite(num_scale).all():
                raise AuthorityValidationError(
                    "static numeric preprocessor contains non-finite values"
                )
            if np.any(num_scale <= 0.0):
                raise AuthorityValidationError(
                    "static numeric preprocessor scale must be positive"
                )
            if not categories or any(not levels for levels in categories.values()):
                raise AuthorityValidationError(
                    "static categorical levels must be non-empty"
                )
            dimension = len(columns) + sum(len(levels) for levels in categories.values())
            if dimension != STATIC_DIMENSION:
                raise AuthorityValidationError(
                    f"frozen static preprocessor must produce {STATIC_DIMENSION} features"
                )
            self._static_num_cols = columns
            self._static_num_mean = num_mean.copy()
            self._static_num_scale = num_scale.copy()
            self._static_categories = categories

    @classmethod
    def from_manifest(
        cls,
        root: Path,
        *,
        stage: Stage,
        fold: int,
        seeds: Sequence[int] | None = None,
        manifest_path: Path = Path(
            "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
        ),
    ) -> "HybridPredictionAdapter":
        manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
        if manifest.get("architecture_hash") != ARCHITECTURE_HASH:
            raise AuthorityValidationError("manifest architecture hash mismatch")
        selected_seeds = tuple(seeds or (42, 1201, 2026, 3407, 7319))
        rows = [
            row
            for row in manifest["checkpoints"]
            if stage.value in row["stages"]
            and int(row["outer_fold"]) == fold
            and int(row["seed"]) in selected_seeds
        ]
        rows.sort(key=lambda row: selected_seeds.index(int(row["seed"])))
        if len(rows) != len(selected_seeds):
            raise AuthorityValidationError("manifest does not cover requested stage/fold/seeds")
        training_authority = json.loads(
            (root / "artifacts/canonical_v3/oulad_h1_training_authority.json").read_text(
                encoding="utf-8"
            )
        )
        role = "endpoint_final" if stage is Stage.FINAL_EVALUATION else "shared_stage"
        source_stage = {
            Stage.EARLY_20: "E1_EARLY_20PCT",
            Stage.EARLY_35: "E2_EARLY_35PCT",
            Stage.MIDDLE_50: "M1_MIDDLE_50PCT",
            Stage.LATE_75: "L1_LATE_75PCT",
            Stage.FINAL_EVALUATION: "FINAL",
        }[stage]
        authority_row = next(
            row for row in training_authority[role] if int(row["outer_fold"]) == fold
        )
        decision_threshold = float(authority_row["thresholds"][source_stage])
        models: list[torch.nn.Module] = []
        references: list[CheckpointReference] = []
        preprocessor_hash: str | None = None
        frozen_preprocessor: Mapping[str, Any] | None = None
        for row in rows:
            checkpoint_path = root / row["provenance"]["source_checkpoint_path"]
            if file_sha256(checkpoint_path) != row["sha256"]:
                raise AuthorityValidationError("checkpoint SHA-256 mismatch")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if payload.get("architecture_hash") != ARCHITECTURE_HASH:
                raise AuthorityValidationError("checkpoint architecture hash mismatch")
            if int(payload.get("parameter_count", -1)) != PARAMETER_COUNT:
                raise AuthorityValidationError("checkpoint parameter count mismatch")
            if int(payload.get("aggregate_dim", -1)) != AGGREGATE_DIMENSION:
                raise AuthorityValidationError("checkpoint aggregate dimension mismatch")
            if int(payload.get("static_dim", -1)) != STATIC_DIMENSION:
                raise AuthorityValidationError("checkpoint static dimension mismatch")
            preprocessor = payload.get("preprocessor")
            if not isinstance(preprocessor, Mapping):
                raise AuthorityValidationError(
                    "checkpoint frozen preprocessor is missing"
                )
            current_hash = _preprocessor_sha256(preprocessor)
            if preprocessor_hash is None:
                preprocessor_hash = current_hash
                frozen_preprocessor = preprocessor
            elif current_hash != preprocessor_hash:
                raise AuthorityValidationError(
                    "seed checkpoints disagree on frozen preprocessor"
                )
            model = CNNBiLSTMTabularResidualOULAD(
                47,
                int(payload["aggregate_dim"]),
                int(payload["static_dim"]),
                payload["config"],
            )
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            models.append(model)
            references.append(
                CheckpointReference(
                    checkpoint_id=row["checkpoint_id"],
                    path=row["provenance"]["source_checkpoint_path"],
                    sha256=row["sha256"],
                    fold=fold,
                    seed=int(row["seed"]),
                )
            )
        if frozen_preprocessor is None:
            raise AuthorityValidationError("no frozen preprocessor was loaded")
        return cls(
            models,
            references,
            stage=stage,
            fold=fold,
            decision_threshold=decision_threshold,
            aggregate_mean=np.asarray(frozen_preprocessor["mean"]),
            aggregate_scale=np.asarray(frozen_preprocessor["scale"]),
            static_num_cols=tuple(frozen_preprocessor["num_cols"]),
            static_num_mean=np.asarray(frozen_preprocessor["num_mean"]),
            static_num_scale=np.asarray(frozen_preprocessor["num_scale"]),
            static_categories=dict(frozen_preprocessor["categories"]),
        )

    @property
    def models(self) -> tuple[torch.nn.Module, ...]:
        return self._models

    @property
    def checkpoint_references(self) -> tuple[CheckpointReference, ...]:
        return self._references

    @property
    def has_aggregate_preprocessor(self) -> bool:
        return self._aggregate_mean is not None and self._aggregate_scale is not None

    @property
    def has_static_preprocessor(self) -> bool:
        return (
            self._static_num_cols is not None
            and self._static_num_mean is not None
            and self._static_num_scale is not None
            and self._static_categories is not None
        )

    @property
    def aggregate_preprocessor_hash(self) -> str | None:
        if not self.has_aggregate_preprocessor:
            return None
        assert self._aggregate_mean is not None
        assert self._aggregate_scale is not None
        return _array_sha256(self._aggregate_mean, self._aggregate_scale)

    @property
    def frozen_preprocessor_hash(self) -> str | None:
        if not self.has_aggregate_preprocessor or not self.has_static_preprocessor:
            return None
        assert self._aggregate_mean is not None
        assert self._aggregate_scale is not None
        assert self._static_num_cols is not None
        assert self._static_num_mean is not None
        assert self._static_num_scale is not None
        assert self._static_categories is not None
        return _preprocessor_sha256(
            {
                "mean": self._aggregate_mean,
                "scale": self._aggregate_scale,
                "num_cols": self._static_num_cols,
                "num_mean": self._static_num_mean,
                "num_scale": self._static_num_scale,
                "categories": self._static_categories,
            }
        )

    def transform_aggregate(self, raw_aggregate: np.ndarray) -> np.ndarray:
        if not self.has_aggregate_preprocessor:
            raise AuthorityValidationError(
                "adapter has no frozen aggregate preprocessor"
            )
        assert self._aggregate_mean is not None
        assert self._aggregate_scale is not None
        values = np.asarray(raw_aggregate, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != AGGREGATE_DIMENSION:
            raise AuthorityValidationError("raw aggregate must be [N, 165]")
        return np.nan_to_num(
            (values - self._aggregate_mean) / self._aggregate_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

    def inverse_transform_aggregate(
        self,
        transformed_aggregate: np.ndarray,
    ) -> np.ndarray:
        if not self.has_aggregate_preprocessor:
            raise AuthorityValidationError(
                "adapter has no frozen aggregate preprocessor"
            )
        assert self._aggregate_mean is not None
        assert self._aggregate_scale is not None
        values = np.asarray(transformed_aggregate, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != AGGREGATE_DIMENSION:
            raise AuthorityValidationError(
                "transformed aggregate must be [N, 165]"
            )
        return (values * self._aggregate_scale + self._aggregate_mean).astype(
            np.float32
        )

    def transform_static(
        self,
        records: Mapping[str, Sequence[Any]],
    ) -> np.ndarray:
        if not self.has_static_preprocessor:
            raise AuthorityValidationError(
                "adapter has no frozen static preprocessor"
            )
        assert self._static_num_cols is not None
        assert self._static_num_mean is not None
        assert self._static_num_scale is not None
        assert self._static_categories is not None
        required = set(self._static_num_cols) | set(self._static_categories)
        missing = sorted(required - set(records))
        if missing:
            raise AuthorityValidationError(
                f"static records are missing columns: {missing}"
            )
        lengths = {len(records[column]) for column in required}
        if len(lengths) != 1:
            raise AuthorityValidationError(
                "static record columns must have equal lengths"
            )
        row_count = lengths.pop()
        if row_count <= 0:
            raise AuthorityValidationError("static records cannot be empty")
        numeric = np.column_stack(
            [_to_numeric(records[column]) for column in self._static_num_cols]
        )
        numeric = (numeric - self._static_num_mean) / self._static_num_scale
        categorical: list[np.ndarray] = []
        for column, levels in self._static_categories.items():
            values = np.asarray(
                ["__MISSING__" if value is None else str(value) for value in records[column]],
                dtype=object,
            )
            categorical.append(
                np.column_stack(
                    [(values == level).astype(np.float32) for level in levels]
                )
            )
        result = np.concatenate([numeric, *categorical], axis=1).astype(np.float32)
        if result.shape != (row_count, STATIC_DIMENSION):
            raise AuthorityValidationError(
                "frozen static transformation produced unexpected dimensions"
            )
        return result

    def predict(self, inputs: Mapping[str, torch.Tensor]) -> HybridPredictionOutput:
        required = {"sequence", "lengths", "mask", "aggregate", "static"}
        if set(inputs) != required:
            raise AuthorityValidationError(f"model inputs must be exactly {sorted(required)}")
        outputs: list[dict[str, torch.Tensor]] = []
        with torch.inference_mode():
            for model in self._models:
                model.eval()
                outputs.append(model(**inputs))
        logits_by_seed = torch.stack([out["binary_logit"] for out in outputs], dim=0)
        risk_by_seed = torch.sigmoid(logits_by_seed)
        risk_probability = risk_by_seed.mean(dim=0)
        probabilities = torch.stack((1.0 - risk_probability, risk_probability), dim=-1)
        logits = logits_by_seed.mean(dim=0)
        predicted_class = (risk_probability >= self.decision_threshold).to(torch.int64)
        confidence = probabilities.max(dim=-1).values
        clipped = risk_probability.clamp(1e-12, 1.0 - 1e-12)
        uncertainty = -(clipped * clipped.log() + (1.0 - clipped) * (1.0 - clipped).log())
        seed_disagreement = risk_by_seed.std(dim=0, unbiased=False)
        student_embedding = torch.stack(
            [out["student_state_embedding"] for out in outputs], dim=0
        ).mean(dim=0)
        tabular_embedding = torch.stack(
            [out["tabular_expert_embedding"] for out in outputs], dim=0
        ).mean(dim=0)
        if student_embedding.shape[-1] != 64 or tabular_embedding.shape[-1] != 32:
            raise AuthorityValidationError("frozen embedding dimension mismatch")
        return HybridPredictionOutput(
            logits=logits.detach(),
            probabilities=probabilities.detach(),
            predicted_class=predicted_class.detach(),
            decision_threshold=self.decision_threshold,
            confidence=confidence.detach(),
            uncertainty=uncertainty.detach(),
            seed_disagreement=seed_disagreement.detach(),
            student_state_embedding=student_embedding.detach(),
            tabular_expert_embedding=tabular_embedding.detach(),
            stage=self.stage,
            fold=self.fold,
            seeds=tuple(ref.seed for ref in self._references),
            checkpoint_references=self._references,
            architecture_hash=ARCHITECTURE_HASH,
        )


__all__ = [
    "AGGREGATE_DIMENSION",
    "ARCHITECTURE_HASH",
    "PARAMETER_COUNT",
    "STATIC_DIMENSION",
    "HybridPredictionAdapter",
    "HybridPredictionOutput",
    "file_sha256",
    "parameter_sha256",
]
