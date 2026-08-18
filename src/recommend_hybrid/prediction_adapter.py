"""Read-only adapter for the frozen Hybrid CNN-BiLSTM prediction authority."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.models.oulad_tabular_residual import CNNBiLSTMTabularResidualOULAD

from .contracts import CheckpointReference, Stage
from .exceptions import AuthorityValidationError

ARCHITECTURE_HASH = "df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e"
PARAMETER_COUNT = 160_492


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
        for row in rows:
            checkpoint_path = root / row["provenance"]["source_checkpoint_path"]
            if file_sha256(checkpoint_path) != row["sha256"]:
                raise AuthorityValidationError("checkpoint SHA-256 mismatch")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if payload.get("architecture_hash") != ARCHITECTURE_HASH:
                raise AuthorityValidationError("checkpoint architecture hash mismatch")
            if int(payload.get("parameter_count", -1)) != PARAMETER_COUNT:
                raise AuthorityValidationError("checkpoint parameter count mismatch")
            model = CNNBiLSTMTabularResidualOULAD(
                47, int(payload["aggregate_dim"]), int(payload["static_dim"]), payload["config"]
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
        return cls(
            models,
            references,
            stage=stage,
            fold=fold,
            decision_threshold=decision_threshold,
        )

    @property
    def models(self) -> tuple[torch.nn.Module, ...]:
        return self._models

    @property
    def checkpoint_references(self) -> tuple[CheckpointReference, ...]:
        return self._references

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
    "ARCHITECTURE_HASH",
    "PARAMETER_COUNT",
    "HybridPredictionAdapter",
    "HybridPredictionOutput",
    "file_sha256",
    "parameter_sha256",
]
