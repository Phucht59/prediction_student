from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.models.oulad_tabular_residual import CNNBiLSTMTabularResidualOULAD
from src.recommend_hybrid.action_catalog import ActionCatalog
from src.recommend_hybrid.contracts import (
    CheckpointReference,
    PredictionContext,
    Stage,
)
from src.recommend_hybrid.observed_state import ActivityEvent, AssessmentEvent, ObservedStateBuilder
from src.recommend_hybrid.prediction_adapter import HybridPredictionAdapter


@pytest.fixture(scope="session")
def root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def checkpoint_row(root: Path) -> dict:
    manifest = json.loads(
        (root / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json").read_text()
    )
    return next(
        row
        for row in manifest["checkpoints"]
        if row["usage"] == "INTERVENTION_STAGE_SHARED"
        and row["outer_fold"] == 0
        and row["seed"] == 42
    )


@pytest.fixture(scope="session")
def frozen_model(root: Path, checkpoint_row: dict) -> torch.nn.Module:
    payload = torch.load(
        root / checkpoint_row["provenance"]["source_checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    model = CNNBiLSTMTabularResidualOULAD(
        47, payload["aggregate_dim"], payload["static_dim"], payload["config"]
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


@pytest.fixture(scope="session")
def model_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2202)
    return {
        "sequence": torch.randn(3, 8, 47, generator=generator),
        "lengths": torch.tensor([8, 6, 4], dtype=torch.int64),
        "mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ),
        "aggregate": torch.randn(3, 165, generator=generator),
        "static": torch.randn(3, 13, generator=generator),
    }


@pytest.fixture(scope="session")
def adapter(frozen_model: torch.nn.Module, checkpoint_row: dict) -> HybridPredictionAdapter:
    reference = CheckpointReference(
        checkpoint_id=checkpoint_row["checkpoint_id"],
        path=checkpoint_row["provenance"]["source_checkpoint_path"],
        sha256=checkpoint_row["sha256"],
        fold=0,
        seed=42,
    )
    return HybridPredictionAdapter(
        [frozen_model], [reference], stage=Stage.MIDDLE_50, fold=0
    )


@pytest.fixture()
def observed_state():
    return ObservedStateBuilder().build(
        stage=Stage.MIDDLE_50,
        cutoff_day=100,
        activity_events=(ActivityEvent(70, 12), ActivityEvent(92, 8)),
        assessment_events=(AssessmentEvent(80, 79), AssessmentEvent(95, 94)),
    )


@pytest.fixture()
def prediction_context(checkpoint_row: dict) -> PredictionContext:
    reference = CheckpointReference(
        checkpoint_id=checkpoint_row["checkpoint_id"],
        path=checkpoint_row["provenance"]["source_checkpoint_path"],
        sha256=checkpoint_row["sha256"],
        fold=0,
        seed=42,
    )
    return PredictionContext(
        student_key="student-key",
        course_key="course-key",
        stage=Stage.MIDDLE_50,
        cutoff_day=100,
        predicted_class=1,
        class_probabilities=(0.25, 0.75),
        confidence=0.75,
        uncertainty=0.56,
        seed_disagreement=0.03,
        fold=0,
        seeds=(42,),
        checkpoint_references=(reference,),
        architecture_hash="df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e",
        parameter_count=160492,
    )


@pytest.fixture()
def catalog(root: Path) -> ActionCatalog:
    return ActionCatalog.load(root / "configs/recommend_hybrid/actions.yaml")
