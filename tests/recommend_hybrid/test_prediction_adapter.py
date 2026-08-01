from __future__ import annotations

import copy

import torch

from src.recommend_hybrid.prediction_adapter import file_sha256, parameter_sha256


def _direct(frozen_model, model_inputs):
    frozen_model.eval()
    with torch.inference_mode():
        output = frozen_model(**model_inputs)
    risk = torch.sigmoid(output["binary_logit"])
    probabilities = torch.stack((1 - risk, risk), dim=-1)
    return output, probabilities


def test_prediction_adapter_preserves_logits(adapter, frozen_model, model_inputs):
    direct, _ = _direct(frozen_model, model_inputs)
    adapted = adapter.predict(model_inputs)
    assert torch.equal(adapted.logits, direct["binary_logit"])


def test_prediction_adapter_preserves_probabilities(adapter, frozen_model, model_inputs):
    _, direct = _direct(frozen_model, model_inputs)
    assert torch.equal(adapter.predict(model_inputs).probabilities, direct)


def test_prediction_adapter_preserves_class(adapter, frozen_model, model_inputs):
    _, direct = _direct(frozen_model, model_inputs)
    expected = (direct[:, 1] >= adapter.decision_threshold).to(torch.int64)
    assert torch.equal(adapter.predict(model_inputs).predicted_class, expected)


def test_checkpoint_hash_unchanged(adapter, root, checkpoint_row, model_inputs):
    path = root / checkpoint_row["provenance"]["source_checkpoint_path"]
    before = file_sha256(path)
    adapter.predict(model_inputs)
    assert file_sha256(path) == before == checkpoint_row["sha256"]


def test_parameter_values_unchanged(adapter, frozen_model, model_inputs):
    before = parameter_sha256(frozen_model)
    adapter.predict(model_inputs)
    assert parameter_sha256(frozen_model) == before


def test_student_embedding_dimension(adapter, model_inputs):
    assert adapter.predict(model_inputs).student_state_embedding.shape == (3, 64)


def test_tabular_embedding_dimension(adapter, model_inputs):
    assert adapter.predict(model_inputs).tabular_expert_embedding.shape == (3, 32)


def test_adapter_eval_deterministic(adapter, model_inputs):
    first = adapter.predict(model_inputs)
    second = adapter.predict(model_inputs)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.student_state_embedding, second.student_state_embedding)
