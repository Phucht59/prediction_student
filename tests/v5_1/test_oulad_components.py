from __future__ import annotations

import torch
import optuna

from src.studies.v5_1.oulad.data import COMPACT_SUMMARIES, compact_aggregate_columns
from src.studies.v5_1.oulad.models import OULADHybridV51, attention_entropy, count_parameters
from src.studies.v5_1.oulad.runner import _unique_completed_trials, _unique_trial_count


def _config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "input_projection": 16,
        "conv_channels": 8,
        "kernels": [2, 3, 5],
        "dilation": 1,
        "lstm_hidden": 12,
        "lstm_layers": 1,
        "pooling": "masked_attention",
        "pooling_projection": 16,
        "aggregate_hidden": 16,
        "static_hidden": 8,
        "fusion_hidden": 16,
        "fusion": "gated_residual",
        "branch_dropout": 0.1,
        "dropout": 0.1,
    }
    config.update(updates)
    return config


def _inputs(batch: int = 4, weeks: int = 8):
    lengths = torch.tensor([weeks, weeks - 2, weeks - 4, 2])[:batch]
    mask = torch.arange(weeks)[None, :] < lengths[:, None]
    sequence = torch.randn(batch, weeks, 47) * mask.unsqueeze(-1)
    return sequence, lengths, mask.float(), torch.randn(batch, 43), torch.randn(batch, 9)


def test_compact_aggregate_contract_is_deterministic_and_reduced() -> None:
    available = [
        f"{channel}__{summary}"
        for channel, summaries in COMPACT_SUMMARIES.items()
        for summary in set(summaries) | {"std", "min", "max"}
    ] + ["inactive_week_count", "unrelated_full_oracle_feature"]
    first = compact_aggregate_columns(available)
    second = compact_aggregate_columns(list(reversed(available)))
    assert first == second
    assert len(first) == 49
    assert len(first) < len(available)
    assert "unrelated_full_oracle_feature" not in first


def test_masked_attention_assigns_zero_probability_to_padding() -> None:
    sequence, lengths, mask, aggregate, static = _inputs()
    model = OULADHybridV51(47, 43, 9, _config()).eval()
    _, diagnostics = model(sequence, lengths, mask, aggregate, static, return_diagnostics=True)
    attention = diagnostics["attention"]
    assert attention is not None
    assert float(attention.masked_select(~mask.bool()).max()) == 0.0
    entropy = attention_entropy(attention, mask)
    assert entropy is not None
    assert torch.all((entropy >= 0) & (entropy <= 1.00001))


def test_residual_multi_kernel_and_gate_receive_gradient() -> None:
    sequence, lengths, mask, aggregate, static = _inputs()
    model = OULADHybridV51(47, 43, 9, _config())
    logits = model(sequence, lengths, mask, aggregate, static)
    logits.sum().backward()
    assert model.temporal.residual.weight.grad is not None
    assert all(layer.weight.grad is not None for layer in model.temporal.convolutions)
    assert model.gates is not None and model.gates[0].weight.grad is not None
    assert count_parameters(model) < 1_500_000


def test_all_registered_temporal_variants_and_fusions_have_valid_shapes() -> None:
    sequence, lengths, mask, aggregate, static = _inputs()
    for variant in ["cnn_bilstm", "cnn_only", "bilstm_only"]:
        for fusion in ["concatenation", "gated_residual"]:
            model = OULADHybridV51(47, 43, 9, _config(fusion=fusion), variant)
            logits, diagnostics = model(
                sequence, lengths, mask, aggregate, static, return_diagnostics=True
            )
            assert logits.shape == (4,)
            assert (diagnostics["gate"] is None) is (fusion == "concatenation")


def test_padded_values_cannot_change_eval_prediction() -> None:
    sequence, lengths, mask, aggregate, static = _inputs()
    changed = sequence.clone()
    changed[~mask.bool()] = 10000.0
    model = OULADHybridV51(47, 43, 9, _config(), "cnn_bilstm").eval()
    first = model(sequence, lengths, mask, aggregate, static)
    second = model(changed, lengths, mask, aggregate, static)
    torch.testing.assert_close(first, second)


def test_stage_gated_budget_counts_exact_optuna_duplicates_once() -> None:
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"width": 32})
    study.enqueue_trial({"width": 32})

    def objective(trial: optuna.Trial) -> float:
        return float(trial.suggest_categorical("width", [32, 64]))

    study.optimize(objective, n_trials=2)
    assert len(study.trials) == 2
    assert len(_unique_completed_trials(study)) == 1
    assert _unique_trial_count(study) == 1
