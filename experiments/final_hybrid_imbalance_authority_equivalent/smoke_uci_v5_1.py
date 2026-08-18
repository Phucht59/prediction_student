"""One-batch structural smoke test for archived UCI V5.1 authority paths."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from adapter_common import RUNTIME, assert_equivalence, atomic_json, fold_recipe, historical_imports, partition, source_partition_for_mat


def _finite(output: dict[str, torch.Tensor]) -> None:
    if output["classification"].ndim != 2 or output["classification"].shape[1] != 3:
        raise RuntimeError(f"Expected [batch,3] logits, got {tuple(output['classification'].shape)}")
    if not all(torch.isfinite(value).all().item() for value in output.values()):
        raise RuntimeError("Non-finite tensor in historical model output")


def smoke(dataset: str) -> dict:
    historical_imports()
    from src.studies.v5_1.common.uci_training import deterministic_seed, multitask_loss
    from src.studies.v5_1.common.uci_model import UCIHybridV51
    from src.studies.v5_1.common.uci_transfer import SharedTrunkSubjectHeadsV51, combine_subject_inputs

    signature = assert_equivalence(dataset)
    _, data, _, outer_test, train, _, transformer = partition(dataset, 0)
    if "G3" in data.context.columns:
        raise RuntimeError("G3 leaked into predictor context")
    recipe = fold_recipe(dataset, 0)
    config = recipe["config"]
    deterministic_seed(42)
    if dataset == "student-mat":
        source_train, _, safe = source_partition_for_mat(data, outer_test, transformer, 0)
        combined = combine_subject_inputs(train, source_train)
        model = SharedTrunkSubjectHeadsV51(combined.temporal.shape[2], combined.context.shape[1], {**config, "subject_embedding_dim": 4})
        batch = min(8, len(combined.target))
        output = model(torch.from_numpy(combined.temporal[:batch]), torch.from_numpy(combined.context[:batch]), torch.from_numpy(combined.subject[:batch]))
        target, raw = torch.from_numpy(combined.target[:batch]), torch.from_numpy(combined.raw_g3[:batch])
        transfer = {"method": "shared_trunk_subject_specific_heads", "pretrained_checkpoint": None, "shared_trunk": True, "mat_head_index": 0, "source_training_records": len(source_train.target), "overlap_safe_source_records": len(safe), "frozen_layers": [], "newly_initialized_heads": ["classifiers.0", "regressors.0", "ordinals.0"]}
    else:
        model = UCIHybridV51(train.temporal.shape[2], train.context.shape[1], config)
        batch = min(8, len(train.target))
        output = model(torch.from_numpy(train.temporal[:batch]), torch.from_numpy(train.context[:batch]))
        target, raw = torch.from_numpy(train.target[:batch]), torch.from_numpy(train.raw_g3[:batch])
        transfer = {"method": "standalone", "pretrained_checkpoint": None, "random_initialization": True, "frozen_layers": []}
    _finite(output)
    loss, _ = multitask_loss(output, target, raw, config=config, class_weights=None, regression_mean=float(train.raw_g3.mean()), regression_std=float(max(train.raw_g3.std(), 1e-6)))
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite smoke loss")
    loss.backward()
    torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])).step()
    result = {"status": "PASS", "dataset": dataset, "outer_fold": 0, "logits_shape": list(output["classification"].shape), "loss_finite": True, "g3_absent": True, "parameter_count": sum(p.numel() for p in model.parameters()), "signature": signature, "transfer": transfer}
    name = "MAT_TRANSFER_VALIDATION.json" if dataset == "student-mat" else "POR_TRAINING_PATH_VALIDATION.json"
    atomic_json(RUNTIME / name, result)
    return result


def main() -> None:
    results = {dataset: smoke(dataset) for dataset in ("student-mat", "student-por")}
    print(json.dumps({name: value["status"] for name, value in results.items()}))


if __name__ == "__main__":
    main()
