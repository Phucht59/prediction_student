"""Run none/class_weight/SMOTE/ADASYN on frozen Hybrid embeddings."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.imbalance import run_frozen_embedding_imbalance_study  # noqa: E402

DEFAULT_INPUT = ROOT / "artifacts/recommend_hybrid/causal/input/frozen_embeddings.npz"
DEFAULT_OUTPUT = ROOT / "artifacts/recommend_hybrid/imbalance/metrics.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(input_path: Path, output_path: Path, *, seed: int) -> dict[str, object]:
    with np.load(input_path, allow_pickle=False) as data:
        required = {
            "train_embeddings",
            "train_target",
            "validation_embeddings",
            "validation_target",
            "test_embeddings",
            "test_target",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"embedding archive is missing keys: {missing}")
        result = run_frozen_embedding_imbalance_study(
            train_features=data["train_embeddings"],
            train_target=data["train_target"],
            validation_features=data["validation_embeddings"],
            validation_target=data["validation_target"],
            test_features=data["test_embeddings"],
            test_target=data["test_target"],
            random_state=seed,
        )
        metadata: dict[str, object] = {}
        for key in ("dataset", "stage", "outer_fold", "checkpoint_sha256"):
            if key in data.files:
                raw = data[key]
                metadata[key] = raw.item() if raw.ndim == 0 else raw.tolist()

    payload = {
        "status": "COMPLETE",
        "input": str(input_path),
        "metadata": metadata,
        **result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    payload = run(args.input, args.output, seed=args.seed)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
