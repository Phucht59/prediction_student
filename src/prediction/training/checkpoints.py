"""One Hybrid class checkpoint semantics with fail-closed loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..model import Hybrid, HybridConfig


INSTANCE_LAYOUT = {
    "uci": "checkpoints/uci",
    "oulad": "checkpoints/oulad",
}


def save_checkpoint(path: str | Path, model: Hybrid, *, instance: str, metadata: dict[str, Any] | None = None) -> None:
    if not isinstance(model, Hybrid):
        raise TypeError("only Hybrid checkpoints are supported")
    if instance not in INSTANCE_LAYOUT:
        raise ValueError(f"unknown fitted instance: {instance}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": "hybrid", "instance": instance, "config": model.config.__dict__, "state_dict": model.state_dict(), "metadata": metadata or {}}, target)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> Hybrid:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("model_id") != "hybrid":
        raise ValueError("checkpoint is not a Hybrid checkpoint")
    fields = set(HybridConfig.__dataclass_fields__)
    config = HybridConfig(**{k: v for k, v in payload["config"].items() if k in fields})
    model = Hybrid(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


__all__ = ["INSTANCE_LAYOUT", "save_checkpoint", "load_checkpoint"]
