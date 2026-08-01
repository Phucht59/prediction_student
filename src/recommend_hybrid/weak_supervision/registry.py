"""Load and serialize the scientific source and action registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.recommend_hybrid.common.policy_contracts import DatasetId

from .contracts import ActionEvidenceMapping, SourceRecord


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_sources(path: Path) -> tuple[SourceRecord, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(
        SourceRecord(
            **{**row, "supported_action_ids": tuple(row["supported_action_ids"])}
        )
        for row in payload["sources"]
    )


def load_action_mappings(path: Path) -> tuple[ActionEvidenceMapping, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    sequence_fields = {
        "supported_stages",
        "required_evidence",
        "prerequisites",
        "contraindications",
        "evidence_source_ids",
    }
    records = []
    for row in payload["actions"]:
        values = {key: tuple(value) if key in sequence_fields else value for key, value in row.items()}
        values["supported_datasets"] = tuple(DatasetId(value) for value in row["supported_datasets"])
        records.append(ActionEvidenceMapping(**values))
    return tuple(records)


__all__ = ["load_action_mappings", "load_sources", "stable_json"]
