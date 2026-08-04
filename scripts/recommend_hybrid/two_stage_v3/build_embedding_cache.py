"""Build cross-fitted and outer-specific frozen hybrid representation caches.

The residual CNN-BiLSTM checkpoint authority is never updated.  Cross-fitted
representations support inner model selection.  For each final outer run, every
training and test group is re-embedded by the fold-k authority that was trained
without outer fold k, preventing outer-test risk-label leakage.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/two_stage_v3"
CACHE = OUT / "cache"
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/two_stage_v3_protocol.yaml"
sys.path.insert(0, str(ROOT))

from src.pipelines.oulad import STATIC_COLUMNS, _build_bundle  # noqa: E402
from src.recommend_hybrid.contracts import Stage  # noqa: E402
from src.recommend_hybrid.prediction_adapter import (  # noqa: E402
    HybridPredictionAdapter,
)

ACTION_ORDER = (
    "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY",
    "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW",
)
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_ORDER)}
STAGE_AUTHORITY = {
    "EARLY_20": ("E1_EARLY_20PCT", Stage.EARLY_20),
    "EARLY_35": ("E2_EARLY_35PCT", Stage.EARLY_35),
    "MIDDLE_50": ("M1_MIDDLE_FROZEN", Stage.MIDDLE_50),
}
EMBEDDING_BATCH_SIZE = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _group_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group in candidates.groupby("group_id", sort=False):
        risk = np.sort(
            pd.to_numeric(group["risk_reduction"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )[::-1]
        margin = float(risk[0] - risk[1]) if len(risk) > 1 else float(risk[0])
        first = group.iloc[0]
        rows.append(
            {
                "group_id": str(group_id),
                "base_record_id": str(first["base_record_id"]),
                "stage": str(first["stage"]),
                "outer_fold": int(first["outer_fold"]),
                "course": str(first["course"]),
                "presentation": str(first["presentation"]),
                "group_has_positive": int(group["silver_positive"].max()),
                "candidate_count": int(len(group)),
                "maximum_risk_reduction": float(risk[0]),
                "mean_risk_reduction": float(risk.mean()),
                "maximum_deficit": float(
                    pd.to_numeric(group["deficit_score"], errors="coerce")
                    .fillna(0.0)
                    .max()
                ),
                "mean_evidence_strength": float(
                    pd.to_numeric(group["evidence_strength"], errors="coerce")
                    .fillna(0.0)
                    .mean()
                ),
                "maximum_evidence_strength": float(
                    pd.to_numeric(group["evidence_strength"], errors="coerce")
                    .fillna(0.0)
                    .max()
                ),
                "top_counterfactual_margin": margin,
            }
        )
    return pd.DataFrame(rows)


def _prepare_action_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    frame["action_index"] = frame["action_family"].map(ACTION_INDEX)
    if frame["action_index"].isna().any():
        unknown = sorted(frame.loc[frame["action_index"].isna(), "action_family"].unique())
        raise RuntimeError(f"unknown action families: {unknown}")
    frame["action_index"] = frame["action_index"].astype(int)
    group_target = frame.groupby("group_id", sort=False)["silver_positive"].transform("max")
    frame["group_has_positive"] = group_target.astype(int)
    required = [
        "group_id",
        "base_record_id",
        "stage",
        "outer_fold",
        "course",
        "presentation",
        "action_family",
        "action_index",
        "risk_reduction",
        "risk_uncertainty",
        "evidence_strength",
        "deficit_score",
        "opportunity_count",
        "workload_minutes",
        "action_available",
        "prerequisite_status",
        "silver_positive",
        "group_has_positive",
    ]
    missing = set(required) - set(frame.columns)
    if missing:
        raise RuntimeError(f"candidate rows are missing columns: {sorted(missing)}")
    duplicate = frame.duplicated(["group_id", "action_index"])
    if duplicate.any():
        raise RuntimeError("duplicate action slot within a ranking group")
    return frame[required].sort_values(
        ["group_id", "action_index"], kind="stable"
    ).reset_index(drop=True)


def _stage_index(stage_frame: pd.DataFrame) -> dict[str, int]:
    values = stage_frame["base_record_id"].astype(str)
    if values.duplicated().any():
        raise RuntimeError("bundle stage contains duplicate base_record_id")
    return {value: index for index, value in enumerate(values)}


def _embed_rows(
    *,
    adapter: HybridPredictionAdapter,
    stage_data: Any,
    indexes: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    for model in adapter.models:
        model.to(device)
        model.eval()
    aggregate = adapter.transform_aggregate(stage_data.aggregate[indexes])
    selected_frame = stage_data.frame.iloc[indexes]
    static = adapter.transform_static(
        {column: selected_frame[column].tolist() for column in STATIC_COLUMNS}
    )
    student_rows: list[np.ndarray] = []
    tabular_rows: list[np.ndarray] = []
    risk_rows: list[np.ndarray] = []
    entropy_rows: list[np.ndarray] = []
    disagreement_rows: list[np.ndarray] = []
    confidence_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indexes), EMBEDDING_BATCH_SIZE):
            stop = min(len(indexes), start + EMBEDDING_BATCH_SIZE)
            inputs = {
                "sequence": torch.as_tensor(
                    stage_data.sequence[indexes[start:stop]],
                    dtype=torch.float32,
                    device=device,
                ),
                "lengths": torch.as_tensor(
                    stage_data.lengths[indexes[start:stop]],
                    dtype=torch.int64,
                    device=device,
                ),
                "mask": torch.as_tensor(
                    stage_data.mask[indexes[start:stop]],
                    dtype=torch.float32,
                    device=device,
                ),
                "aggregate": torch.as_tensor(
                    aggregate[start:stop], dtype=torch.float32, device=device
                ),
                "static": torch.as_tensor(
                    static[start:stop], dtype=torch.float32, device=device
                ),
            }
            output = adapter.predict(inputs)
            student_rows.append(output.student_state_embedding.cpu().numpy())
            tabular_rows.append(output.tabular_expert_embedding.cpu().numpy())
            risk_rows.append(output.probabilities[:, 1].cpu().numpy())
            entropy_rows.append(output.uncertainty.cpu().numpy())
            disagreement_rows.append(output.seed_disagreement.cpu().numpy())
            confidence_rows.append(output.confidence.cpu().numpy())
    return {
        "student": np.concatenate(student_rows).astype(np.float32),
        "tabular": np.concatenate(tabular_rows).astype(np.float32),
        "risk_probability": np.concatenate(risk_rows).astype(np.float32),
        "risk_entropy": np.concatenate(entropy_rows).astype(np.float32),
        "seed_disagreement": np.concatenate(disagreement_rows).astype(np.float32),
        "risk_confidence": np.concatenate(confidence_rows).astype(np.float32),
    }


def _build_cache(
    *,
    groups: pd.DataFrame,
    bundle: Any,
    output_path: Path,
    authority_mode: str,
    fixed_fold: int | None,
    device: torch.device,
) -> dict[str, Any]:
    result_rows: list[pd.DataFrame] = []
    authority_rows: list[dict[str, Any]] = []
    for stage_name, (bundle_stage, canonical_stage) in STAGE_AUTHORITY.items():
        stage_data = bundle.stages[bundle_stage]
        index_map = _stage_index(stage_data.frame)
        stage_groups = groups[groups["stage"] == stage_name].copy()
        fold_values = (
            [int(fixed_fold)]
            if fixed_fold is not None
            else sorted(stage_groups["outer_fold"].unique())
        )
        for fold in fold_values:
            selected = (
                stage_groups
                if fixed_fold is not None
                else stage_groups[stage_groups["outer_fold"] == fold]
            ).copy()
            if selected.empty:
                continue
            indexes = selected["base_record_id"].map(index_map)
            if indexes.isna().any():
                missing = selected.loc[indexes.isna(), "base_record_id"].head(10).tolist()
                raise RuntimeError(
                    f"{stage_name}: groups missing from OULAD bundle: {missing}"
                )
            adapter = HybridPredictionAdapter.from_manifest(
                ROOT,
                stage=canonical_stage,
                fold=int(fold),
            )
            embedding = _embed_rows(
                adapter=adapter,
                stage_data=stage_data,
                indexes=indexes.to_numpy(dtype=np.int64),
                device=device,
            )
            enriched = selected.reset_index(drop=True)
            for column in range(embedding["student"].shape[1]):
                enriched[f"student_state_{column:03d}"] = embedding["student"][:, column]
            for column in range(embedding["tabular"].shape[1]):
                enriched[f"tabular_expert_{column:03d}"] = embedding["tabular"][:, column]
            for name in (
                "risk_probability",
                "risk_entropy",
                "seed_disagreement",
                "risk_confidence",
            ):
                enriched[name] = embedding[name]
            enriched["embedding_authority_fold"] = int(fold)
            enriched["embedding_authority_mode"] = authority_mode
            result_rows.append(enriched)
            authority_rows.append(
                {
                    "stage": stage_name,
                    "authority_fold": int(fold),
                    "group_count": int(len(enriched)),
                    "checkpoint_ids": [
                        reference.checkpoint_id
                        for reference in adapter.checkpoint_references
                    ],
                    "checkpoint_sha256": [
                        reference.sha256 for reference in adapter.checkpoint_references
                    ],
                    "frozen_preprocessor_hash": adapter.frozen_preprocessor_hash,
                }
            )
            for model in adapter.models:
                model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    cache_frame = pd.concat(result_rows, ignore_index=True)
    if cache_frame["group_id"].duplicated().any():
        raise RuntimeError(f"{authority_mode}: duplicate group embeddings")
    if len(cache_frame) != len(groups):
        raise RuntimeError(
            f"{authority_mode}: cache rows={len(cache_frame)} groups={len(groups)}"
        )
    cache_frame.sort_values("group_id", kind="stable", inplace=True)
    _atomic_parquet(output_path, cache_frame)
    return {
        "path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256(output_path),
        "rows": int(len(cache_frame)),
        "authority_mode": authority_mode,
        "authority_rows": authority_rows,
    }


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_HEAD_TRAINING":
        raise RuntimeError("two-stage V3 protocol is not frozen before training")
    source_path = ROOT / protocol["frozen_data_authority"]["candidate_rows"] if "frozen_data_authority" in protocol else ROOT / "artifacts/recommend_hybrid/hybrid_only_final/dataset/candidate_rows.parquet"
    if not source_path.exists():
        source_path = ROOT / "artifacts/recommend_hybrid/hybrid_only_final/dataset/candidate_rows.parquet"
    candidates = pd.read_parquet(source_path)
    action_candidates = _prepare_action_candidates(candidates)
    groups = _group_summary(action_candidates)
    if int(groups["group_has_positive"].sum()) != 9304:
        raise RuntimeError("diagnostic positive-group authority changed")
    if len(groups) != 29043:
        raise RuntimeError("diagnostic ranking-group authority changed")

    CACHE.mkdir(parents=True, exist_ok=True)
    action_path = CACHE / "ACTION_CANDIDATES.parquet"
    _atomic_parquet(action_path, action_candidates)
    bundle = _build_bundle()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caches: list[dict[str, Any]] = []
    caches.append(
        _build_cache(
            groups=groups,
            bundle=bundle,
            output_path=CACHE / "cross_fitted/GROUP_FEATURES.parquet",
            authority_mode="CROSS_FITTED_OWN_OUTER_AUTHORITY",
            fixed_fold=None,
            device=device,
        )
    )
    for fold in protocol["evaluation"]["outer_folds"]:
        caches.append(
            _build_cache(
                groups=groups,
                bundle=bundle,
                output_path=CACHE / f"outer_{fold}/GROUP_FEATURES.parquet",
                authority_mode=f"OUTER_{fold}_AUTHORITY_FOR_ALL_GROUPS",
                fixed_fold=int(fold),
                device=device,
            )
        )
    registry = {
        "schema_version": "two_stage_v3_embedding_cache_v1",
        "status": "COMPLETE",
        "device": str(device),
        "backbone_trainable": False,
        "student_state_dimension": 64,
        "tabular_expert_dimension": 32,
        "groups": int(len(groups)),
        "positive_groups": int(groups["group_has_positive"].sum()),
        "action_candidates": {
            "path": str(action_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(action_path),
            "rows": int(len(action_candidates)),
        },
        "caches": caches,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
    }
    _atomic_json(CACHE / "CACHE_REGISTRY.json", registry)
    print(json.dumps(registry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
