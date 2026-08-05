"""Create stage-specific frozen Hybrid embedding archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows.parquet"
DEFAULT_OUTPUT = ROOT / "artifacts/recommend_hybrid/causal/input/embeddings_by_stage"
STAGES = ("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75")
SPLITS = ("train", "validation", "test")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def run(input_path: Path, output_dir: Path) -> dict[str, object]:
    frame = _read(input_path)
    required = {"student_id", "stage", "protocol_split", "prediction_target"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"landmark table is missing columns: {missing}")
    embedding_columns = sorted(
        column for column in frame.columns if column.startswith("embedding__")
    )
    if not embedding_columns:
        raise ValueError("landmark table has no embedding__ columns")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for stage in STAGES:
        stage_frame = frame.loc[frame["stage"].eq(stage)].drop_duplicates(
            ["student_id", "stage"]
        )
        if stage_frame.empty:
            raise ValueError(f"no embedding rows available for {stage}")
        payload: dict[str, np.ndarray] = {
            "stage": np.asarray(stage),
            "embedding_names": np.asarray(embedding_columns, dtype=str),
        }
        counts[stage] = {}
        for split in SPLITS:
            selected = stage_frame.loc[stage_frame["protocol_split"].eq(split)]
            if selected.empty:
                raise ValueError(f"{stage} has no rows for split {split}")
            values = selected.loc[:, embedding_columns].to_numpy(dtype=np.float32)
            target = selected["prediction_target"].to_numpy(dtype=np.int8)
            if not np.isfinite(values).all() or not np.isin(target, [0, 1]).all():
                raise ValueError(f"{stage}/{split} contains invalid embedding data")
            payload[f"{split}_embeddings"] = values
            payload[f"{split}_target"] = target
            payload[f"{split}_student_ids"] = selected["student_id"].astype(
                str
            ).to_numpy(dtype=str)
            counts[stage][split] = int(len(selected))
        path = output_dir / f"frozen_embeddings_{stage.lower()}.npz"
        np.savez_compressed(path, **payload)
        artifacts[stage] = str(path.relative_to(ROOT))
    manifest = {
        "status": "COMPLETE",
        "source": str(input_path.relative_to(ROOT)),
        "artifacts": artifacts,
        "counts": counts,
        "resampling_applied": False,
        "split_rule": "PREEXISTING_STUDENT_GROUPED_PROTOCOL_SPLIT",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir)))


if __name__ == "__main__":
    main()
