"""Build Phase 7 per-action weak-label matrices from frozen Phase 6 labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.weak_supervision.matrix import (  # noqa: E402
    A4SourceGateError,
    SOURCES_BY_ACTION,
    build_matrices,
    load_canonical_sources,
    validate_phase7_authority,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/phase6_llm_labels.parquet")
    parser.add_argument("--behavior", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet")
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "artifacts/recommendation/labeling/phase6_source_manifest.json")
    parser.add_argument("--phase7-input", type=Path, default=ROOT / "configs/recommendation/phase7_input.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/recommendation/weak_supervision.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/matrices")
    args = parser.parse_args()
    try:
        validate_phase7_authority(args.source_manifest, args.phase7_input, args.panel_a, args.panel_b, weak_supervision_path=args.config)
        sources = load_canonical_sources(args.llm, args.behavior, args.panel_a, args.panel_b)
        matrices = build_matrices(sources, args.panel_a, args.output_dir)
    except A4SourceGateError as exc:
        print(str(exc))
        return 2
    summary = {
        "actions": {
            action_id: {"rows": len(frame), "sources": list(SOURCES_BY_ACTION[action_id]), "shape": [int(frame.shape[0]), int(frame.shape[1] - 1)]}
            for action_id, frame in matrices.items()
        }
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
