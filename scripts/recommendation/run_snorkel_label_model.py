"""Fit independent Phase 7 aggregators and write silver labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.weak_supervision.diagnostics import assign_quality_status, collapse_flags, pre_snorkel_diagnostics  # noqa: E402
from src.recommendation.weak_supervision.label_model import A5_ACTION, fit_label_models  # noqa: E402
from src.recommendation.weak_supervision.matrix import (  # noqa: E402
    A4SourceGateError,
    build_matrices,
    load_canonical_sources,
    load_matrices,
    validate_phase7_authority,
)
from src.recommendation.weak_supervision.silver import (  # noqa: E402
    SILVER_COLUMNS,
    apply_action_review_status,
    attach_feasibility,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/recommendation/weak_supervision.yaml")
    parser.add_argument("--llm", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/phase6_llm_labels.parquet")
    parser.add_argument("--behavior", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet")
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "artifacts/recommendation/labeling/phase6_source_manifest.json")
    parser.add_argument("--phase7-input", type=Path, default=ROOT / "configs/recommendation/phase7_input.yaml")
    parser.add_argument("--feasibility", type=Path, default=ROOT / "artifacts/recommendation/feasibility/oulad_action_feasibility.parquet")
    parser.add_argument("--matrix-dir", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/matrices")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/silver_labels.parquet")
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/phase7_run_diagnostics.json")
    args = parser.parse_args()
    try:
        manifest, _phase7 = validate_phase7_authority(
            args.source_manifest,
            args.phase7_input,
            args.panel_a,
            args.panel_b,
            weak_supervision_path=args.config,
        )
        if not args.matrix_dir.exists() or not (args.matrix_dir / "assessment_recovery.parquet").exists():
            sources = load_canonical_sources(args.llm, args.behavior, args.panel_a, args.panel_b)
            matrices = build_matrices(sources, args.panel_a, args.matrix_dir)
        else:
            matrices = load_matrices(args.matrix_dir)
    except A4SourceGateError as exc:
        print(str(exc))
        return 2
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    silver, diagnostics = fit_label_models(
        matrices,
        seeds=tuple(config.get("label_model_seeds") or config["project_seeds"]),
        train_config=config.get("label_model"),
        stochasticity_threshold=float(config.get("stochasticity", {}).get("max_abs_deviation_threshold", 1e-6)),
        average_if_stochastic=bool(config.get("stochasticity", {}).get("average_if_stochastic", True)),
        a4_config=config.get("a4"),
        label_model_version=str(config.get("label_model_version") or config["version"]),
        phase6_source_manifest_version=str(config.get("phase6_source_manifest_version") or manifest["version"]),
    )
    panel_a_ids = set(pd.read_parquet(args.panel_a, columns=["case_id"])["case_id"].astype(str))
    silver = attach_feasibility(silver, args.feasibility, panel_a_ids)
    pre = pre_snorkel_diagnostics(matrices)
    review_actions = set()
    if A5_ACTION in diagnostics:
        collapse = collapse_flags(
            silver[silver["action_id"] == A5_ACTION],
            pre[A5_ACTION]["pairwise"],
            float(diagnostics[A5_ACTION]["cross_seed_max_abs_deviation"]),
            bool(diagnostics[A5_ACTION]["meaningfully_stochastic"]),
        )
        status, _reasons = assign_quality_status(
            pairwise=pre[A5_ACTION]["pairwise"],
            collapse=collapse,
            aggregator_type=diagnostics[A5_ACTION]["aggregator_type"],
            correlated_family=False,
            a5_config=config.get("a5"),
            usable_count=int(diagnostics[A5_ACTION]["usable_count"]),
        )
        if status == "REVIEW":
            review_actions.add(A5_ACTION)
    silver = apply_action_review_status(silver, review_actions)
    silver = silver.loc[:, list(SILVER_COLUMNS)].sort_values(["case_id", "action_id"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(args.output, index=False)
    write_json(args.diagnostics, diagnostics)
    print(json.dumps({
        "silver_rows": int(len(silver)),
        "status_counts": silver["silver_status"].value_counts().astype(int).to_dict(),
        "aggregators": {action_id: item["aggregator_type"] for action_id, item in diagnostics.items()},
        "seed_policies": {action_id: item["seed_policy"] for action_id, item in diagnostics.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
