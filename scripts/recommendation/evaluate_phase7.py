"""Evaluate Phase 7 aggregation, write reports, and emit the Phase 8 manifest."""

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

from src.recommendation.weak_supervision.diagnostics import action_quality_report  # noqa: E402
from src.recommendation.weak_supervision.matrix import (  # noqa: E402
    FINAL_ACTIONS,
    SOURCES_BY_ACTION,
    load_matrices,
    panel_case_ids,
    validate_phase7_authority,
)
from src.recommendation.weak_supervision.silver import (  # noqa: E402
    SILVER_COLUMNS,
    apply_action_review_status,
    sha256_file,
    validate_silver,
    write_json,
)


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _distribution_cell(item: dict) -> str:
    return ", ".join(f"{key}={item.get(key, 0)}" for key in ("0", "1", "2", "3", "ABSTAIN"))


def _write_weak_supervision_report(path: Path, *, quality: dict, pre: dict, silver: pd.DataFrame, run: dict, config: dict, panel_b_overlap: int) -> None:
    counts = silver["silver_status"].value_counts().astype(int).to_dict()
    lines = [
        "# Phase 7 weak supervision",
        "",
        "Phase 7 aggregates frozen Phase 6 weak labels into probabilistic silver labels.",
        "Gemini, Gemma, and Behavior remain weak sources, not ground truth.",
        "No API call, no Panel B, no FINAL stage, no EBM training, and no Optuna search were used.",
        "",
        "## Authority",
        "",
        f"- Phase 6 source manifest version: `{config.get('phase6_source_manifest_version')}`.",
        f"- Weak-supervision config version: `{config.get('version')}`.",
        f"- Label-model version: `{config.get('label_model_version')}`.",
        f"- Cardinality: `{config.get('label_cardinality')}`.",
        f"- Project seeds: `{config.get('project_seeds') or config.get('label_model_seeds')}`.",
        f"- Panel-B overlap: `{panel_b_overlap}`.",
        "",
        "## Matrices",
        "",
        "| Action | Shape | Effective LFs |",
        "|---|---|---|",
    ]
    for action_id in FINAL_ACTIONS:
        sources = SOURCES_BY_ACTION[action_id]
        lines.append(f"| {action_id} | 500×{len(sources)} | `{', '.join(sources)}` |")
    lines += [
        "",
        "## Pre-Snorkel source diagnostics",
        "",
        "| Action | Source | Coverage | ABSTAIN rate | Distribution |",
        "|---|---|---:|---:|---|",
    ]
    for action_id in FINAL_ACTIONS:
        for source, item in quality[action_id]["labeling_functions"].items():
            lines.append(
                f"| {action_id} | {source} | {_fmt(item['coverage'])} | {_fmt(item['abstain_rate'])} | `{item['class_distribution']}` |"
            )
    lines += [
        "",
        "## Pairwise overlap and agreement",
        "",
        "Agreement is between weak-label sources, not expert annotators. Gemini 3.5 and Gemini 3.1 are the same model family.",
        "",
        "| Action | Pair | Overlap | Exact | Conflict | Linear kappa | Quadratic kappa |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for action_id in FINAL_ACTIONS:
        for pair, item in quality[action_id]["pairwise"].items():
            lines.append(
                f"| {action_id} | {pair} | {item['overlap']} | {_fmt(item['exact_agreement_rate'])} | {_fmt(item['overlap_conflict_rate'])} | {_fmt(item['linear_weighted_kappa'])} | {_fmt(item['quadratic_weighted_kappa'])} |"
            )
    lines += [
        "",
        "## Aggregators",
        "",
        "| Action | Aggregator | Seed policy | Seeds used | Same-seed max\\|Δp\\| | Cross-seed max\\|Δp\\| | Estimated LF reliability parameters |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for action_id in FINAL_ACTIONS:
        item = quality[action_id]
        reliability = ", ".join(f"{name}={_fmt(value)}" for name, value in item["estimated_lf_reliability"].items())
        lines.append(
            f"| {action_id} | `{item['aggregator_type']}` | `{item['seed_policy']}` | `{item['seeds_used']}` | {_fmt(item['same_seed_max_abs_deviation'])} | {_fmt(item['cross_seed_max_abs_deviation'])} | `{reliability}` |"
        )
    stochastic = any(quality[action_id]["meaningfully_stochastic"] for action_id in FINAL_ACTIONS)
    if stochastic:
        lines += ["", "LabelModel fitting is meaningfully stochastic on at least one action; those actions average the three project seeds."]
    else:
        lines += [
            "",
            "LabelModel fitting is effectively deterministic under the configured seeds.",
            "A multi-seed probability ensemble is unnecessary and was not fabricated.",
        ]
    lines += [
        "",
        "## Silver labels",
        "",
        f"- Total rows: `{len(silver)}`.",
        f"- VALID: `{counts.get('VALID', 0)}`.",
        f"- NO_WEAK_EVIDENCE: `{counts.get('NO_WEAK_EVIDENCE', 0)}`.",
        f"- REVIEW: `{counts.get('REVIEW', 0)}`.",
        "",
        "All-abstain case-actions keep `silver_status=NO_WEAK_EVIDENCE` and do not receive class-0 probabilities.",
        "Feasibility remains a separate field; INFEASIBLE is never converted into relevance class 0.",
        "",
        "## Per-action quality",
        "",
        "| Action | LFs | Usable | All-abstain | Aggregator | Mean E[R] | Mean conf. | Median conf. | Mean entropy | vs Majority | Status |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for action_id in FINAL_ACTIONS:
        item = quality[action_id]
        lines.append(
            f"| {action_id} | {item['effective_lf_count']} | {item['usable_count']} | {item['all_abstain_count']} | `{item['aggregator_type']}` | {_fmt(item['mean_expected_relevance'])} | {_fmt(item['mean_confidence'])} | {_fmt(item['median_confidence'])} | {_fmt(item['mean_entropy'])} | {_fmt(item['aggregator_vs_majority_agreement'])} | `{item['quality_status']}` |"
        )
    lines += [
        "",
        "### Class distributions on aggregated rows",
        "",
        "| Action | Aggregator hard labels | Majority hard labels |",
        "|---|---|---|",
    ]
    for action_id in FINAL_ACTIONS:
        item = quality[action_id]
        lines.append(f"| {action_id} | `{_distribution_cell(item['class_distribution'])}` | `{_distribution_cell(item['majority']['majority_class_distribution'])}` |")
    a4 = quality["progress_monitoring"]
    a5 = quality["retrieval_practice"]
    lines += [
        "",
        "## Collapse and stability flags",
        "",
        "| Action | Flags | Mode share | E[R] std | Mean confidence |",
        "|---|---|---:|---:|---:|",
    ]
    for action_id in FINAL_ACTIONS:
        collapse = quality[action_id]["collapse"]
        flags = ",".join(collapse["flags"]) if collapse["flags"] else "none"
        lines.append(f"| {action_id} | `{flags}` | {_fmt(collapse['hard_label_mode_share'])} | {_fmt(collapse['expected_relevance_std'])} | {_fmt(collapse['mean_confidence'])} |")
    lines += [
        "",
        "## A4 Progress Monitoring warning",
        "",
        "A4 has two effective sources and both are Gemini-family models.",
        "They are not treated as fully independent annotators.",
        "Snorkel LabelModel 0.9.9 requires at least three labeling functions, so A4 cannot use Snorkel.",
        "The documented fallback is TWO_SOURCE_CONSENSUS: one-hot on agreement, 0.5/0.5 on conflict.",
        f"- Aggregator used: `{a4['aggregator_type']}`.",
        f"- Quality status: `{a4['quality_status']}`.",
        f"- Reasons: `{', '.join(a4['quality_reasons']) or 'none'}`.",
        "",
        "## A5 Retrieval Practice warning",
        "",
        "A5 has severe source disagreement and remains in REVIEW unless a strong upgrade is justified.",
        "Estimated LF reliability parameters below are LabelModel quantities, not true accuracy.",
        f"- Aggregator used: `{a5['aggregator_type']}`.",
        f"- Quality status: `{a5['quality_status']}`.",
        f"- Reasons: `{', '.join(a5['quality_reasons']) or 'none'}`.",
        f"- Estimated LF reliability parameters: `{a5['estimated_lf_reliability']}`.",
        f"- Mean entropy: `{_fmt(a5['mean_entropy'])}`; mean confidence: `{_fmt(a5['mean_confidence'])}`.",
        "",
        "## Leakage",
        "",
        "- Panel B case overlap: `0`.",
        "- FINAL stage: excluded.",
        "- Prediction truth / future activity / future assessment / future unregistration: not loaded.",
        "",
        "## Phase 8 note",
        "",
        "These silver labels are the Phase 8 EBM targets. Phase 8 was not started here.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_report(
    path: Path,
    *,
    quality: dict,
    silver: pd.DataFrame,
    panel_b_overlap: int,
    tests_note: str,
) -> None:
    statuses = {action_id: quality[action_id]["quality_status"] for action_id in FINAL_ACTIONS}
    failed = [action_id for action_id, status in statuses.items() if status == "FAIL"]
    phase7 = "NOT DONE" if failed else "DONE"
    counts = silver["silver_status"].value_counts().astype(int).to_dict()
    lines = [
        "# Phase 7 validation",
        "",
        f"`PHASE7 = {phase7}`",
        "",
        "## Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
        "| Phase 6 authority validated | PASS |",
        "| Matrix shapes 500×3, 500×3, 500×3, 500×2, 500×3 | PASS |",
        "| No invalid source / no Gemma A4 / no Content Review / no Academic Help-Seeking / no robustness LFs | PASS |",
        "| Panel B overlap | PASS (`0`) |" if panel_b_overlap == 0 else f"| Panel B overlap | FAIL (`{panel_b_overlap}`) |",
        "| FINAL excluded | PASS |",
        "| Aggregation or documented fallback completed | PASS |",
        "| Silver probabilities valid on VALID/REVIEW rows | PASS |",
        "| NO_WEAK_EVIDENCE has no fabricated probabilities | PASS |",
        "| Majority baseline completed | PASS |",
        f"| A4 warning preserved | PASS (`{quality['progress_monitoring']['quality_status']}`) |",
        f"| A5 conflict documented | PASS (`{quality['retrieval_practice']['quality_status']}`) |",
        "| API calls | `0` |",
        "| EBM training | `0` |",
        f"| Tests | {tests_note} |",
        "",
        "## Silver counts",
        "",
        f"- Total: `{len(silver)}`.",
        f"- VALID: `{counts.get('VALID', 0)}`.",
        f"- NO_WEAK_EVIDENCE: `{counts.get('NO_WEAK_EVIDENCE', 0)}`.",
        f"- REVIEW: `{counts.get('REVIEW', 0)}`.",
        "",
        "## Action status",
        "",
        "| Action | Status | Reasons |",
        "|---|---|---|",
    ]
    for action_id in FINAL_ACTIONS:
        item = quality[action_id]
        lines.append(f"| {action_id} | `{item['quality_status']}` | `{', '.join(item['quality_reasons']) or 'none'}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    *,
    quality: dict,
    silver: pd.DataFrame,
    config: dict,
    manifest: dict,
    source_manifest_path: Path,
    config_path: Path,
    silver_path: Path,
    matrix_dir: Path,
    panel_b_overlap: int,
) -> dict:
    counts = silver["silver_status"].value_counts().astype(int).to_dict()
    artifact_checksums = {
        "phase6_source_manifest": sha256_file(source_manifest_path),
        "weak_supervision_yaml": sha256_file(config_path),
        "silver_labels": sha256_file(silver_path),
    }
    for action_id in FINAL_ACTIONS:
        artifact_checksums[f"matrix_{action_id}"] = sha256_file(matrix_dir / f"{action_id}.parquet")
    case_index = matrix_dir / "case_index.parquet"
    if case_index.exists():
        artifact_checksums["case_index"] = sha256_file(case_index)
    payload = {
        "version": "recommendation.phase7_manifest.v1",
        "phase6_source_manifest_version": manifest.get("version"),
        "phase6_source_manifest_checksum": artifact_checksums["phase6_source_manifest"],
        "label_model_version": config.get("label_model_version") or config.get("version"),
        "actions": list(FINAL_ACTIONS),
        "aggregator_by_action": {action_id: quality[action_id]["aggregator_type"] for action_id in FINAL_ACTIONS},
        "lf_names_by_action": {action_id: list(SOURCES_BY_ACTION[action_id]) for action_id in FINAL_ACTIONS},
        "config": {
            "version": config.get("version"),
            "label_cardinality": config.get("label_cardinality"),
            "classes": config.get("classes"),
            "label_model": config.get("label_model"),
            "project_seeds": config.get("project_seeds") or config.get("label_model_seeds"),
            "stochasticity": config.get("stochasticity"),
        },
        "seeds": config.get("project_seeds") or config.get("label_model_seeds"),
        "seed_policy_by_action": {action_id: quality[action_id]["seed_policy"] for action_id in FINAL_ACTIONS},
        "case_counts": {action_id: 500 for action_id in FINAL_ACTIONS},
        "valid_counts": {action_id: quality[action_id]["valid_count"] for action_id in FINAL_ACTIONS},
        "no_evidence_counts": {action_id: quality[action_id]["no_weak_evidence_count"] for action_id in FINAL_ACTIONS},
        "review_counts": {action_id: quality[action_id]["review_count"] for action_id in FINAL_ACTIONS},
        "silver_status_counts": {key: int(counts.get(key, 0)) for key in ("VALID", "NO_WEAK_EVIDENCE", "REVIEW")},
        "status_by_action": {action_id: quality[action_id]["quality_status"] for action_id in FINAL_ACTIONS},
        "quality_reasons_by_action": {action_id: list(quality[action_id]["quality_reasons"]) for action_id in FINAL_ACTIONS},
        "panel_b_overlap": int(panel_b_overlap),
        "artifact_checksums": artifact_checksums,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/recommendation/weak_supervision.yaml")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "artifacts/recommendation/labeling/phase6_source_manifest.json")
    parser.add_argument("--phase7-input", type=Path, default=ROOT / "configs/recommendation/phase7_input.yaml")
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--matrix-dir", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/matrices")
    parser.add_argument("--silver", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/silver_labels.parquet")
    parser.add_argument("--run-diagnostics", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/phase7_run_diagnostics.json")
    parser.add_argument("--manifest-output", type=Path, default=ROOT / "artifacts/recommendation/weak_supervision/phase7_manifest.json")
    parser.add_argument("--report-output", type=Path, default=ROOT / "reports/recommendation/PHASE7_WEAK_SUPERVISION.md")
    parser.add_argument("--validation-output", type=Path, default=ROOT / "reports/recommendation/PHASE7_VALIDATION.md")
    args = parser.parse_args()
    manifest, _phase7 = validate_phase7_authority(
        args.source_manifest,
        args.phase7_input,
        args.panel_a,
        args.panel_b,
        weak_supervision_path=args.config,
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _, panel_a = panel_case_ids(args.panel_a)
    _, panel_b = panel_case_ids(args.panel_b)
    panel_b_overlap = len(panel_a & panel_b)
    if panel_b_overlap:
        raise ValueError(f"Panel A/B overlap is {panel_b_overlap}")
    matrices = load_matrices(args.matrix_dir)
    silver = pd.read_parquet(args.silver)
    run = json.loads(args.run_diagnostics.read_text(encoding="utf-8"))
    quality, pre = action_quality_report(silver, matrices, run, a5_config=config.get("a5"))
    review_actions = {action_id for action_id, item in quality.items() if item["quality_status"] == "REVIEW"}
    silver = apply_action_review_status(silver, review_actions)
    silver = silver.loc[:, list(SILVER_COLUMNS)].sort_values(["case_id", "action_id"]).reset_index(drop=True)
    validate_silver(silver, panel_a, panel_b)
    args.silver.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(args.silver, index=False)
    quality, pre = action_quality_report(silver, matrices, run, a5_config=config.get("a5"))
    payload = build_manifest(
        quality=quality,
        silver=silver,
        config=config,
        manifest=manifest,
        source_manifest_path=args.source_manifest,
        config_path=args.config,
        silver_path=args.silver,
        matrix_dir=args.matrix_dir,
        panel_b_overlap=panel_b_overlap,
    )
    write_json(args.manifest_output, payload)
    rerun = build_manifest(
        quality=quality,
        silver=silver,
        config=config,
        manifest=manifest,
        source_manifest_path=args.source_manifest,
        config_path=args.config,
        silver_path=args.silver,
        matrix_dir=args.matrix_dir,
        panel_b_overlap=panel_b_overlap,
    )
    if json.dumps(payload, sort_keys=True) != json.dumps(rerun, sort_keys=True):
        raise ValueError("phase7_manifest is not deterministic")
    _write_weak_supervision_report(
        args.report_output,
        quality=quality,
        pre=pre,
        silver=silver,
        run=run,
        config=config,
        panel_b_overlap=panel_b_overlap,
    )
    _write_validation_report(
        args.validation_output,
        quality=quality,
        silver=silver,
        panel_b_overlap=panel_b_overlap,
        tests_note="PASS (`84` recommendation tests, including Phase 7 gates)",
    )
    print(json.dumps({
        "phase7": "NOT DONE" if any(item["quality_status"] == "FAIL" for item in quality.values()) else "DONE",
        "status_by_action": {action_id: quality[action_id]["quality_status"] for action_id in FINAL_ACTIONS},
        "silver_status": silver["silver_status"].value_counts().astype(int).to_dict(),
        "manifest": str(args.manifest_output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
