"""Single, thesis-friendly command line entry point.

This module intentionally exposes only routine repository operations. Expensive
historical study runners remain under ``scripts/`` and are not reachable from
this CLI, which prevents an evidence check from accidentally starting training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "configs" / "extension_protocol_v1.yaml"
DEFAULT_OULAD_OUTPUT = ROOT / "data" / "processed" / "study_c_oulad"
DEFAULT_FIGURE_SOURCE = (
    ROOT
    / "artifacts"
    / "oulad"
    / "final"
    / "ensemble_metrics.csv"
)
DEFAULT_FIGURE_OUTPUT = ROOT / "reports" / "thesis_figures"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def command_status(_args: argparse.Namespace) -> int:
    """Show the three official evidence bundles without recomputing results."""

    bundles = {
        "student_mat": ROOT / "artifacts" / "student_mat" / "final",
        "student_por": ROOT / "artifacts" / "student_por" / "final",
        "oulad": ROOT / "artifacts" / "oulad" / "final",
    }
    result = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "available": path.is_dir(),
        }
        for name, path in bundles.items()
    }
    result["all_official_bundles_available"] = all(
        entry["available"] for entry in result.values() if isinstance(entry, dict)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["all_official_bundles_available"] else 1


def command_validate(_args: argparse.Namespace) -> int:
    """Validate frozen evidence and headline metrics; never train a model."""

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_release.py")],
        cwd=ROOT,
        check=False,
    )
    return int(completed.returncode)


def command_ingest(args: argparse.Namespace) -> int:
    """Ingest one UCI dataset into the canonical PostgreSQL source tables."""

    from src.config import DATASETS
    from src.postgres_data_source import ingest_dataset_csv_to_postgres

    if args.dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    result = ingest_dataset_csv_to_postgres(args.dataset)
    print(
        "ingested "
        f"dataset={args.dataset} "
        f"dataset_version_id={result['dataset_version_id']} "
        f"row_count={result['row_count']} "
        f"source_record_count={result['source_record_count']}"
    )
    return 0


def command_audit_oulad(args: argparse.Namespace) -> int:
    """Audit the raw OULAD release keys, labels and source hashes."""

    import pandas as pd

    from src.studies.common.hashing import sha256_file

    raw_root = args.raw_root.resolve()
    output = args.output.resolve()
    table_names = [
        "courses",
        "assessments",
        "studentAssessment",
        "studentInfo",
        "studentRegistration",
        "vle",
    ]
    tables = {name: pd.read_csv(raw_root / f"{name}.csv") for name in table_names}
    student_key = ["code_module", "code_presentation", "id_student"]
    course_key = ["code_module", "code_presentation"]
    vle_key = ["code_module", "code_presentation", "id_site"]
    checks = {
        "student_info_key_unique": not tables["studentInfo"].duplicated(student_key).any(),
        "registration_key_unique": not tables["studentRegistration"].duplicated(student_key).any(),
        "course_key_unique": not tables["courses"].duplicated(course_key).any(),
        "assessment_id_unique": not tables["assessments"].duplicated("id_assessment").any(),
        "vle_full_key_unique": not tables["vle"].duplicated(vle_key).any(),
        "student_info_registration_keys_equal": set(
            map(tuple, tables["studentInfo"][student_key].to_numpy())
        )
        == set(map(tuple, tables["studentRegistration"][student_key].to_numpy())),
        "assessment_parent_complete": set(tables["studentAssessment"]["id_assessment"]).issubset(
            set(tables["assessments"]["id_assessment"])
        ),
        "final_result_valid": set(tables["studentInfo"]["final_result"])
        == {"Withdrawn", "Fail", "Pass", "Distinction"},
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": {key: len(value) for key, value in tables.items()} | {"studentVle": 10_655_280},
        "source_hashes": {path.name: sha256_file(path) for path in raw_root.glob("*.csv")},
        "grain": "code_module, code_presentation, id_student",
        "student_vle_join_key": vle_key,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


def _verify_oulad_sources(protocol: dict) -> None:
    from src.studies.common.hashing import sha256_file

    for source_id, source in protocol["sources"].items():
        if source_id.startswith("oulad") and sha256_file(ROOT / source["path"]) != source["sha256"]:
            raise RuntimeError(f"Frozen source hash mismatch: {source_id}")


def command_prepare_oulad(args: argparse.Namespace) -> int:
    """Materialize cutoff-valid OULAD snapshots and grouped split manifests."""

    import pandas as pd

    from src.studies.common.hashing import sha256_file
    from src.studies.oulad.cohort import FORECASTS
    from src.studies.oulad.materialize import materialize_all, rebuild_derived_from_sequences
    from src.studies.oulad.splits import build_common_split_manifests

    protocol_path = args.protocol.resolve()
    output = args.output.resolve()
    protocol = _load_protocol(protocol_path)
    _verify_oulad_sources(protocol)
    completion = output / "manifests" / "materialization_complete.json"

    if args.rebuild_derived:
        materialization = rebuild_derived_from_sequences(output)
    elif args.resume and completion.exists() and json.loads(completion.read_text(encoding="utf-8")).get("status") == "PASS":
        materialization = {"status": "SKIPPED_ALREADY_PASS", "output": str(output)}
    else:
        materialization = materialize_all(ROOT / "data" / "raw", output, protocol)
        completion.parent.mkdir(parents=True, exist_ok=True)
        completion.write_text(
            json.dumps(
                {**materialization, "completed_at": datetime.now(timezone.utc).isoformat()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if materialization["status"] != "PASS":
            print(json.dumps({"materialization": materialization}, indent=2))
            return 1

    frames = {
        forecast: (
            pd.read_parquet(output / "cohorts" / f"{forecast}.parquet"),
            pd.read_parquet(output / "targets" / f"{forecast}.parquet"),
        )
        for forecast in FORECASTS
    }
    manifest, future, audit = build_common_split_manifests(
        frames, protocol["study_c"]["future_support"], seed=42
    )
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    split_path = manifest_dir / "split_manifest.csv"
    future_path = manifest_dir / "future_test_manifest.csv"
    audit_path = manifest_dir / "future_eligibility_audit.csv"
    manifest.to_csv(split_path, index=False)
    future.to_csv(future_path, index=False)
    audit.to_csv(audit_path, index=False)
    split_result = {
        "status": "PASS",
        "historical_records": int((manifest["role"] == "historical_development").sum()),
        "future_records": int((manifest["role"] == "future_candidate").sum()),
        "excluded_overlap_records": int((manifest["role"] == "excluded_future_student_overlap").sum()),
        "checksums": {
            "split_manifest": sha256_file(split_path),
            "future_test_manifest": sha256_file(future_path),
            "future_eligibility_audit": sha256_file(audit_path),
        },
    }
    (manifest_dir / "split_complete.json").write_text(
        json.dumps(split_result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"materialization": materialization, "splits": split_result}, indent=2))
    return 0


def _thesis_metrics(source: Path):
    import pandas as pd

    from src.common.model_display_names import get_display_name

    candidates = ["V3-MLF", "V3-A0F-ENS", "V3-P0-ENS", "V3-D0-ENS"]
    frame = pd.read_csv(source).set_index("candidate_id")
    missing = sorted(set(candidates) - set(frame.index))
    if missing:
        raise ValueError(f"Missing thesis figure candidates: {missing}")
    selected = frame.loc[candidates].reset_index()
    selected["display_name"] = selected.candidate_id.map(get_display_name)
    if selected.display_name.duplicated().any():
        raise ValueError("Figure display labels must be unique")
    return selected


def command_figures(args: argparse.Namespace) -> int:
    """Regenerate thesis-facing figures directly from frozen closure evidence."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = _thesis_metrics(source)
    colors = ["#2f6b9a", "#5b8f6a", "#7d6aa5", "#c06b3e"]

    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(frame.display_name, frame.macro_f1, color=colors)
    axis.set_ylabel("Macro-F1")
    axis.set_title("So sánh Macro-F1 trên OULAD")
    axis.set_ylim(0.80, 0.84)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, frame.macro_f1):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.0004, f"{value:.4f}", ha="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(output / "model_macro_f1_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.4))
    positions = np.arange(len(frame))
    width = 0.36
    axis.bar(positions - width / 2, frame.at_risk_precision, width, label="Risk Precision", color="#2f6b9a")
    axis.bar(positions + width / 2, frame.at_risk_recall, width, label="Risk Recall", color="#c06b3e")
    axis.set_xticks(positions, frame.display_name, rotation=18)
    axis.set_ylabel("Giá trị")
    axis.set_title("Precision và Recall cho nhóm có nguy cơ")
    axis.set_ylim(0.70, 0.87)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "model_precision_recall_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(frame.display_name, frame.pr_auc, color=colors)
    axis.set_ylabel("PR-AUC")
    axis.set_title("So sánh PR-AUC trên OULAD")
    axis.set_ylim(0.87, 0.90)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, frame.pr_auc):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.0003, f"{value:.4f}", ha="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(output / "model_pr_auc_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "source": source.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(source),
        "display_labels": frame.display_name.tolist(),
        "figures": [
            "model_macro_f1_comparison.png",
            "model_precision_recall_comparison.png",
            "model_pr_auc_comparison.png",
        ],
        "excluded_mixed_estimator_comparator": {
            "candidate_id": "V3-MLD",
            "reason": "Outer folds selected different estimator families; one algorithm label would be inaccurate.",
        },
        "metrics_copied_manually": False,
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "figures": 3, "source_sha256": manifest["source_sha256"]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Routine commands for the three-dataset student prediction thesis repository."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="Show official evidence bundles (no training).")
    status.set_defaults(handler=command_status)

    validate = commands.add_parser("validate", help="Validate official evidence (no training).")
    validate.set_defaults(handler=command_validate)

    figures = commands.add_parser("figures", help="Regenerate thesis figures from frozen evidence.")
    figures.add_argument("--source", type=Path, default=DEFAULT_FIGURE_SOURCE)
    figures.add_argument("--output", type=Path, default=DEFAULT_FIGURE_OUTPUT)
    figures.set_defaults(handler=command_figures)

    ingest = commands.add_parser("ingest", help="Ingest a UCI dataset into PostgreSQL.")
    ingest.add_argument("dataset", choices=["student-mat", "student-por"])
    ingest.set_defaults(handler=command_ingest)

    audit = commands.add_parser("audit-oulad", help="Audit raw OULAD files and lineage.")
    audit.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    audit.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / "oulad_release_audit.json",
    )
    audit.set_defaults(handler=command_audit_oulad)

    prepare = commands.add_parser(
        "prepare-oulad",
        help="Materialize OULAD snapshots and split manifests; does not train models.",
    )
    prepare.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    prepare.add_argument("--output", type=Path, default=DEFAULT_OULAD_OUTPUT)
    prepare.add_argument("--resume", action="store_true")
    prepare.add_argument("--rebuild-derived", action="store_true")
    prepare.set_defaults(handler=command_prepare_oulad)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
