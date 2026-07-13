"""Materialize immutable legacy-V1 and deterministic protocol-V2 metadata.

This command does not train a model and never modifies an existing artifact.
It derives only lineage metadata from the already frozen evidence bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT_DIR
from src.evaluation.protocol import file_checksum, semantic_checksum, source_record_identity


FINAL_RUN_ID = "a2945d79-9845-4979-b148-159f4853eca3"
DB_VERIFICATION_RUN_ID = "5a0b5041-5216-4a48-9e46-b0c16ab14866"
SELECTION_RUN_ID = "nested-full-20260710"
DATASET_VERSION_ID = 1
DATASET_CHECKSUM = "e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80"
HISTORICAL_SOURCE_COMMIT = "74e43fc276d6abfff3829d415315d30d8240da33"
CREATED_AT = "2026-07-13T00:00:00+00:00"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_immutable(path: Path, payload: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def materialize() -> list[Path]:
    final_dir = ROOT_DIR / "artifacts" / "final" / f"final-{FINAL_RUN_ID}"
    selection_dir = ROOT_DIR / "artifacts" / "model_selection" / SELECTION_RUN_ID
    oof_rows = read_csv(selection_dir / "outer_oof_predictions.csv")
    observed_rows = read_csv(final_dir / "locked_test_predictions.csv")
    development = sorted(
        [{"source_row_number": int(row["source_row_number"]), "true_label": int(row["true_label"])} for row in oof_rows],
        key=lambda row: row["source_row_number"],
    )
    observed = sorted(
        [{"source_row_number": int(row["__source_row_number"]), "true_label": int(row["True_Label"])} for row in observed_rows],
        key=lambda row: row["source_row_number"],
    )
    if len(development) != 316 or len(observed) != 79:
        raise ValueError("Frozen artifacts do not contain the expected 316/79 cohort sizes.")
    if {row["source_row_number"] for row in development} & {row["source_row_number"] for row in observed}:
        raise ValueError("Historical development and observed-holdout memberships overlap.")

    legacy_dir = ROOT_DIR / "artifacts" / "legacy_v1"
    legacy_membership = {
        "legacy_version": "legacy_v1",
        "dataset_version_id": DATASET_VERSION_ID,
        "identity_contract": "source_record_identity = dataset_version_id + immutable source_row_number; source row is never used alone",
        "development_records": [
            {**row, "source_record_identity": source_record_identity(DATASET_VERSION_ID, row["source_row_number"]), "role": "development"}
            for row in development
        ],
        "legacy_heldout_observed_records": [
            {**row, "source_record_identity": source_record_identity(DATASET_VERSION_ID, row["source_row_number"]), "role": "legacy_heldout_observed"}
            for row in observed
        ],
    }
    legacy_membership["manifest_checksum"] = semantic_checksum(legacy_membership)
    membership_path = legacy_dir / "legacy_split_membership.json"
    write_immutable(membership_path, json.dumps(legacy_membership, ensure_ascii=False, indent=2) + "\n")

    selected_config = selection_dir / "selected_config.json"
    prediction_path = final_dir / "locked_test_predictions.csv"
    metric_path = final_dir / "classification_report.json"
    recommendation_path = final_dir / "recommendation_evaluation.json"
    legacy_manifest = {
        "legacy_version": "legacy_v1",
        "created_at": CREATED_AT,
        "scientific_role": "historical_observed_holdout_only",
        "source_commit": HISTORICAL_SOURCE_COMMIT,
        "dataset_name": "student-mat",
        "dataset_checksum": DATASET_CHECKSUM,
        "dataset_version_id": DATASET_VERSION_ID,
        "selection_run_id": SELECTION_RUN_ID,
        "final_run_id": FINAL_RUN_ID,
        "database_verification_run": DB_VERIFICATION_RUN_ID,
        "selected_config_path": str(selected_config.relative_to(ROOT_DIR).as_posix()),
        "selected_config_checksum": file_checksum(selected_config),
        "model_checkpoint_path": None,
        "model_checkpoint_checksum": None,
        "model_checkpoint_status": "No serialized checkpoint was retained in the frozen final evidence; model_checksums.json explicitly records an empty checkpoint map.",
        "development_split_manifest_path": str(membership_path.relative_to(ROOT_DIR).as_posix()),
        "development_split_checksum": legacy_membership["manifest_checksum"],
        "current_79_split_manifest_path": str(membership_path.relative_to(ROOT_DIR).as_posix()),
        "current_79_split_checksum": legacy_membership["manifest_checksum"],
        "current_79_scientific_role": "legacy_heldout_observed",
        "current_79_record_ids": [source_record_identity(DATASET_VERSION_ID, row["source_row_number"]) for row in observed],
        "prediction_artifact_path": str(prediction_path.relative_to(ROOT_DIR).as_posix()),
        "prediction_checksum": file_checksum(prediction_path),
        "metric_artifact_path": str(metric_path.relative_to(ROOT_DIR).as_posix()),
        "metric_checksum": file_checksum(metric_path),
        "recommendation_artifact_path": str(recommendation_path.relative_to(ROOT_DIR).as_posix()),
        "recommendation_checksum": file_checksum(recommendation_path),
        "recommendation_policy_version": "student_mat_rule_policy_v3",
        "python_version": "3.10.8",
        "package_versions": {"numpy": "2.2.6", "pandas": "2.3.3", "scikit-learn": "1.7.2", "torch": "2.12.0", "optuna": "4.8.0", "imbalanced-learn": "0.14.1"},
        "preservation": "This manifest aliases existing artifacts. No legacy artifact was moved, renamed, or altered.",
    }
    legacy_manifest["manifest_checksum"] = semantic_checksum(legacy_manifest)
    legacy_manifest_path = legacy_dir / "legacy_manifest.json"
    write_immutable(legacy_manifest_path, json.dumps(legacy_manifest, ensure_ascii=False, indent=2) + "\n")
    write_immutable(legacy_dir / "legacy_manifest.sha256", file_checksum(legacy_manifest_path) + "  legacy_manifest.json\n")

    protocol_dir = ROOT_DIR / "artifacts" / "protocol_v2"
    labels = np.asarray([row["true_label"] for row in development], dtype=int)
    indices = np.arange(len(development))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    records = [
        {**row, "source_record_identity": source_record_identity(DATASET_VERSION_ID, row["source_row_number"])}
        for row in development
    ]
    assignments = []
    for fold, (train_idx, validation_idx) in enumerate(splitter.split(indices, labels)):
        for index in train_idx:
            assignments.append({"source_record_identity": records[int(index)]["source_record_identity"], "outer_fold": fold, "outer_role": "train", "inner_fold": None, "split_seed": 42, "stratification_target": "G3_3class"})
        for index in validation_idx:
            assignments.append({"source_record_identity": records[int(index)]["source_record_identity"], "outer_fold": fold, "outer_role": "validation", "inner_fold": None, "split_seed": 42, "stratification_target": "G3_3class"})
    fold_manifest = {
        "protocol_version": "scientific_protocol_v2",
        "created_at": CREATED_AT,
        "dataset_name": "student-mat",
        "dataset_version_id": DATASET_VERSION_ID,
        "dataset_checksum": DATASET_CHECKSUM,
        "identity_contract": "source_record_identity = dataset_version_id + immutable source_row_number; database record_id may be joined at persistence time",
        "cohort_role": "legacy_development_only",
        "outer_folds": 5,
        "split_seed": 42,
        "stratification_target": "G3_3class",
        "development_records": records,
        "assignments": assignments,
    }
    fold_manifest["manifest_checksum"] = semantic_checksum(fold_manifest)
    fold_path = protocol_dir / "student_mat_development_outer_folds.json"
    write_immutable(fold_path, json.dumps(fold_manifest, ensure_ascii=False, indent=2) + "\n")
    csv_rows = []
    by_id = {record["source_record_identity"]: record for record in records}
    for assignment in assignments:
        csv_rows.append({"record_id": assignment["source_record_identity"], "source_row_number": by_id[assignment["source_record_identity"]]["source_row_number"], "dataset_version_id": DATASET_VERSION_ID, **assignment})
    csv_path = protocol_dir / "student_mat_development_outer_folds.csv"
    if csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {csv_path}")
    protocol_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_immutable(protocol_dir / "student_mat_development_outer_folds.sha256", f"{file_checksum(fold_path)}  {fold_path.name}\n{file_checksum(csv_path)}  {csv_path.name}\n")
    return [legacy_manifest_path, fold_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify existing checksum sidecars without writing files.")
    args = parser.parse_args()
    if args.verify:
        for sidecar in (ROOT_DIR / "artifacts" / "legacy_v1" / "legacy_manifest.sha256", ROOT_DIR / "artifacts" / "protocol_v2" / "student_mat_development_outer_folds.sha256"):
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                expected, name = line.split(maxsplit=1)
                target = sidecar.parent / name
                if file_checksum(target) != expected:
                    raise SystemExit(f"Checksum mismatch: {target}")
        print("Protocol V2 immutable artifacts verified.")
        return
    print("\n".join(str(path) for path in materialize()))


if __name__ == "__main__":
    main()
