from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (  # noqa: E402
    SCENARIOS,
    apply_student_feature_engineering,
    build_preprocessor,
    create_targets,
    get_excluded_columns,
    get_feature_columns,
    get_output_feature_names,
    load_student_dataset,
    run_leakage_checks,
    save_processed_artifacts,
    split_data,
    transform_splits,
)


DATASETS = {
    "student-mat": PROJECT_ROOT / "data" / "raw" / "student-mat.csv",
    "student-por": PROJECT_ROOT / "data" / "raw" / "student-por.csv",
    "student-combined": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare processed Student Performance datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to process.",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS],
        default="all",
        help="Prediction scenario to process.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified splits.")
    return parser.parse_args()


def selected_items(selection: str, values: list[str]) -> list[str]:
    return values if selection == "all" else [selection]


def remove_deprecated_scenario_outputs(dataset_names: list[str]) -> None:
    for dataset_name in dataset_names:
        output_dir = PROJECT_ROOT / "data" / "processed" / dataset_name / "early"
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Removed deprecated processed scenario output: {output_dir}")


def load_student_combined() -> tuple:
    mat_path = PROJECT_ROOT / "data" / "raw" / "student-mat.csv"
    por_path = PROJECT_ROOT / "data" / "raw" / "student-por.csv"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {mat_path}")
    if not por_path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {por_path}")

    mat = load_student_dataset(str(mat_path))
    por = load_student_dataset(str(por_path))
    mat_columns = list(mat.columns)
    por_columns = list(por.columns)
    only_mat = [column for column in mat_columns if column not in por_columns]
    only_por = [column for column in por_columns if column not in mat_columns]
    common_columns = [column for column in mat_columns if column in set(por_columns)]
    if only_mat or only_por:
        print(f"  warning: student-combined aligning by intersection columns.")
        print(f"  columns only in student-mat: {only_mat}")
        print(f"  columns only in student-por: {only_por}")

    mat = mat[common_columns].copy()
    por = por[common_columns].copy()
    mat["dataset_id"] = 0
    por["dataset_id"] = 1
    combined = pd.concat([mat, por], ignore_index=True)
    metadata = {
        "student_mat_rows": int(mat.shape[0]),
        "student_por_rows": int(por.shape[0]),
        "combined_rows": int(combined.shape[0]),
        "columns_only_student_mat": only_mat,
        "columns_only_student_por": only_por,
        "combined_columns": common_columns + ["dataset_id"],
    }
    return combined, metadata, f"{mat_path.name}+{por_path.name}"


def process_dataset_scenario(dataset_name: str, source_path: Path | None, scenario: str, seed: int) -> dict:
    print(f"\nProcessing dataset={dataset_name}, scenario={scenario}")
    combined_metadata = {}
    if dataset_name == "student-combined":
        data, combined_metadata, source_file = load_student_combined()
        print(
            "  combined rows: student-mat={}, student-por={}, total={}".format(
                combined_metadata["student_mat_rows"],
                combined_metadata["student_por_rows"],
                combined_metadata["combined_rows"],
            )
        )
    else:
        if source_path is None:
            raise ValueError(f"source_path is required for dataset={dataset_name}")
        data = load_student_dataset(str(source_path))
        source_file = source_path
    data = create_targets(data)
    data, feature_engineering_metadata = apply_student_feature_engineering(data, scenario)
    print(f"  engineered features: {feature_engineering_metadata['engineered_features_created']}")
    for warning in feature_engineering_metadata["feature_engineering_warnings"]:
        print(f"  warning: {warning}")
    feature_columns = get_feature_columns(data, scenario)
    excluded_columns = get_excluded_columns(data, feature_columns)

    split_result = split_data(data, feature_columns, seed=seed)
    preprocessor = build_preprocessor(split_result["X_train_raw"])
    X_train, X_val, X_test = transform_splits(
        preprocessor,
        split_result["X_train_raw"],
        split_result["X_val_raw"],
        split_result["X_test_raw"],
    )
    feature_names = get_output_feature_names(preprocessor)
    leakage_checks = run_leakage_checks(feature_columns, feature_names, scenario)
    if not leakage_checks["passed"]:
        raise RuntimeError(
            f"Leakage check failed for dataset={dataset_name}, scenario={scenario}: "
            f"{leakage_checks}"
        )

    output_dir = PROJECT_ROOT / "data" / "processed" / dataset_name / scenario
    metadata = save_processed_artifacts(
        dataset_name=dataset_name,
        source_file=source_file,
        scenario=scenario,
        random_seed=seed,
        output_dir=output_dir,
        full_df=data,
        split_result=split_result,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        raw_feature_columns=feature_columns,
        excluded_columns=excluded_columns,
        leakage_checks=leakage_checks,
        feature_engineering_metadata={
            **feature_engineering_metadata,
            "student_combined": combined_metadata,
        },
    )

    print(f"Rows: train={metadata['n_train']}, val={metadata['n_val']}, test={metadata['n_test']}")
    print(f"Features: raw={metadata['n_features_raw']}, processed={metadata['n_features_processed']}")
    print(f"Class train: {metadata['class_distribution_train']}")
    print(f"Class val: {metadata['class_distribution_val']}")
    print(f"Class test: {metadata['class_distribution_test']}")
    print(f"Class total: {metadata['class_distribution_total']}")
    print(f"Leakage check: {metadata['leakage_checks']}")
    print(f"Saved to: {output_dir}")
    return metadata


def main() -> int:
    args = parse_args()
    dataset_names = selected_items(args.dataset, list(DATASETS.keys()))
    scenarios = selected_items(args.scenario, list(SCENARIOS))
    remove_deprecated_scenario_outputs(dataset_names)

    for dataset_name in dataset_names:
        source_path = DATASETS[dataset_name]
        if source_path is not None and not source_path.exists():
            raise FileNotFoundError(f"Missing raw dataset: {source_path}")
        for scenario in scenarios:
            process_dataset_scenario(dataset_name, source_path, scenario, args.seed)

    print("\nAll requested preprocessing tasks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
