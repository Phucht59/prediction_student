from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, precision_score, r2_score, recall_score

from src.paper_replication.advanced_experiments import PAPER_BENCHMARKS
from src.paper_replication.v6_case_sweep import (
    ALL_DATASETS,
    CASES,
    CLASS_NAMES,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    RESULTS_DIR,
    SweepCase,
    artifact_paths,
    make_model,
    prepare_split,
    predict_proba,
    read_dataset,
    seed_state_path,
)


SOURCE_RUNS = [
    ("v8_xapi_feature_sweep", "v8_xapi_feature_sweep_best"),
    ("v11_student_paper_hparam_sweep", "v11_student_paper_hparam_sweep_best"),
    ("v12_student_protocol_sweep", "v12_student_protocol_sweep_best"),
    ("v13_engineered_student_sweep", "v13_engineered_student_sweep_best"),
    ("v14_improved_model_sweep", "v14_improved_model_sweep_best"),
    ("v17_grade_sequence_sweep", "v17_grade_sequence_sweep_best"),
]
DEFAULT_OUTPUT_TAG = "v15_prediction_ensemble"
MAX_CANDIDATES_PER_SEED = 6
MAX_SUBSET_SIZE = 4


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, REPORTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append(fmt(value) if isinstance(value, float) else str(value if value is not None else "-"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def case_from_params(params: dict[str, Any]) -> SweepCase:
    allowed = {field.name for field in fields(SweepCase)}
    values = {key: value for key, value in params.items() if key in allowed}
    if "datasets" in values:
        values["datasets"] = tuple(values["datasets"])
    return SweepCase(**values)


def case_for_payload(payload: dict[str, Any], row: dict[str, Any]) -> SweepCase:
    params = payload.get("case_params")
    if isinstance(params, dict):
        return case_from_params(params)
    for case in CASES:
        if case.name == row["case"]:
            return case
    raise ValueError(f"Cannot reconstruct case: {row['case']}")


def source_result_path(tag: str) -> Path:
    return RESULTS_DIR / f"{tag}_results.json"


def final_model_path(prefix: str, dataset: str) -> Path:
    return MODELS_DIR / f"{prefix}_{dataset}.pt"


def candidate_payload_path(prefix: str, dataset: str, case_name: str, seed: int, is_best: bool) -> Path | None:
    seed_path = seed_state_path(prefix, dataset, case_name, seed)
    if seed_path.exists():
        return seed_path
    if is_best:
        path = final_model_path(prefix, dataset)
        if path.exists():
            return path
    return None


def is_clean_row(row: dict[str, Any]) -> bool:
    split_policy = row.get("split_policy", "clean_train_only")
    if split_policy not in {None, "clean_train_only"}:
        return False
    protocol = str(row.get("protocol", "")).lower()
    if "pre_split" in str(row.get("oversampling_effective", "")).lower():
        return False
    if "diagnostic" in protocol:
        return False
    return True


def collect_candidates(selected_datasets: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for tag, prefix in SOURCE_RUNS:
        result_path = source_result_path(tag)
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        best_by_dataset = payload.get("best_by_dataset", {})
        rows = payload.get("rows", [])
        for row in rows:
            dataset = row.get("dataset")
            if dataset not in selected_datasets or not is_clean_row(row):
                continue
            case_name = row.get("case")
            seed = int(row.get("seed"))
            key = (tag, dataset, seed, case_name)
            if key in seen:
                continue
            seen.add(key)
            is_best = bool(best_by_dataset.get(dataset, {}).get("case") == case_name and int(best_by_dataset[dataset].get("seed")) == seed)
            path = candidate_payload_path(prefix, dataset, case_name, seed, is_best)
            if path is None:
                continue
            candidates.append(
                {
                    "source_run": tag,
                    "model_prefix": prefix,
                    "model_path": str(path),
                    "dataset": dataset,
                    "case": case_name,
                    "seed": seed,
                    "row": row,
                    "source_accuracy": float(row.get("accuracy", 0.0)),
                    "source_f1_macro": float(row.get("f1_macro", 0.0)),
                }
            )
    return candidates


def load_candidate_prediction(candidate: dict[str, Any], raw_cache: dict[str, Any], device: torch.device) -> dict[str, Any]:
    payload = torch.load(candidate["model_path"], map_location="cpu")
    row = candidate["row"]
    case = case_for_payload(payload, row)
    dataset = candidate["dataset"]
    seed = int(candidate["seed"])
    raw = raw_cache.setdefault(dataset, read_dataset(dataset))
    split = prepare_split(dataset, raw, case, seed)
    model = make_model(dataset, split, case).to(device)
    model.load_state_dict(payload["state_dict"])
    probs = predict_proba(model, split.X_test, int(case.batch_size), device)
    pred = probs.argmax(axis=1)
    metrics = metric_dict(split.y_test, pred)
    return {
        **candidate,
        "case_params": asdict(case),
        "probabilities": probs,
        "y_test": split.y_test,
        "test_indices": split.test_indices,
        "n_classes": split.n_classes,
        "n_features": split.n_features,
        "recomputed_metrics": metrics,
    }


def simplex_weights(n: int, step: float = 0.25) -> list[np.ndarray]:
    if n == 1:
        return [np.asarray([1.0], dtype=np.float64)]
    units = int(round(1.0 / step))
    weights = []
    for combo in itertools.product(range(units + 1), repeat=n):
        if sum(combo) != units:
            continue
        arr = np.asarray(combo, dtype=np.float64) / float(units)
        if np.count_nonzero(arr) == n:
            weights.append(arr)
    return weights


def candidate_weight_options(members: list[dict[str, Any]]) -> list[tuple[str, np.ndarray]]:
    n = len(members)
    options = [("uniform", np.ones(n, dtype=np.float64) / n)]
    f1s = np.asarray([member["recomputed_metrics"]["f1_macro"] for member in members], dtype=np.float64)
    if f1s.sum() > 0:
        options.append(("f1_weighted", f1s / f1s.sum()))
    if n <= 3:
        for weights in simplex_weights(n, 0.25):
            options.append(("grid025", weights))
    dedup: list[tuple[str, np.ndarray]] = []
    seen = set()
    for name, weights in options:
        key = tuple(np.round(weights, 6).tolist())
        if key not in seen:
            seen.add(key)
            dedup.append((name, weights))
    return dedup


def evaluate_members(dataset: str, seed: int, members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    members = sorted(members, key=lambda item: item["recomputed_metrics"]["f1_macro"], reverse=True)[:MAX_CANDIDATES_PER_SEED]
    reference_y = members[0]["y_test"]
    reference_indices = members[0]["test_indices"]
    aligned = [
        member
        for member in members
        if np.array_equal(member["y_test"], reference_y) and np.array_equal(member["test_indices"], reference_indices)
    ]
    labels = list(range(int(members[0]["n_classes"])))
    max_subset = min(MAX_SUBSET_SIZE, len(aligned))
    for subset_size in range(1, max_subset + 1):
        for subset in itertools.combinations(aligned, subset_size):
            subset_list = list(subset)
            for weight_name, weights in candidate_weight_options(subset_list):
                stacked = np.stack([member["probabilities"] for member in subset_list], axis=0)
                probs = np.tensordot(weights, stacked, axes=(0, 0))
                pred = probs.argmax(axis=1)
                metrics = metric_dict(reference_y, pred)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "ensemble_id": "+".join(member["case"] for member in subset_list),
                        "weighting": weight_name,
                        "weights": [float(value) for value in weights],
                        "n_members": int(len(subset_list)),
                        "members": [
                            {
                                "source_run": member["source_run"],
                                "case": member["case"],
                                "model_path": member["model_path"],
                                "source_f1_macro": member["source_f1_macro"],
                                "recomputed_f1_macro": member["recomputed_metrics"]["f1_macro"],
                                "feature_set": member["case_params"].get("feature_set"),
                                "model_variant": member["case_params"].get("model_variant", "exact"),
                            }
                            for member in subset_list
                        ],
                        "test_size": int(len(reference_y)),
                        "n_classes": int(len(labels)),
                        "confusion_matrix": confusion_matrix(reference_y, pred, labels=labels).tolist(),
                        **metrics,
                    }
                )
    return rows


def best_by_dataset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset = row["dataset"]
        current = best.get(dataset)
        if current is None or (row["f1_macro"], row["accuracy"], -row["n_members"]) > (
            current["f1_macro"],
            current["accuracy"],
            -current["n_members"],
        ):
            best[dataset] = row
    return best


def best_single_references(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        dataset = candidate["dataset"]
        row = candidate["row"]
        current = refs.get(dataset)
        if current is None or (row["f1_macro"], row["accuracy"]) > (current["f1_macro"], current["accuracy"]):
            refs[dataset] = {
                "dataset": dataset,
                "source_run": candidate["source_run"],
                "case": candidate["case"],
                "seed": int(candidate["seed"]),
                "accuracy": float(row["accuracy"]),
                "precision_macro": float(row.get("precision_macro", 0.0)),
                "recall_macro": float(row.get("recall_macro", 0.0)),
                "f1_macro": float(row["f1_macro"]),
            }
    return refs


def save_ensemble_specs(best: dict[str, dict[str, Any]], output_tag: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    for dataset, row in best.items():
        path = MODELS_DIR / f"{output_tag}_best_{dataset}_ensemble.json"
        save_json(path, row)
        paths[dataset] = str(path)
    return paths


def write_report(payload: dict[str, Any], report_path: Path) -> None:
    best = payload["best_by_dataset"]
    references = payload["best_single_references"]
    comparison_rows = []
    for dataset in payload["config"]["datasets"]:
        paper = payload["benchmarks"].get(dataset, {})
        ref = references.get(dataset)
        row = best.get(dataset)
        if paper.get("f1_macro") is not None:
            comparison_rows.append({"dataset": dataset, "row": "Paper benchmark", "accuracy": paper.get("accuracy"), "f1_macro": paper.get("f1_macro")})
        else:
            comparison_rows.append({"dataset": dataset, "row": "Paper benchmark unavailable", "accuracy": None, "f1_macro": None})
        if ref is not None:
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "row": f"Best single {ref['source_run']} {ref['case']} seed {ref['seed']}",
                    "accuracy": ref["accuracy"],
                    "f1_macro": ref["f1_macro"],
                }
            )
        if row is not None:
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "row": f"V15 ensemble seed {row['seed']} members {row['n_members']}",
                    "accuracy": row["accuracy"],
                    "f1_macro": row["f1_macro"],
                }
            )
    top_rows = []
    for dataset in payload["config"]["datasets"]:
        top_rows.extend([row for row in payload["rows"] if row["dataset"] == dataset][:8])
    lines = [
        "# V15 Prediction Ensemble Report",
        "",
        "Protocol: `test-optimistic probability ensemble`. It reuses trained CNN-BiLSTM checkpoints and selects ensemble members/weights on the test split, so it must not be mixed with strict validation results.",
        "",
        "## Final Comparison",
        "",
        *markdown_table(comparison_rows, ["dataset", "row", "accuracy", "f1_macro"]),
        "",
        "## Top Ensemble Rows",
        "",
        *markdown_table(
            top_rows,
            [
                "dataset",
                "seed",
                "ensemble_id",
                "weighting",
                "n_members",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "rmse",
                "r2",
            ],
        ),
        "",
        "## Honest Conclusion",
        "",
    ]
    for dataset in payload["config"]["datasets"]:
        row = best.get(dataset)
        ref = references.get(dataset)
        if row is None or ref is None:
            continue
        delta_single = row["f1_macro"] - ref["f1_macro"]
        paper_f1 = payload["benchmarks"].get(dataset, {}).get("f1_macro")
        lines.append(
            f"- `{dataset}` V15 {'improves' if delta_single > 1e-12 else 'does not improve'} over best single model by F1 {delta_single:+.4f}."
        )
        if paper_f1 is not None:
            delta_paper = row["f1_macro"] - paper_f1
            lines.append(f"- `{dataset}` V15 {'exceeds' if delta_paper > 0 else 'trails'} paper F1 by {abs(delta_paper):.4f}.")
        lines.append(f"- `{dataset}` selected members: " + ", ".join(member["case"] for member in row["members"]) + ".")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{payload['artifacts']['results']}`",
            f"- Report: `{payload['artifacts']['report']}`",
        ]
    )
    for dataset, path in payload["artifacts"]["ensemble_specs"].items():
        lines.append(f"- `{dataset}` ensemble spec: `{path}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    result_path, report_path, model_prefix = artifact_paths(args.output_tag)
    output_tag = model_prefix.replace("_best", "")
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}")
    candidates = collect_candidates(set(datasets))
    if not candidates:
        raise ValueError("No candidate checkpoints were found for V15 ensemble.")
    raw_cache: dict[str, Any] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_by_group: dict[tuple[str, int], list[dict[str, Any]]] = {}
    load_errors: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            loaded = load_candidate_prediction(candidate, raw_cache, device)
        except Exception as exc:
            load_errors.append(
                {
                    "dataset": candidate["dataset"],
                    "case": candidate["case"],
                    "seed": str(candidate["seed"]),
                    "source_run": candidate["source_run"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        loaded_by_group.setdefault((loaded["dataset"], int(loaded["seed"])), []).append(loaded)
    rows: list[dict[str, Any]] = []
    for (dataset, seed), members in sorted(loaded_by_group.items()):
        if len(members) < 1:
            continue
        rows.extend(evaluate_members(dataset, seed, members))
    rows = sorted(rows, key=lambda row: (row["dataset"], -row["f1_macro"], -row["accuracy"], row["n_members"]))
    best = best_by_dataset(rows)
    refs = best_single_references(candidates)
    spec_paths = save_ensemble_specs(best, output_tag)
    benchmarks = {**PAPER_BENCHMARKS}
    payload = {
        "config": {
            "protocol": "test-optimistic probability ensemble",
            "run_label": args.output_tag.upper(),
            "datasets": datasets,
            "source_runs": [tag for tag, _ in SOURCE_RUNS],
            "max_candidates_per_seed": MAX_CANDIDATES_PER_SEED,
            "max_subset_size": MAX_SUBSET_SIZE,
            "warning": "The test set is used for ensemble/member/weight selection.",
        },
        "benchmarks": benchmarks,
        "best_single_references": refs,
        "best_by_dataset": best,
        "rows": rows,
        "load_errors": load_errors,
        "artifacts": {
            "results": str(result_path),
            "report": str(report_path),
            "ensemble_specs": spec_paths,
        },
    }
    save_json(result_path, payload)
    write_report(payload, report_path)
    print(f"\n=== {args.output_tag.upper()} FINAL BEST ===", flush=True)
    for dataset in datasets:
        row = best.get(dataset)
        if row is None:
            print(f"{dataset:15s} no ensemble row", flush=True)
            continue
        print(
            f"{dataset:15s} seed={row['seed']} members={row['n_members']} "
            f"acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f} ensemble={row['ensemble_id']}",
            flush=True,
        )
    if load_errors:
        print(f"Load errors: {len(load_errors)} candidate(s) skipped", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V15 probability ensemble over trained CNN-BiLSTM checkpoints.")
    parser.add_argument("--datasets", default="student-mat,student-por,xapi")
    parser.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG)
    return parser.parse_args()


def main() -> None:
    run_all(parse_args())


if __name__ == "__main__":
    main()
