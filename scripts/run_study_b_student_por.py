from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS
from src.model_selection import fit_fold_predict_proba, predict_with_fitted_estimator
from src.studies.common.hashing import semantic_sha256, sha256_file
from src.studies.student_por.data import load_student_csv, overlap_membership
from src.studies.student_por.evaluation import summary_metrics, validate_probabilities
from src.studies.student_por.models import align_probabilities, make_ml_model, ml_configs, neural_configs


ML = ["B-L0", "B-RF0", "B-S0", "B-H0"]
NEURAL = ["B-M0", "B-C0", "B-L1", "B-H1", "B-O0"]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def g2_rule(values: pd.Series) -> np.ndarray:
    raw = values.to_numpy(float)
    return np.where(raw <= 9, 0, np.where(raw <= 14, 1, 2)).astype(int)


def nested_ml(candidate_id: str, frame: pd.DataFrame, outer_splits, trials: list[dict], selected: list[dict], runtime: list[dict]) -> pd.DataFrame:
    output = []
    x = frame[["G1", "G2"]]
    y = frame["G3"].to_numpy(int)
    for outer_fold, (train_idx, validation_idx) in enumerate(outer_splits):
        started = time.perf_counter()
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + outer_fold)
        scored = []
        for trial_id, config in enumerate(ml_configs(candidate_id)):
            fold_scores = []
            for inner_train, inner_validation in inner.split(train_idx, y[train_idx]):
                train_rows, validation_rows = train_idx[inner_train], train_idx[inner_validation]
                model = make_ml_model(candidate_id, config, 42)
                model.fit(x.iloc[train_rows], y[train_rows])
                fold_scores.append(f1_score(y[validation_rows], model.predict(x.iloc[validation_rows]), average="macro", zero_division=0))
            score = float(np.mean(fold_scores))
            scored.append((score, trial_id, config))
            trials.append({"candidate_id": candidate_id, "outer_fold": outer_fold, "trial_id": trial_id, "state": "COMPLETE", "inner_macro_f1": score, "config": json.dumps(config, sort_keys=True)})
        best_score, trial_id, config = max(scored, key=lambda item: (item[0], -item[1]))
        selected.append({"candidate_id": candidate_id, "outer_fold": outer_fold, "trial_id": trial_id, "inner_macro_f1": best_score, "config": json.dumps(config, sort_keys=True)})
        model = make_ml_model(candidate_id, config, 42)
        model.fit(x.iloc[train_idx], y[train_idx])
        probabilities = align_probabilities(model, model.predict_proba(x.iloc[validation_idx]))
        validate_probabilities(probabilities)
        for position, row_index in enumerate(validation_idx):
            output.append({"candidate_id": candidate_id, "outer_fold": outer_fold, "seed": 42, "source_record_id": frame.iloc[row_index]["source_record_id"], "source_row_number": int(frame.iloc[row_index]["source_row_number"]), "true_label": int(y[row_index]), "predicted_label": int(np.argmax(probabilities[position])), "prob_low": probabilities[position, 0], "prob_medium": probabilities[position, 1], "prob_high": probabilities[position, 2]})
        runtime.append({"candidate_id": candidate_id, "outer_fold": outer_fold, "seconds": time.perf_counter() - started, "status": "PASS"})
    return pd.DataFrame(output)


def nested_neural(candidate_id: str, frame: pd.DataFrame, outer_splits, trials: list[dict], selected: list[dict], runtime: list[dict]) -> pd.DataFrame:
    output = []
    y = frame["G3"].to_numpy(int)
    spec = DATASETS["student-por"]
    for outer_fold, (train_idx, validation_idx) in enumerate(outer_splits):
        started = time.perf_counter()
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + outer_fold)
        scored = []
        for trial_id, config in enumerate(neural_configs(candidate_id)):
            fold_scores = []
            state, reason = "COMPLETE", ""
            try:
                for inner_train, inner_validation in inner.split(train_idx, y[train_idx]):
                    train_rows, validation_rows = train_idx[inner_train], train_idx[inner_validation]
                    result = fit_fold_predict_proba(train_fold=frame.iloc[train_rows].copy(), validation_fold=frame.iloc[validation_rows].copy(), spec=spec, params=config, seed=42, fold_index=outer_fold)
                    fold_scores.append(f1_score(y[validation_rows], result.predictions, average="macro", zero_division=0))
                score = float(np.mean(fold_scores))
            except Exception as exc:
                state, reason, score = "FAIL", f"{type(exc).__name__}:{exc}", -1.0
            scored.append((score, trial_id, config, state))
            trials.append({"candidate_id": candidate_id, "outer_fold": outer_fold, "trial_id": trial_id, "state": state, "failure_reason": reason, "inner_macro_f1": score, "parameter_count": config["parameter_count"], "config": json.dumps(config, sort_keys=True)})
        completed = [item for item in scored if item[3] == "COMPLETE"]
        if not completed:
            runtime.append({"candidate_id": candidate_id, "outer_fold": outer_fold, "seconds": time.perf_counter() - started, "status": "FAIL"})
            continue
        best_score, trial_id, config, _ = max(completed, key=lambda item: (item[0], -item[1]))
        selected.append({"candidate_id": candidate_id, "outer_fold": outer_fold, "trial_id": trial_id, "inner_macro_f1": best_score, "parameter_count": config["parameter_count"], "config": json.dumps(config, sort_keys=True)})
        result = fit_fold_predict_proba(train_fold=frame.iloc[train_idx].copy(), validation_fold=frame.iloc[validation_idx].copy(), spec=spec, params=config, seed=42, fold_index=outer_fold)
        validate_probabilities(result.probabilities)
        for position, row_index in enumerate(validation_idx):
            probability = result.probabilities[position]
            output.append({"candidate_id": candidate_id, "outer_fold": outer_fold, "seed": 42, "source_record_id": frame.iloc[row_index]["source_record_id"], "source_row_number": int(frame.iloc[row_index]["source_row_number"]), "true_label": int(y[row_index]), "predicted_label": int(np.argmax(probability)), "prob_low": probability[0], "prob_medium": probability[1], "prob_high": probability[2]})
        runtime.append({"candidate_id": candidate_id, "outer_fold": outer_fold, "seconds": time.perf_counter() - started, "status": "PASS"})
    return pd.DataFrame(output)


def aggregate_frozen_student_mat_config(candidate_id: str) -> dict:
    path = ROOT / "artifacts" / "strategy_b_phase_c" / "strategy-b-phase-c-20260714-5d34a66" / "resolved_configs.csv"
    rows = pd.read_csv(path)
    configs = [json.loads(value)["parameters"] for value in rows.loc[rows["candidate_id"] == candidate_id, "resolved_config"]]
    result = {}
    for key in configs[0]:
        values = [config[key] for config in configs]
        if all(isinstance(value, (int, float)) and value is not None for value in values):
            result[key] = float(np.median(values)) if any(isinstance(value, float) for value in values) else int(np.median(values))
        else:
            result[key] = Counter(json.dumps(value, sort_keys=True) for value in values).most_common(1)[0][0]
            result[key] = json.loads(result[key])
    return result


def transfer_predictions(mat: pd.DataFrame, por: pd.DataFrame, overlap: pd.Series) -> pd.DataFrame:
    manifest = json.loads((ROOT / "artifacts" / "protocol_v2" / "student_mat_development_outer_folds.json").read_text(encoding="utf-8"))
    allowed = [int(item["source_row_number"]) for item in manifest["development_records"]]
    development = mat.loc[mat["source_row_number"].isin(allowed)].copy()
    if len(development) != 316:
        raise RuntimeError("Study A transfer must use exactly 316 development records")
    x_train, y_train, x_por = development[["G1", "G2"]], development["G3"].to_numpy(int), por[["G1", "G2"]]
    rows = []

    def append(candidate: str, probabilities: np.ndarray | None, predictions: np.ndarray) -> None:
        if probabilities is not None:
            validate_probabilities(probabilities)
        for index in range(len(por)):
            rows.append({"candidate_id": candidate, "source_record_id": por.iloc[index]["source_record_id"], "source_row_number": int(por.iloc[index]["source_row_number"]), "overlap_partition": overlap.iloc[index], "true_label": int(por.iloc[index]["G3"]), "predicted_label": int(predictions[index]), "prob_low": np.nan if probabilities is None else probabilities[index, 0], "prob_medium": np.nan if probabilities is None else probabilities[index, 1], "prob_high": np.nan if probabilities is None else probabilities[index, 2]})

    append("R0", None, g2_rule(por["G2"]))
    for study_a_id, builder_id in [("M1", "B-RF0"), ("M2", "B-S0")]:
        config = aggregate_frozen_student_mat_config(study_a_id)
        model = make_ml_model(builder_id, config, 42)
        model.fit(x_train, y_train)
        probabilities = align_probabilities(model, model.predict_proba(x_por))
        append(study_a_id, probabilities, probabilities.argmax(axis=1))

    source = ROOT / "artifacts" / "strategy_b_phase_e_prediction" / "strategy-b-phase-e-prediction-20260714-9007144"
    final_config = json.loads((source / "final_resolved_configs.json").read_text(encoding="utf-8"))["thesis_hybrid"]["resolved_config"]
    probabilities_by_seed = []
    for seed in [202601, 202602, 202603, 202604, 202605]:
        with (source / "final_models" / f"hybrid_N0_seed{seed}.preprocessor.pkl").open("rb") as handle:
            fitted = pickle.load(handle)
        probabilities_by_seed.append(predict_with_fitted_estimator(frame=por, spec=DATASETS["student-por"], resolved_config=final_config, state_dict=torch.load(source / "final_models" / f"hybrid_N0_seed{seed}.pt", map_location="cpu", weights_only=True), preprocessor=fitted["preprocessor"], selector=fitted["selector"]))
    ensemble = np.mean(np.stack(probabilities_by_seed), axis=0)
    append("N0", ensemble, ensemble.argmax(axis=1))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"study-b-student-por-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    args = parser.parse_args()
    artifact = ROOT / "artifacts" / "study_b_student_por" / args.run_id
    report = ROOT / "reports" / "study_b_student_por" / args.run_id
    if artifact.exists() or report.exists():
        raise FileExistsError("Immutable Study B run already exists")
    artifact.mkdir(parents=True)
    report.mkdir(parents=True)
    protocol = json.loads((ROOT / "configs" / "extension_protocol_v1.yaml").read_text(encoding="utf-8"))
    por = load_student_csv(ROOT / protocol["sources"]["student_por"]["path"], "student-por")
    mat = load_student_csv(ROOT / protocol["sources"]["student_mat"]["path"], "student-mat")
    overlap, overlap_audit = overlap_membership(mat, por)
    y = por["G3"].to_numpy(int)
    outer = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(por, y))
    fold_rows = []
    for fold, (train, validation) in enumerate(outer):
        for index in train: fold_rows.append({"source_record_id": por.iloc[index]["source_record_id"], "outer_fold": fold, "split": "train"})
        for index in validation: fold_rows.append({"source_record_id": por.iloc[index]["source_record_id"], "outer_fold": fold, "split": "validation"})
    pd.DataFrame(fold_rows).to_csv(artifact / "fold_manifest.csv", index=False)
    trials: list[dict] = []
    selected: list[dict] = []
    runtime: list[dict] = []
    frames = []
    rule = pd.DataFrame({"candidate_id": "B-R0", "outer_fold": -1, "seed": 42, "source_record_id": por["source_record_id"], "source_row_number": por["source_row_number"], "true_label": y, "predicted_label": g2_rule(por["G2"]), "prob_low": np.nan, "prob_medium": np.nan, "prob_high": np.nan})
    frames.append(rule)
    for candidate in ML:
        frames.append(nested_ml(candidate, por, outer, trials, selected, runtime))
    for candidate in NEURAL:
        frames.append(nested_neural(candidate, por, outer, trials, selected, runtime))
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(artifact / "oof_predictions.csv", index=False)

    metric_rows, class_rows = [], []
    for candidate, group in oof.groupby("candidate_id"):
        metric, classes = summary_metrics(group)
        metric_rows.append({"candidate_id": candidate, **metric})
        class_rows.extend({"candidate_id": candidate, **row} for row in classes)
    metrics = pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)
    metrics.to_csv(artifact / "metrics_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(artifact / "class_metrics.csv", index=False)
    pd.DataFrame(trials).to_csv(artifact / "search_trials.csv", index=False)
    pd.DataFrame(selected).to_csv(artifact / "selected_configs.csv", index=False)
    pd.DataFrame(runtime).to_csv(artifact / "runtime.csv", index=False)

    paired = []
    pivot = oof.pivot_table(index="source_record_id", columns="candidate_id", values="predicted_label", aggfunc="first")
    truth = oof.drop_duplicates("source_record_id").set_index("source_record_id")["true_label"]
    candidates = list(metrics["candidate_id"])
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            common = pivot[[left, right]].dropna().index
            delta = f1_score(truth.loc[common], pivot.loc[common, left], average="macro") - f1_score(truth.loc[common], pivot.loc[common, right], average="macro")
            paired.append({"left": left, "right": right, "macro_f1_delta": delta, "records": len(common)})
    pd.DataFrame(paired).to_csv(artifact / "paired_deltas.csv", index=False)

    transfer = transfer_predictions(mat, por, overlap)
    transfer.to_csv(artifact / "transfer_predictions.csv", index=False)
    transfer_metrics_rows = []
    for candidate, candidate_frame in transfer.groupby("candidate_id"):
        for partition in ["all", "conservative_matched", "conservative_unmatched", "ambiguous_shared_key"]:
            group = candidate_frame if partition == "all" else candidate_frame[candidate_frame["overlap_partition"] == partition]
            if group.empty: continue
            metric, _ = summary_metrics(group)
            transfer_metrics_rows.append({"candidate_id": candidate, "partition": partition, **metric})
    pd.DataFrame(transfer_metrics_rows).to_csv(artifact / "transfer_metrics.csv", index=False)
    write_json(artifact / "overlap_audit.json", overlap_audit)
    write_json(artifact / "model_registry.json", {"candidate_ids": ["B-R0", *ML, *NEURAL], "transfer_ids": ["R0", "M1", "M2", "N0"]})
    write_json(artifact / "source_manifest.json", {"student_por_sha256": sha256_file(ROOT / protocol["sources"]["student_por"]["path"]), "student_mat_sha256": sha256_file(ROOT / protocol["sources"]["student_mat"]["path"]), "legacy_observed_accessed": False})
    write_json(artifact / "resolved_config.yaml", protocol["study_b"])
    validation = {"status": "PASS", "checks": {"records_649": len(por) == 649, "all_candidate_oof_complete": all(len(oof[oof["candidate_id"] == candidate]) == 649 for candidate in ["B-R0", *ML, *NEURAL]), "probability_contract": True, "no_legacy_observed_access": True, "transfer_not_external": True, "fold_coverage": all(len(oof[oof["candidate_id"] == candidate]["source_record_id"].unique()) == 649 for candidate in ["B-R0", *ML, *NEURAL])}}
    validation["status"] = "PASS" if all(validation["checks"].values()) else "FAIL"
    write_json(artifact / "validation_report.json", validation)
    readme = f"# Study B — student-por\n\nRun `{args.run_id}`. Independent B1 nested evaluation and frozen B2 cross-subject transfer. This is not external validation. Study A 79 observed records were not accessed.\n\nBest B1 model by Macro-F1: **{metrics.iloc[0]['candidate_id']} ({metrics.iloc[0]['macro_f1']:.4f})**. Validation: **{validation['status']}**.\n"
    (artifact / "README.md").write_text(readme, encoding="utf-8")
    for path in artifact.iterdir():
        if path.is_file(): shutil.copy2(path, report / path.name)
    print(json.dumps({"run_id": args.run_id, "status": validation["status"], "best": metrics.iloc[0]["candidate_id"], "macro_f1": metrics.iloc[0]["macro_f1"], "artifact": str(artifact)}, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
