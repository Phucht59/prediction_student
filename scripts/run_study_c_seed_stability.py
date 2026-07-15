from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.oulad.cohort import FORECASTS
from src.studies.oulad.data import load_forecast, record_positions
from src.studies.oulad.evaluate import binary_metrics
from src.studies.oulad.models_deep import fit_deep


SEEDS = [42, 2026, 3407]
NEURAL = ["C-H1", "C-H2"]


def indices(data, fold: int, validation: bool) -> np.ndarray:
    rows = data.split[data.split["role"] == "historical_development"]
    rows = rows[rows["outer_fold"].astype(int) == fold] if validation else rows[rows["outer_fold"].astype(int) != fold]
    return record_positions(data, set(rows["record_id"].astype(str)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--processed", type=Path, default=ROOT / "data" / "processed" / "study_c_oulad")
    args = parser.parse_args()
    artifact = ROOT / "artifacts" / "study_c_oulad" / args.run_id
    selected = pd.read_csv(artifact / "selected_configs.csv")
    base = pd.read_parquet(artifact / "oof_predictions.parquet")
    rows = base[(base["candidate_id"].isin(NEURAL)) & (base["seed"] == 42)].to_dict("records")
    runtime = []
    for forecast_id in FORECASTS:
        data = load_forecast(args.processed, forecast_id)
        for fold in range(3):
            train, validation = indices(data, fold, False), indices(data, fold, True)
            for candidate in NEURAL:
                selected_row = selected[(selected["scope"] == "development_oof") & (selected["forecast_id"] == forecast_id) & (selected["outer_fold"] == fold) & (selected["candidate_id"] == candidate)].iloc[0]
                threshold = float(selected_row["threshold"]); epochs = int(float(selected_row["selected_epoch"]))
                for seed in SEEDS[1:]:
                    started = time.perf_counter()
                    result = fit_deep(data, candidate, train, validation, seed, fixed_epochs=epochs)
                    for position, index in enumerate(validation):
                        probability = float(result["probabilities"][position])
                        rows.append({"candidate_id": candidate, "forecast_id": forecast_id, "scope": "development_oof", "outer_fold": fold, "seed": seed, "record_id": str(data.record_ids[index]), "code_module": data.cohort.iloc[index]["code_module"], "code_presentation": data.cohort.iloc[index]["code_presentation"], "id_student": int(data.cohort.iloc[index]["id_student"]), "true_label": int(data.y[index]), "probability_at_risk": probability, "threshold": threshold, "predicted_label": int(probability >= threshold)})
                    runtime.append({"candidate_id": candidate, "forecast_id": forecast_id, "outer_fold": fold, "seed": seed, "seconds": time.perf_counter() - started, "status": "PASS", "fixed_config": True, "fixed_epoch": epochs, "fixed_threshold": threshold})
    predictions = pd.DataFrame(rows)
    predictions.to_parquet(artifact / "seed_stability_predictions.parquet", index=False)
    metric_rows = []
    for (candidate, forecast, seed), group in predictions.groupby(["candidate_id", "forecast_id", "seed"]):
        metric = binary_metrics(group["true_label"].to_numpy(int), group["probability_at_risk"].to_numpy(float), "fold_specific", group["predicted_label"].to_numpy(int))
        metric_rows.append({"candidate_id": candidate, "forecast_id": forecast, "seed": seed, **metric})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(artifact / "seed_stability.csv", index=False)
    disagreement = []
    for (candidate, forecast), group in predictions.groupby(["candidate_id", "forecast_id"]):
        pivot = group.pivot(index="record_id", columns="seed", values="predicted_label")
        disagreement.append({"candidate_id": candidate, "forecast_id": forecast, "records": len(pivot), "prediction_disagreement_rate": float((pivot.nunique(axis=1) > 1).mean()), "probability_variance_mean": float(group.pivot(index="record_id", columns="seed", values="probability_at_risk").var(axis=1, ddof=0).mean()), "seed_macro_f1_mean": float(metrics[(metrics.candidate_id == candidate) & (metrics.forecast_id == forecast)].macro_f1.mean()), "seed_macro_f1_std": float(metrics[(metrics.candidate_id == candidate) & (metrics.forecast_id == forecast)].macro_f1.std(ddof=0)), "worst_seed_macro_f1": float(metrics[(metrics.candidate_id == candidate) & (metrics.forecast_id == forecast)].macro_f1.min())})
    pd.DataFrame(disagreement).to_csv(artifact / "seed_disagreement.csv", index=False)
    pd.DataFrame(runtime).to_csv(artifact / "seed_stability_runtime.csv", index=False)
    declaration = {"declared_seeds": SEEDS, "best_seed_selection": False, "stochastic_candidates": NEURAL, "strongest_ml": "C-L0", "strongest_ml_seed_not_applicable": True, "reason": "Logistic Regression implementation is deterministic under fixed data/config; fake repeated seed rows were not created."}
    (artifact / "seed_registry.json").write_text(json.dumps(declaration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(predictions), "candidates": NEURAL, "seeds": SEEDS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
