from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS
from src.model_selection import fit_fold_predict_proba
from src.studies.student_por.data import load_student_csv
from src.studies.student_por.evaluation import summary_metrics
from src.studies.student_por.models import align_probabilities, make_ml_model


SEEDS = [42, 2026, 3407]
CANDIDATES = ["B-RF0", "B-H1"]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); args = parser.parse_args()
    artifact = ROOT / "artifacts" / "study_b_student_por" / args.run_id
    frame = load_student_csv(ROOT / "data" / "raw" / "student-por.csv", "student-por")
    y = frame["G3"].to_numpy(int); outer = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(frame, y))
    selected = pd.read_csv(artifact / "selected_configs.csv")
    base = pd.read_csv(artifact / "oof_predictions.csv")
    rows = base[(base.candidate_id.isin(CANDIDATES)) & (base.seed == 42)].to_dict("records"); runtime = []
    for candidate in CANDIDATES:
        for fold, (train, validation) in enumerate(outer):
            config = json.loads(selected[(selected.candidate_id == candidate) & (selected.outer_fold == fold)].iloc[0].config)
            for seed in SEEDS[1:]:
                started = time.perf_counter()
                if candidate == "B-RF0":
                    model = make_ml_model(candidate, config, seed); model.fit(frame.iloc[train][["G1", "G2"]], y[train]); probabilities = align_probabilities(model, model.predict_proba(frame.iloc[validation][["G1", "G2"]]))
                else:
                    result = fit_fold_predict_proba(train_fold=frame.iloc[train].copy(), validation_fold=frame.iloc[validation].copy(), spec=DATASETS["student-por"], params=config, seed=seed, fold_index=fold); probabilities = result.probabilities
                for position, index in enumerate(validation):
                    rows.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "source_record_id": frame.iloc[index].source_record_id, "source_row_number": int(frame.iloc[index].source_row_number), "true_label": int(y[index]), "predicted_label": int(np.argmax(probabilities[position])), "prob_low": probabilities[position, 0], "prob_medium": probabilities[position, 1], "prob_high": probabilities[position, 2]})
                runtime.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "seconds": time.perf_counter() - started, "status": "PASS", "best_seed_selection": False})
    predictions = pd.DataFrame(rows); predictions.to_csv(artifact / "seed_stability_predictions.csv", index=False)
    metric_rows = []
    for (candidate, seed), group in predictions.groupby(["candidate_id", "seed"]):
        metric, _ = summary_metrics(group); metric_rows.append({"candidate_id": candidate, "seed": seed, **metric})
    metrics = pd.DataFrame(metric_rows); metrics.to_csv(artifact / "seed_stability.csv", index=False)
    summary = []
    for candidate, group in metrics.groupby("candidate_id"):
        pivot = predictions[predictions.candidate_id == candidate].pivot(index="source_record_id", columns="seed", values="predicted_label")
        summary.append({"candidate_id": candidate, "seed_macro_f1_mean": group.macro_f1.mean(), "seed_macro_f1_std": group.macro_f1.std(ddof=0), "worst_seed_macro_f1": group.macro_f1.min(), "prediction_disagreement_rate": (pivot.nunique(axis=1) > 1).mean()})
    pd.DataFrame(summary).to_csv(artifact / "seed_disagreement.csv", index=False); pd.DataFrame(runtime).to_csv(artifact / "seed_stability_runtime.csv", index=False)
    (artifact / "seed_registry.json").write_text(json.dumps({"seeds": SEEDS, "best_seed_selection": False, "candidates": CANDIDATES}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(predictions), "seeds": SEEDS}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
