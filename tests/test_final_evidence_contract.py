import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evidence_metrics import classification_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_non_debug_final_pipeline_requires_frozen_selection_config():
    source = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "A frozen --selection-config-json is required for a non-debug final run" in source


def test_latest_final_evidence_metrics_recompute_from_predictions():
    run_name = (ROOT / "artifacts" / "final" / "LATEST_RUN.txt").read_text(encoding="utf-8").strip()
    run_dir = ROOT / "artifacts" / "final" / run_name
    assert not (run_dir / "SMOKE_RUN.md").exists()
    frame = pd.read_csv(run_dir / "locked_test_predictions.csv")
    probabilities = frame[["Prob_Class_0", "Prob_Class_1", "Prob_Class_2"]].to_numpy(dtype=float)
    assert len(frame) == 79
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    metrics = classification_metrics(frame["True_Label"], frame["Pred_Label"], probabilities)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert metrics["macro_f1"] == manifest["metrics_summary"]["macro_f1"]
