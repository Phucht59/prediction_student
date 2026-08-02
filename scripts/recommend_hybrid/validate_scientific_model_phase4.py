"""Release validator: fail closed until nested OOF artefacts and every gate pass."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/recommend_hybrid/scientific_model"

REQUIRED = ["nested_cv_manifest.json", "outer_oof_predictions.parquet", "outer_oof_rankings.parquet", "metrics_nested_cv.json", "calibration_metrics.json", "personalization_audit.json", "shortcut_audit.json", "bootstrap_ci.json", "final_feature_schema.json", "final_architecture_manifest.json", "final_model_registry.json", "checksums.sha256"]

def main():
    missing = [p for p in REQUIRED if not (ART / p).exists()]
    if missing:
        print("PHASE4_FINAL_MODEL_STRENGTH_GATE_FAILED")
        print("missing final artefacts: " + ", ".join(missing))
        return
    gate = json.loads((ROOT / "reports/recommend_hybrid/scientific_model/PHASE4_GATE.json").read_text())
    if gate.get("evaluation_leakage"):
        print("PHASE4_EVALUATION_LEAKAGE_DETECTED"); return
    if not gate.get("personalization_pass"):
        print("PHASE4_PERSONALIZATION_GATE_FAILED"); return
    if not gate.get("strength_pass"):
        print("PHASE4_FINAL_MODEL_STRENGTH_GATE_FAILED"); return
    print("RECOMMEND_SCIENTIFIC_MODEL_PHASE4_FINAL_PASS")

if __name__ == "__main__": main()
