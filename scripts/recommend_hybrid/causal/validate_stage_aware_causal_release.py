"""Validate stage-aware causal and imbalance evidence before reporting."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.causal.imbalance import IMBALANCE_MODES  # noqa: E402
from src.recommend_hybrid.causal.protocol import STAGE_ORDER  # noqa: E402
from src.recommend_hybrid.final.actions import ACTION_ORDER  # noqa: E402

DEFAULT_CAUSAL = ROOT / "artifacts/recommend_hybrid/causal/target_trials/stage_action_effects.json"
DEFAULT_INDIVIDUAL = ROOT / "artifacts/recommend_hybrid/causal/target_trials/individual_effects.csv"
DEFAULT_IMBALANCE = ROOT / "artifacts/recommend_hybrid/causal/imbalance/metrics.json"
DEFAULT_OUTPUT = ROOT / "reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_VALIDATION.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {path}")


def validate(
    causal_path: Path,
    individual_path: Path,
    imbalance_path: Path,
) -> dict[str, object]:
    causal = _load_json(causal_path)
    imbalance = _load_json(imbalance_path)
    _finite_tree(causal)
    _finite_tree(imbalance)

    failures: list[str] = []
    if causal.get("status") != "COMPLETE":
        failures.append("CAUSAL_RUN_NOT_COMPLETE")
    if causal.get("stage_order") != list(STAGE_ORDER):
        failures.append("CAUSAL_STAGE_ORDER_MISMATCH")
    if causal.get("action_order") != list(ACTION_ORDER):
        failures.append("CAUSAL_ACTION_ORDER_MISMATCH")
    trials = causal.get("trials")
    if not isinstance(trials, list) or len(trials) != len(STAGE_ORDER) * len(ACTION_ORDER):
        failures.append("CAUSAL_TRIAL_MATRIX_INCOMPLETE")
        trials = []

    seen: set[tuple[str, str]] = set()
    identifiable_count = 0
    for trial in trials:
        protocol = trial.get("protocol", {})
        key = (str(protocol.get("stage")), str(protocol.get("action_id")))
        if key in seen:
            failures.append(f"DUPLICATE_TRIAL:{key[0]}:{key[1]}")
        seen.add(key)
        if key[0] not in STAGE_ORDER or key[1] not in ACTION_ORDER:
            failures.append(f"UNKNOWN_TRIAL_IDENTITY:{key[0]}:{key[1]}")
        status = trial.get("status")
        if status == "CAUSAL_EFFECT_ESTIMATED":
            identifiable_count += 1
            report = trial.get("identifiability", {})
            effect = trial.get("effect", {})
            if report.get("identifiable") is not True:
                failures.append(f"EFFECT_WITHOUT_IDENTIFIABILITY:{key[0]}:{key[1]}")
            interval = effect.get("confidence_interval")
            if not isinstance(interval, list) or len(interval) != 2:
                failures.append(f"EFFECT_INTERVAL_MISSING:{key[0]}:{key[1]}")
            if effect.get("uncertainty_method") != "STUDENT_CLUSTER_PERCENTILE_BOOTSTRAP":
                failures.append(f"CLUSTER_BOOTSTRAP_MISSING:{key[0]}:{key[1]}")
        elif status not in {
            "CAUSAL_EVIDENCE_NOT_IDENTIFIABLE",
            "TRIAL_DATA_NOT_AVAILABLE",
            "TRIAL_EXECUTION_FAILED",
        }:
            failures.append(f"UNKNOWN_TRIAL_STATUS:{key[0]}:{key[1]}:{status}")

    if imbalance.get("status") != "COMPLETE":
        failures.append("IMBALANCE_RUN_NOT_COMPLETE")
    if imbalance.get("modes") != list(IMBALANCE_MODES):
        failures.append("IMBALANCE_MODE_ORDER_MISMATCH")
    if imbalance.get("canonical_prediction_checkpoint_replaced") is not False:
        failures.append("CANONICAL_CHECKPOINT_WAS_REPLACED")
    result_rows = imbalance.get("results")
    if not isinstance(result_rows, list) or len(result_rows) != len(IMBALANCE_MODES):
        failures.append("IMBALANCE_RESULTS_INCOMPLETE")
        result_rows = []
    for result in result_rows:
        if result.get("resampling_scope") != "TRAIN_EMBEDDINGS_ONLY":
            failures.append(f"INVALID_RESAMPLING_SCOPE:{result.get('mode')}")
        if result.get("canonical_checkpoint_replaced") is not False:
            failures.append(f"MODE_REPLACED_CHECKPOINT:{result.get('mode')}")

    individual_count = 0
    if individual_path.is_file() and individual_path.stat().st_size:
        with individual_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "student_id",
                "stage",
                "action_id",
                "cate",
                "propensity",
                "retained_in_overlap",
                "cross_fit_fold",
            }
            if not required.issubset(reader.fieldnames or []):
                failures.append("INDIVIDUAL_EFFECT_SCHEMA_INVALID")
            for row in reader:
                individual_count += 1
                if row.get("stage") not in STAGE_ORDER or row.get("action_id") not in ACTION_ORDER:
                    failures.append("INDIVIDUAL_EFFECT_IDENTITY_INVALID")
                    break

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "stage_order": list(STAGE_ORDER),
        "action_order": list(ACTION_ORDER),
        "trial_count": len(trials),
        "identifiable_trial_count": identifiable_count,
        "individual_effect_count": individual_count,
        "claim_boundary": (
            "PASS validates artifact integrity and observational protocol only; "
            "it does not prove deployment or randomized causal effectiveness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", type=Path, default=DEFAULT_CAUSAL)
    parser.add_argument("--individual", type=Path, default=DEFAULT_INDIVIDUAL)
    parser.add_argument("--imbalance", type=Path, default=DEFAULT_IMBALANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = validate(args.causal, args.individual, args.imbalance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
