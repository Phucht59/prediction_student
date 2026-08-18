"""Derive Panel-A thresholds and build one primary behavioral LF per action."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.behavioral import (  # noqa: E402
    ACTION_KEYS, BEHAVIOR_LF_NAMES, FINAL_ACTIONS, behavioral_label, derive_thresholds,
)


def _feasibility_map(path: Path) -> dict[str, dict[str, str]]:
    frame = pd.read_parquet(path)
    result = {}
    for case_id, group in frame.groupby("case_id", sort=False):
        mapping = dict(zip(group["action_id"].astype(str), group["feasibility_status"].astype(str)))
        result[str(case_id)] = {ACTION_KEYS[key]: mapping[key] for key in ACTION_KEYS}
    return result


def build(panel_path: Path, feasibility_path: Path, config_path: Path, output_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path).copy()
    panel["case_id"] = panel["case_id"].astype(str)
    if len(panel) != 500 or panel["case_id"].duplicated().any():
        raise ValueError("behavioral labels require exactly 500 unique Panel A cases")
    if set(panel["stage"]) - {"20pct", "35pct", "50pct", "75pct"}:
        raise ValueError("behavioral labels cannot consume FINAL or invalid stages")
    feasibility = _feasibility_map(feasibility_path)
    if set(panel["case_id"]) - set(feasibility):
        raise ValueError("missing feasibility for Panel A case")
    derived = derive_thresholds(panel)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.update(derived)
    config["derived_from"] = str(panel_path).replace("\\", "/")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    thresholds = {
        "quantiles": config["quantiles"],
        "operational_score_cutpoints": config["operational_score_cutpoints"],
        "rule_parameters": config["rule_parameters"],
    }
    rows = []
    for _, row in panel.sort_values("case_id").iterrows():
        case_id = str(row["case_id"])
        state = row.to_dict()
        for action_id in FINAL_ACTIONS:
            result = behavioral_label(state, action_id, feasibility[case_id][action_id], thresholds)
            rows.append({
                "case_id": case_id,
                "action_id": action_id,
                "lf_name": BEHAVIOR_LF_NAMES[action_id],
                "label": result["label"],
                "abstain": bool(result["abstain"]),
                "rule_version": config["version"],
                "evidence_score": result["evidence_score"],
                "reason_code": result["reason_code"],
            })
    frame = pd.DataFrame(rows).sort_values(["case_id", "action_id"])
    if len(frame) != 2500 or frame.duplicated(["case_id", "action_id"]).any():
        raise ValueError("behavioral output must be 500 cases x 5 actions")
    if set(frame["action_id"]) != set(FINAL_ACTIONS):
        raise ValueError("behavioral output action contract mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--feasibility", type=Path, default=ROOT / "artifacts/recommendation/feasibility/oulad_action_feasibility.parquet")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/recommendation/behavioral_lf.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet")
    args = parser.parse_args()
    frame = build(args.panel_a, args.feasibility, args.config, args.output)
    print(json.dumps({"rows": len(frame), "coverage": {str(k): int(v) for k, v in frame.groupby("action_id")["abstain"].apply(lambda x: (~x).sum()).items()}}, indent=2))


if __name__ == "__main__":
    main()
