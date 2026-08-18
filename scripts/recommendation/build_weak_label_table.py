"""Normalize completed provider responses without invoking Snorkel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.constants import ACTION_IDS  # noqa: E402
from src.recommendation.labeling.runtime import load_jsonl  # noqa: E402


def build(raw_path: Path, output: Path, lf_name: str, state_path: Path, feasibility_path: Path) -> pd.DataFrame:
    state = pd.read_parquet(state_path)
    panel_cases = set(state.loc[state["recommendation_eligible"].astype(bool), "case_id"].astype(str))
    feasibility = pd.read_parquet(feasibility_path)
    f_map = {(str(row.case_id), str(row.action_id)): str(row.feasibility_status) for row in feasibility.itertuples()}
    rows = []
    seen = set()
    for record in load_jsonl(raw_path):
        if record.get("status") != "completed":
            continue
        case_id = str(record["case_id"])
        if case_id not in panel_cases:
            raise ValueError(f"case is not an eligible state: {case_id}")
        labels = record["parsed_labels"]["labels"]
        for action_id in ACTION_IDS:
            key = (case_id, action_id, lf_name)
            if key in seen:
                raise ValueError(f"duplicate normalized grain: {key}")
            seen.add(key)
            item = labels[action_id]
            label = str(item["label"])
            feasibility_status = f_map[(case_id, action_id)]
            if feasibility_status == "INFEASIBLE" and label != "ABSTAIN":
                raise ValueError(f"infeasible action has numeric label: {key}")
            rows.append({"case_id": case_id, "action_id": action_id, "lf_name": lf_name,
                         "label": label, "abstain": label == "ABSTAIN", "provider": record["provider"],
                         "model": record["model"], "prompt_version": record["prompt_version"],
                         "reason": item.get("reason"),
                         "feasibility_status": feasibility_status})
    if not rows:
        raise ValueError("no completed labels available")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["case_id", "action_id", "lf_name"]).any():
        raise ValueError("normalized table grain is duplicated")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["case_id", "action_id"]).to_parquet(output, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lf-name", required=True, choices=("LF_GEMMA", "LF_GEMINI"))
    parser.add_argument("--state", type=Path, default=ROOT / "artifacts/recommendation/states/oulad_student_states.parquet")
    parser.add_argument("--feasibility", type=Path, default=ROOT / "artifacts/recommendation/feasibility/oulad_action_feasibility.parquet")
    args = parser.parse_args()
    print({"rows": len(build(args.input, args.output, args.lf_name, args.state, args.feasibility))})


if __name__ == "__main__":
    main()
