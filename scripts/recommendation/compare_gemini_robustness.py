"""Compare main Gemini labels with repeatability and prompt-robustness runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _kappa(left: pd.Series, right: pd.Series) -> str:
    mask = left.ne("ABSTAIN") & right.ne("ABSTAIN")
    left_num = pd.to_numeric(left[mask], errors="coerce")
    right_num = pd.to_numeric(right[mask], errors="coerce")
    valid = left_num.notna() & right_num.notna()
    if not valid.any():
        return "UNAVAILABLE"
    try:
        from sklearn.metrics import cohen_kappa_score
        value = float(cohen_kappa_score(left_num[valid], right_num[valid], weights="quadratic"))
        return "UNAVAILABLE" if math.isnan(value) else f"{value:.6f}"
    except Exception:
        return "UNAVAILABLE"


def _pair_metrics(frame: pd.DataFrame, left: str, right: str) -> dict:
    exact = frame[left].eq(frame[right])
    return {
        "n": int(len(frame)),
        "exact_agreement": int(exact.sum()),
        "exact_agreement_rate": float(exact.mean()) if len(frame) else None,
        "disagreement": int((~exact).sum()),
        "weighted_cohen_kappa_numeric_non_abstain": _kappa(frame[left], frame[right]),
        "left_abstain_rate": float(frame[left].eq("ABSTAIN").mean()) if len(frame) else None,
        "right_abstain_rate": float(frame[right].eq("ABSTAIN").mean()) if len(frame) else None,
        "left_distribution": {str(k): int(v) for k, v in frame[left].value_counts().items()},
        "right_distribution": {str(k): int(v) for k, v in frame[right].value_counts().items()},
    }


def _by_dimension(frame: pd.DataFrame, left: str, right: str, column: str) -> dict:
    return {str(value): _pair_metrics(group, left, right) for value, group in frame.groupby(column, sort=True)}


def _attach_state(frame: pd.DataFrame, state_path: Path) -> pd.DataFrame:
    state = pd.read_parquet(state_path)[["case_id", "stage", "risk_band"]].drop_duplicates("case_id")
    return frame.merge(state, on="case_id", how="left", validate="many_to_one")


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"case_id", "action_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing normalized columns in {path}: {sorted(required - set(frame.columns))}")
    return frame[["case_id", "action_id", "label", *(["reason"] if "reason" in frame.columns else [])]].copy()


def _compare(main: pd.DataFrame, other: pd.DataFrame, state_path: Path, left_name: str, right_name: str) -> dict:
    merged = main.merge(other, on=["case_id", "action_id"], how="inner", suffixes=("_main", "_other"), validate="one_to_one")
    if not len(merged):
        raise ValueError(f"no overlap for {left_name} vs {right_name}")
    merged = _attach_state(merged, state_path)
    return {
        "comparison": f"{left_name} vs {right_name}",
        "metric_name": "LLM self-consistency" if right_name == "REPEAT" else "prompt robustness",
        "overall": _pair_metrics(merged, "label_main", "label_other"),
        "by_action": _by_dimension(merged, "label_main", "label_other", "action_id"),
        "by_stage": _by_dimension(merged, "label_main", "label_other", "stage"),
        "by_risk_band": _by_dimension(merged, "label_main", "label_other", "risk_band"),
    }


def _a4_diagnostic(name: str, frame: pd.DataFrame) -> dict:
    a4 = frame[frame["action_id"] == "A4"]
    result = {"run": name, "n": int(len(a4)), "numeric_rate": float((a4["label"] != "ABSTAIN").mean()) if len(a4) else None,
              "abstain_rate": float(a4["label"].eq("ABSTAIN").mean()) if len(a4) else None}
    if "reason" in a4.columns:
        result["abstain_reasons"] = {str(k): int(v) for k, v in a4.loc[a4["label"].eq("ABSTAIN"), "reason"].value_counts(dropna=False).items()}
    else:
        result["abstain_reasons"] = "UNAVAILABLE_IN_NORMALIZED_INPUT"
    return result


def build(main_path: Path, repeat_path: Path, v1b_path: Path, state_path: Path, output: Path) -> None:
    main = _load(main_path)
    repeat = _load(repeat_path)
    v1b = _load(v1b_path)
    reports = [_compare(main, repeat, state_path, "MAIN", "REPEAT"),
               _compare(main, v1b, state_path, "MAIN", "V1B")]
    diagnostics = [_a4_diagnostic("MAIN", main), _a4_diagnostic("REPEAT", repeat), _a4_diagnostic("V1B", v1b)]
    nearly_all_abstain = [item["abstain_rate"] is not None and item["abstain_rate"] >= 0.95 for item in diagnostics]
    flag = "A4 lacks observable evidence in current Student State" if all(nearly_all_abstain) else "NOT_FLAGGED"
    lines = ["# Gemini robustness comparison", "",
             "Repeat is reported as **LLM self-consistency**; v1b is reported as **prompt robustness**.",
             "Neither run is an independent labeling function or a Snorkel input.", ""]
    for report in reports:
        lines += [f"## {report['metric_name']}: {report['comparison']}", "", f"- Overall: `{report['overall']}`", "- By action:"]
        lines.extend(f"  - {key}: `{value}`" for key, value in report["by_action"].items())
        lines.append("- By stage:")
        lines.extend(f"  - {key}: `{value}`" for key, value in report["by_stage"].items())
        lines.append("- By risk band:")
        lines.extend(f"  - {key}: `{value}`" for key, value in report["by_risk_band"].items())
        lines.append("")
    lines += ["## A4 diagnostic", "", f"- Flag: `{flag}`"]
    lines.extend(f"- {item['run']}: `{item}`" for item in diagnostics)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--v1b", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=ROOT / "artifacts/recommendation/states/oulad_student_states.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/recommendation/GEMINI_ROBUSTNESS_COMPARISON.md")
    args = parser.parse_args()
    build(args.main, args.repeat, args.v1b, args.state, args.output)


if __name__ == "__main__":
    main()
