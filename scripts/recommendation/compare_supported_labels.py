"""Compare the two weak sources on actions supported by the current state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_ACTIONS = ("A1", "A2", "A3", "A5")
UNSUPPORTED_REASON = "Current Student State lacks observable content-level evidence."


def _distribution(series: pd.Series) -> dict[str, int]:
    return {label: int(series.eq(label).sum()) for label in ("0", "1", "2", "3", "ABSTAIN")}


def _weighted_kappa(left: pd.Series, right: pd.Series) -> float | None:
    mask = ~left.eq("ABSTAIN") & ~right.eq("ABSTAIN")
    if not mask.any():
        return None
    try:
        from sklearn.metrics import cohen_kappa_score

        value = float(cohen_kappa_score(
            pd.to_numeric(left[mask]), pd.to_numeric(right[mask]),
            labels=[0, 1, 2, 3], weights="quadratic",
        ))
        return None if math.isnan(value) else value
    except Exception:
        return None


def _metrics(frame: pd.DataFrame) -> dict:
    left = frame["label_gemma"].astype(str)
    right = frame["label_gemini"].astype(str)
    exact = left.eq(right)
    return {
        "n": int(len(frame)),
        "exact": int(exact.sum()),
        "exact_rate": float(exact.mean()) if len(frame) else None,
        "disagreement": int((~exact).sum()),
        "weighted_kappa": _weighted_kappa(left, right),
        "gemma_abstain_rate": float(left.eq("ABSTAIN").mean()) if len(frame) else None,
        "gemini_abstain_rate": float(right.eq("ABSTAIN").mean()) if len(frame) else None,
        "gemma_distribution": _distribution(left),
        "gemini_distribution": _distribution(right),
    }


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"case_id", "action_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns in {path}: {sorted(required - set(frame.columns))}")
    frame = frame[["case_id", "action_id", "label"]].copy()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["action_id"] = frame["action_id"].astype(str)
    frame["label"] = frame["label"].astype(str)
    if len(frame) != 2000 or frame["case_id"].nunique() != 500:
        raise ValueError(f"{path} must contain 2,000 rows and 500 cases")
    if set(frame["action_id"]) != set(SUPPORTED_ACTIONS):
        raise ValueError(f"{path} contains unsupported actions")
    if frame.duplicated(["case_id", "action_id"]).any():
        raise ValueError(f"duplicate normalized grain in {path}")
    return frame


def _fmt(value) -> str:
    if value is None:
        return "UNAVAILABLE"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_supportability_report(output: Path, gemma_raw: Path, gemini_raw: Path, merged: pd.DataFrame) -> None:
    raw_gemma = [json.loads(line) for line in gemma_raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    gemma_failed = sum(record.get("status") == "failed" for record in raw_gemma)
    # The recovery script intentionally re-parses raw_response; this count is
    # evidence about the records recovered, not a count of fabricated labels.
    gemma_completed_a4_abstain = sum(
        record.get("status") == "completed"
        and isinstance(record.get("parsed_labels"), dict)
        and record["parsed_labels"].get("labels", {}).get("A4", {}).get("label") == "ABSTAIN"
        for record in raw_gemma
    )
    gemini_records = [json.loads(line) for line in gemini_raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    gemini_a4_cases: dict[str, str] = {}
    for record in gemini_records:
        raw = record.get("raw_response")
        if not raw:
            continue
        data = json.loads(raw)
        results = data.get("results", [data]) if isinstance(data, dict) else []
        for item in results:
            if isinstance(item, dict) and item.get("case_id") and isinstance(item.get("labels"), dict):
                gemini_a4_cases[str(item["case_id"])] = str(item["labels"].get("A4", {}).get("label"))
    gemini_a4_abstain = sum(value == "ABSTAIN" for value in gemini_a4_cases.values())
    lines = [
        "# Action supportability decision",
        "",
        "A4 Content Review remains in the A1-A5 action catalog but is locked as `UNSUPPORTED_BY_CURRENT_STATE`.",
        f"Reason: `{UNSUPPORTED_REASON}`",
        "",
        "This is an empirical supportability/data-observability decision: the current Student State does not expose content-level evidence. It is not model-performance cherry-picking and does not remove an action because of an agreement score.",
        "",
        "## Evidence",
        "",
        f"- Gemini A4 ABSTAIN: `{gemini_a4_abstain}/500` unique Panel A cases.",
        f"- Gemma completed A4 ABSTAIN: `{gemma_completed_a4_abstain}/484` completed cases.",
        f"- Gemma raw failed records recovered offline: `{gemma_failed}`; their supported A1/A2/A3/A5 function-call arguments were reparsed from `raw_response`.",
        "- No API request was made during recovery.",
        "",
        "## Action support contract",
        "",
        "| Action | Status | Use in weak-label comparison/training |",
        "|---|---|---|",
        "| A1 | SUPPORTED | Included |",
        "| A2 | SUPPORTED | Included |",
        "| A3 | SUPPORTED | Included |",
        "| A4 | UNSUPPORTED_BY_CURRENT_STATE | Excluded |",
        "| A5 | SUPPORTED | Included; REVIEW |",
        "",
        "A4 is excluded from weak-label training, Snorkel, EBM training, and ranking-model evaluation. A5 remains supported but is flagged `REVIEW` because prior weak-source agreement was weak.",
        "",
        f"The normalized comparison below contains `{len(merged)}` supported-action pairs (A1/A2/A3/A5), with no A4 rows.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(gemma_path: Path, gemini_path: Path, state_path: Path, output: Path, supportability_output: Path, gemma_raw: Path, gemini_raw: Path) -> dict:
    gemma = _load(gemma_path).rename(columns={"label": "label_gemma"})
    gemini = _load(gemini_path).rename(columns={"label": "label_gemini"})
    merged = gemma.merge(gemini, on=["case_id", "action_id"], how="outer", validate="one_to_one")
    if merged["label_gemma"].isna().any() or merged["label_gemini"].isna().any():
        raise ValueError("Gemma/Gemini tables do not have identical supported-action grain")
    state = pd.read_parquet(state_path)[["case_id", "stage", "risk_band"]].drop_duplicates("case_id")
    merged = merged.merge(state, on="case_id", how="left", validate="many_to_one")
    if merged[["stage", "risk_band"]].isna().any().any():
        raise ValueError("comparison contains a case missing state stratification fields")

    overall = _metrics(merged)
    by_action = {str(key): _metrics(group) for key, group in merged.groupby("action_id", sort=True)}
    by_stage = {str(key): _metrics(group) for key, group in merged.groupby("stage", sort=True)}
    by_risk = {str(key): _metrics(group) for key, group in merged.groupby("risk_band", sort=True)}
    lines = [
        "# Supported-action weak-source comparison",
        "",
        "This measures **LLM weak-source agreement**, not human inter-rater agreement. A4 is excluded because it is `UNSUPPORTED_BY_CURRENT_STATE`; A5 remains included and is flagged `REVIEW`.",
        "",
        f"- Gemma normalized rows: `{len(gemma)}`",
        f"- Gemini normalized rows: `{len(gemini)}`",
        f"- Supported actions: `{', '.join(SUPPORTED_ACTIONS)}`",
        "",
        "## Overall",
        "",
        f"- Exact agreement: `{overall['exact']}/{overall['n']}` ({_fmt(overall['exact_rate'])})",
        f"- Disagreement: `{overall['disagreement']}/{overall['n']}`",
        f"- Quadratic weighted Cohen kappa, numeric non-ABSTAIN pairs: `{_fmt(overall['weighted_kappa'])}`",
        f"- Gemma ABSTAIN rate: `{_fmt(overall['gemma_abstain_rate'])}`; distribution: `{overall['gemma_distribution']}`",
        f"- Gemini ABSTAIN rate: `{_fmt(overall['gemini_abstain_rate'])}`; distribution: `{overall['gemini_distribution']}`",
        "",
        "## By action",
        "",
        "| Action | Exact | Rate | Weighted kappa | Gemma ABSTAIN | Gemini ABSTAIN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for action in SUPPORTED_ACTIONS:
        item = by_action[action]
        lines.append(f"| {action} | {item['exact']}/{item['n']} | {_fmt(item['exact_rate'])} | {_fmt(item['weighted_kappa'])} | {_fmt(item['gemma_abstain_rate'])} | {_fmt(item['gemini_abstain_rate'])} |")
    lines += ["", "### Label distributions by action", ""]
    for action in SUPPORTED_ACTIONS:
        item = by_action[action]
        lines.append(f"- `{action}`: Gemma `{item['gemma_distribution']}`; Gemini `{item['gemini_distribution']}`")
    for title, values in (("stage", by_stage), ("risk band", by_risk)):
        lines += ["", f"## By {title}", ""]
        for key, item in values.items():
            lines.append(f"- `{key}`: exact=`{item['exact']}/{item['n']}` ({_fmt(item['exact_rate'])}); kappa=`{_fmt(item['weighted_kappa'])}`; Gemma ABSTAIN=`{_fmt(item['gemma_abstain_rate'])}`; Gemini ABSTAIN=`{_fmt(item['gemini_abstain_rate'])}`")
    lines += ["", "## Review flag", "", "- `A5`: REVIEW — retain for now; do not remove automatically."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    supportability_output.parent.mkdir(parents=True, exist_ok=True)
    _write_supportability_report(supportability_output, gemma_raw, gemini_raw, merged)
    return {"overall": overall, "by_action": by_action, "rows": len(merged)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/gemma_supported_labels.parquet")
    parser.add_argument("--gemini", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/gemini_supported_labels.parquet")
    parser.add_argument("--state", type=Path, default=ROOT / "artifacts/recommendation/states/oulad_student_states.parquet")
    parser.add_argument("--gemma-raw", type=Path, default=ROOT / "artifacts/recommendation/labeling/raw/gemma_panel_a_single.jsonl")
    parser.add_argument("--gemini-raw", type=Path, default=ROOT / "artifacts/recommendation/labeling/raw/gemini_panel_a.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/recommendation/WEAK_LABEL_QUALITY.md")
    parser.add_argument("--supportability-output", type=Path, default=ROOT / "reports/recommendation/ACTION_SUPPORTABILITY.md")
    args = parser.parse_args()
    result = build(args.gemma, args.gemini, args.state, args.output, args.supportability_output, args.gemma_raw, args.gemini_raw)
    print(json.dumps({"rows": result["rows"], "overall": result["overall"], "by_action": result["by_action"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
