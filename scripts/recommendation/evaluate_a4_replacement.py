"""Offline diagnostics for the B1/B2 A4 replacement experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.a4_replacement import (  # noqa: E402
    REPLACEMENT_ACTIONS,
    REPLACEMENT_STATE_FIELDS,
    parse_replacement_response,
)
from src.recommendation.labeling.parser import LabelParseError  # noqa: E402
from src.recommendation.labeling.runtime import load_jsonl  # noqa: E402

DEGENERACY_THRESHOLD = 0.95
VARIATION_THRESHOLD = 0.10
NUMERIC_STATE_FIELDS = {
    "risk_probability", "inactive_streak", "active_days_ratio",
    "assessment_completion", "missing_assessments", "course_progress", "quiz_activity",
}


def _records_by_job(raw_path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in load_jsonl(raw_path):
        grouped.setdefault(str(record["job_id"]), []).append(record)
    return grouped


def load_normalized(raw_path: Path, jobs_path: Path) -> tuple[pd.DataFrame, dict]:
    jobs = load_jsonl(jobs_path)
    raw_by_job = _records_by_job(raw_path)
    rows = []
    failed_jobs = []
    for job in jobs:
        job_id = str(job["job_id"])
        records = raw_by_job.get(job_id, [])
        completed = [record for record in records if record.get("status") == "completed" and record.get("raw_response")]
        if not completed:
            failed_jobs.append(job_id)
            continue
        expected_ids = [str(case_id) for case_id in job["case_ids"]]
        try:
            parsed = parse_replacement_response(completed[0]["raw_response"], expected_ids)
        except LabelParseError as exc:
            raise ValueError(f"invalid replacement response in {job_id}: {exc}") from exc
        record_cases = {str(record.get("case_id")) for record in completed}
        if record_cases != set(expected_ids):
            raise ValueError(f"completed raw records do not cover exactly one batch in {job_id}")
        for case_id in expected_ids:
            for action_id in REPLACEMENT_ACTIONS:
                label = parsed[case_id]["labels"][action_id]["label"]
                rows.append({
                    "case_id": case_id,
                    "action_id": action_id,
                    "label": str(label),
                    "abstain": label == "ABSTAIN",
                    "provider": "gemini",
                    "model": job["model"],
                    "prompt_version": job["prompt_version"],
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no completed A4 replacement responses are available")
    if frame.duplicated(["case_id", "action_id"]).any():
        raise ValueError("duplicate replacement normalized grain")
    if not set(frame["action_id"]) <= set(REPLACEMENT_ACTIONS):
        raise ValueError("unexpected replacement action")
    return frame, {"jobs": len(jobs), "completed_jobs": len(jobs) - len(failed_jobs), "failed_jobs": failed_jobs}


def _distribution(series: pd.Series) -> dict[str, int]:
    return {label: int(series.eq(label).sum()) for label in ("0", "1", "2", "3", "ABSTAIN")}


def _coverage(series: pd.Series) -> float:
    return float((~series.eq("ABSTAIN")).mean()) if len(series) else 0.0


def _numeric_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series[~series.eq("ABSTAIN")], errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else None


def _group_diagnostic(frame: pd.DataFrame, column: str) -> dict:
    groups = []
    for value, group in frame.groupby(column, dropna=False, sort=True, observed=True):
        groups.append({"group": str(value), "n": len(group), "numeric_coverage": _coverage(group["label"]), "numeric_mean": _numeric_mean(group["label"])})
    coverages = [item["numeric_coverage"] for item in groups]
    means = [item["numeric_mean"] for item in groups if item["numeric_mean"] is not None]
    coverage_range = max(coverages) - min(coverages) if len(coverages) >= 2 else 0.0
    mean_range = max(means) - min(means) if len(means) >= 2 else 0.0
    return {"feature": column, "groups": groups, "coverage_range": coverage_range, "numeric_mean_range": mean_range,
            "variation_flag": coverage_range >= VARIATION_THRESHOLD or mean_range >= 0.25}


def _state_diagnostics(labels: pd.DataFrame, state: pd.DataFrame) -> dict[str, list[dict]]:
    merged = labels.merge(state, on="case_id", how="left", validate="many_to_one")
    diagnostics = {}
    for action_id in REPLACEMENT_ACTIONS:
        subset = merged[merged["action_id"] == action_id]
        action_results = []
        for field in (*NUMERIC_STATE_FIELDS, "stage", "risk_band", "recent_activity", "activity_trend", "vle_available"):
            if field not in subset.columns:
                continue
            work = subset[[field, "label"]].dropna(subset=[field])
            if field in NUMERIC_STATE_FIELDS:
                if work[field].nunique() < 2:
                    continue
                work["_bin"] = pd.qcut(work[field], q=min(4, work[field].nunique()), duplicates="drop")
                result = _group_diagnostic(work[["_bin", "label"]].rename(columns={"_bin": "_group"}), "_group")
                result["feature"] = field
            else:
                result = _group_diagnostic(work, field)
            action_results.append(result)
        diagnostics[action_id] = action_results
    return diagnostics


def _metric_table(labels: pd.DataFrame, column: str) -> list[dict]:
    return [{"group": str(value), "n": len(group), "numeric_coverage": _coverage(group["label"]),
             "abstain_rate": float(group["label"].eq("ABSTAIN").mean()),
             "distribution": _distribution(group["label"])}
            for value, group in labels.groupby(column, sort=True)]


def evaluate(labels: pd.DataFrame, state: pd.DataFrame) -> dict:
    diagnostics = _state_diagnostics(labels, state)
    result = {"by_action": {}, "state_diagnostics": diagnostics}
    for action_id in REPLACEMENT_ACTIONS:
        subset = labels[labels["action_id"] == action_id]
        distribution = _distribution(subset["label"])
        max_share = max(distribution.values()) / len(subset) if len(subset) else 0.0
        result["by_action"][action_id] = {
            "n": len(subset),
            "numeric_coverage": _coverage(subset["label"]),
            "abstain_rate": float(subset["label"].eq("ABSTAIN").mean()),
            "distribution": distribution,
            "degeneracy_flag": max_share >= DEGENERACY_THRESHOLD,
            "stage": _metric_table(subset.merge(state[["case_id", "stage"]], on="case_id", validate="many_to_one"), "stage"),
            "risk_band": _metric_table(subset.merge(state[["case_id", "risk_band"]], on="case_id", validate="many_to_one"), "risk_band"),
            "state_variation_features": [item["feature"] for item in diagnostics[action_id] if item["variation_flag"]],
        }
    return result


def _fmt(value) -> str:
    return "UNAVAILABLE" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def write_report(metrics: dict, output: Path, *, metadata: dict) -> None:
    lines = [
        "# A4 replacement evaluation (B1 vs B2)",
        "",
        "Offline diagnostics only. This report does not choose a replacement and does not use mean relevance as a selection rule.",
        "Candidate distinction: B1 monitors current progress/gaps; B2 seeks legitimate academic help. Neither is A3 Study Planning.",
        "",
        f"- Model configured: `gemini-3.5-flash-lite`",
        f"- Prompt version: `recommendation_a4_replacement_v1`",
        f"- Jobs: `{metadata['completed_jobs']}/{metadata['jobs']}` completed; failed jobs: `{len(metadata['failed_jobs'])}`",
        f"- Degeneracy diagnostic threshold: `{DEGENERACY_THRESHOLD:.2f}` of all labels in one class.",
        f"- State-variation diagnostic threshold: coverage range >= `{VARIATION_THRESHOLD:.2f}` or numeric-label mean range >= `0.25`; heuristic only.",
        "",
        "## Overall candidate diagnostics",
        "",
        "| Candidate | Numeric coverage | ABSTAIN rate | Distribution | Degeneracy | State variation features |",
        "|---|---:|---:|---|---|---|",
    ]
    for action_id in REPLACEMENT_ACTIONS:
        item = metrics["by_action"][action_id]
        lines.append(f"| {action_id} | {_fmt(item['numeric_coverage'])} | {_fmt(item['abstain_rate'])} | `{item['distribution']}` | `{ 'FLAG' if item['degeneracy_flag'] else 'NOT_FLAGGED' }` | `{', '.join(item['state_variation_features']) or 'NONE_DETECTED'}` |")
    for dimension in ("stage", "risk_band"):
        lines += ["", f"## Coverage by {dimension}", "", "| Candidate | Group | N | Numeric coverage | ABSTAIN rate |", "|---|---|---:|---:|---:|"]
        for action_id in REPLACEMENT_ACTIONS:
            for item in metrics["by_action"][action_id][dimension]:
                lines.append(f"| {action_id} | {item['group']} | {item['n']} | {_fmt(item['numeric_coverage'])} | {_fmt(item['abstain_rate'])} |")
    lines += ["", "## Relationship with observable Student State", "", "Coverage and numeric-label variation are reported by the supplied state features only. A `NONE_DETECTED` result means the diagnostic did not observe the configured variation threshold; it is not evidence that a candidate is universally irrelevant.", ""]
    for action_id in REPLACEMENT_ACTIONS:
        lines.append(f"### {action_id}")
        for item in metrics["state_diagnostics"][action_id]:
            lines.append(f"- `{item['feature']}`: coverage_range=`{item['coverage_range']:.4f}`, numeric_mean_range=`{item['numeric_mean_range']:.4f}`, variation=`{ 'FLAG' if item['variation_flag'] else 'NOT_FLAGGED' }`")
    lines += ["", "## Decision boundary", "", "No candidate is selected automatically. Final replacement selection requires reviewing these diagnostics together with semantic distinction from A1/A2/A3/A5 and supportability in the current Student State."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "artifacts/recommendation/labeling/raw/a4_replacement_gemini.jsonl")
    parser.add_argument("--jobs", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/a4_replacement_gemini_jobs.jsonl")
    parser.add_argument("--state", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/recommendation/A4_REPLACEMENT_EVALUATION.md")
    parser.add_argument("--normalized-output", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/a4_replacement_gemini_labels.parquet")
    args = parser.parse_args()
    labels, metadata = load_normalized(args.raw, args.jobs)
    state_columns = list(dict.fromkeys(["case_id", *REPLACEMENT_STATE_FIELDS]))
    state = pd.read_parquet(args.state)[state_columns].drop_duplicates("case_id")
    if set(labels["case_id"]) - set(state["case_id"].astype(str)):
        raise ValueError("replacement labels contain cases absent from Student State")
    metrics = evaluate(labels, state)
    args.normalized_output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.normalized_output, index=False)
    write_report(metrics, args.output, metadata=metadata)
    print(json.dumps({"rows": len(labels), "metadata": metadata, "metrics": metrics["by_action"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
