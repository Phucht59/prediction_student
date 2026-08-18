"""Offline comparison of Gemini 3.1 and Gemini 3.5 A4 weak labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.progress_monitoring_gemini31 import parse_progress_monitoring_gemini31_response  # noqa: E402
from src.recommendation.labeling.runtime import load_jsonl  # noqa: E402

LABELS = ("0", "1", "2", "3", "ABSTAIN")
NUMERIC_FIELDS = (
    "risk_probability", "inactive_streak", "active_days_ratio", "recent_activity",
    "activity_trend", "assessment_completion", "missing_assessments", "course_progress",
)


def _normalize_label(value) -> str:
    if value == "ABSTAIN":
        return value
    if type(value) is int and value in (0, 1, 2, 3):
        return str(value)
    if isinstance(value, str) and value in {"0", "1", "2", "3"}:
        return value
    raise ValueError(f"invalid label: {value!r}")


def load_gemini31(raw_path: Path, jobs_path: Path) -> pd.DataFrame:
    jobs = load_jsonl(jobs_path)
    records = load_jsonl(raw_path)
    by_job: dict[str, list[dict]] = {}
    for record in records:
        by_job.setdefault(str(record.get("job_id")), []).append(record)
    rows = []
    seen = set()
    for job in jobs:
        job_id = str(job["job_id"])
        job_records = by_job.get(job_id, [])
        if len(job_records) != len(job["case_ids"]) or any(record.get("status") != "completed" for record in job_records):
            raise ValueError(f"Gemini 3.1 job is incomplete: {job_id}")
        raw_responses = {record.get("raw_response") for record in job_records}
        if len(raw_responses) != 1:
            raise ValueError(f"Gemini 3.1 job has inconsistent raw responses: {job_id}")
        parsed = parse_progress_monitoring_gemini31_response(next(iter(raw_responses)), job["case_ids"])
        for case_id in job["case_ids"]:
            case_id = str(case_id)
            if case_id in seen:
                raise ValueError(f"duplicate Gemini 3.1 case: {case_id}")
            seen.add(case_id)
            rows.append({"case_id": case_id, "label": _normalize_label(parsed[case_id]["labels"]["A4"]["label"])})
    if len(rows) != 500 or len(seen) != 500:
        raise ValueError("Gemini 3.1 must contain exactly 500 unique Panel-A cases")
    return pd.DataFrame(rows).rename(columns={"label": "label_gemini31"})


def load_gemini35(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)[["case_id", "action_id", "label"]].copy()
    frame = frame[frame["action_id"] == "B1_PROGRESS_MONITORING"]
    frame["case_id"] = frame["case_id"].astype(str)
    frame["label"] = frame["label"].map(_normalize_label)
    if len(frame) != 500 or frame["case_id"].nunique() != 500:
        raise ValueError("Gemini 3.5 Progress Monitoring source must contain 500 unique cases")
    return frame[["case_id", "label"]].rename(columns={"label": "label_gemini35"})


def _distribution(series: pd.Series) -> dict[str, int]:
    return {label: int(series.eq(label).sum()) for label in LABELS}


def _kappa(left: pd.Series, right: pd.Series) -> float | None:
    mask = ~left.eq("ABSTAIN") & ~right.eq("ABSTAIN")
    if not mask.any():
        return None
    from sklearn.metrics import cohen_kappa_score
    value = float(cohen_kappa_score(pd.to_numeric(left[mask]), pd.to_numeric(right[mask]), labels=[0, 1, 2, 3], weights="quadratic"))
    return None if math.isnan(value) else value


def _pair_metrics(frame: pd.DataFrame) -> dict:
    exact = frame["label_gemini31"].eq(frame["label_gemini35"])
    return {
        "n": len(frame),
        "exact_agreement": float(exact.mean()) if len(frame) else None,
        "exact_count": int(exact.sum()),
        "weighted_kappa_numeric_non_abstain": _kappa(frame["label_gemini31"], frame["label_gemini35"]),
        "gemini31_abstain_rate": float(frame["label_gemini31"].eq("ABSTAIN").mean()) if len(frame) else None,
        "gemini35_abstain_rate": float(frame["label_gemini35"].eq("ABSTAIN").mean()) if len(frame) else None,
    }


def _by_group(frame: pd.DataFrame, field: str) -> list[dict]:
    return [{"group": str(value), **_pair_metrics(group)} for value, group in frame.groupby(field, sort=True, dropna=False, observed=True)]


def _state_variation(frame: pd.DataFrame) -> list[dict]:
    results = []
    numeric = frame["label_gemini31"].map(lambda value: float(value) if value != "ABSTAIN" else float("nan"))
    for field in NUMERIC_FIELDS:
        values = pd.to_numeric(frame[field], errors="coerce")
        if values.nunique(dropna=True) < 2:
            continue
        bins = pd.qcut(values, q=min(4, values.nunique()), duplicates="drop")
        work = pd.DataFrame({"bin": bins, "label": numeric}).dropna(subset=["bin"])
        grouped = work.groupby("bin", observed=True)["label"]
        means = grouped.mean().dropna()
        coverage = grouped.count() / work.groupby("bin", observed=True)["label"].size()
        results.append({
            "feature": field,
            "numeric_label_mean_range": float(means.max() - means.min()) if len(means) >= 2 else 0.0,
            "numeric_coverage_range": float(coverage.max() - coverage.min()) if len(coverage) >= 2 else 0.0,
            "variation_observed": bool(len(means) >= 2 and (means.max() - means.min() >= 0.25 or coverage.max() - coverage.min() >= 0.10)),
        })
    return results


def evaluate(gemini31: pd.DataFrame, gemini35: pd.DataFrame, state: pd.DataFrame) -> dict:
    state = state.copy()
    state["case_id"] = state["case_id"].astype(str)
    merged = gemini31.merge(gemini35, on="case_id", how="inner", validate="one_to_one").merge(
        state[["case_id", "stage", "risk_band", *NUMERIC_FIELDS]], on="case_id", how="left", validate="one_to_one"
    )
    if len(merged) != 500 or merged[["stage", "risk_band"]].isna().any().any():
        raise ValueError("Gemini 3.1/3.5 comparison does not cover all Panel-A state rows")
    max_share = max(_distribution(merged["label_gemini31"]).values()) / len(merged)
    collapsed = max_share == 1.0
    variation = _state_variation(merged)
    return {
        "n": len(merged),
        "overall": _pair_metrics(merged),
        "gemini31_distribution": _distribution(merged["label_gemini31"]),
        "gemini35_distribution": _distribution(merged["label_gemini35"]),
        "by_stage": _by_group(merged, "stage"),
        "by_risk_band": _by_group(merged, "risk_band"),
        "confusion_matrix": pd.crosstab(merged["label_gemini31"], merged["label_gemini35"]).reindex(index=LABELS, columns=LABELS, fill_value=0).to_dict(),
        "state_variation": variation,
        "max_gemini31_class_share": max_share,
        "gemini31_collapsed_to_one_class": collapsed,
        "state_variation_observed": any(item["variation_observed"] for item in variation),
        "status": "FAIL" if collapsed else "REVIEW",
    }


def _write_normalized_gemini31(frame: pd.DataFrame, output: Path) -> None:
    normalized = frame.rename(columns={"label_gemini31": "label"}).copy()
    normalized["action_id"] = "progress_monitoring"
    normalized["lf_name"] = "LF_GEMINI31"
    normalized["label"] = normalized["label"].map(_normalize_label)
    normalized["abstain"] = normalized["label"].eq("ABSTAIN")
    normalized["provider"] = "gemini"
    normalized["model"] = "gemini-3.1-flash-lite"
    normalized["prompt_version"] = "recommendation_progress_monitoring_gemini31_v1"
    normalized["source_artifact"] = "artifacts/recommendation/labeling/raw/progress_monitoring_gemini31.jsonl"
    normalized = normalized[["case_id", "action_id", "lf_name", "label", "abstain", "provider", "model", "prompt_version", "source_artifact"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.sort_values("case_id").to_parquet(output, index=False)


def write_report(result: dict, output: Path) -> None:
    overall = result["overall"]
    kappa = overall["weighted_kappa_numeric_non_abstain"]
    lines = [
        "# A4 Progress Monitoring: Gemini 3.1 vs Gemini 3.5",
        "",
        "These are two distinct LLM weak-label sources from the Gemini model family, not fully independent annotators.",
        "",
        f"- Decision: `{result['status']}`",
        "- Gemini 3.1: `gemini-3.1-flash-lite`",
        "- Gemini 3.5: `gemini-3.5-flash-lite`",
        f"- Cases compared: `{result['n']}`",
        f"- Exact agreement: `{overall['exact_count']}/{overall['n']}` ({overall['exact_agreement']:.6f})",
        f"- Weighted Cohen kappa on numeric non-ABSTAIN pairs: `{kappa if kappa is not None else 'UNAVAILABLE'}`",
        f"- Gemini 3.1 ABSTAIN rate: `{overall['gemini31_abstain_rate']:.6f}`; distribution `{result['gemini31_distribution']}`",
        f"- Gemini 3.5 ABSTAIN rate: `{overall['gemini35_abstain_rate']:.6f}`; distribution `{result['gemini35_distribution']}`",
        f"- Gemini 3.1 max class share: `{result['max_gemini31_class_share']:.6f}`; collapsed: `{result['gemini31_collapsed_to_one_class']}`",
        f"- Observable Student State variation detected: `{result['state_variation_observed']}`",
        "",
        "No automatic hard agreement threshold is applied. `REVIEW` means the agreement is available for substantive review rather than being declared materially reasonable by an invented cutoff.",
        "",
        "## Agreement by stage",
        "",
        "| Stage | N | Exact | Weighted kappa | Gemini 3.1 ABSTAIN | Gemini 3.5 ABSTAIN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in result["by_stage"]:
        lines.append(f"| {item['group']} | {item['n']} | {item['exact_agreement']:.6f} | {item['weighted_kappa_numeric_non_abstain'] if item['weighted_kappa_numeric_non_abstain'] is not None else 'UNAVAILABLE'} | {item['gemini31_abstain_rate']:.6f} | {item['gemini35_abstain_rate']:.6f} |")
    lines += ["", "## Agreement by risk band", "", "| Risk band | N | Exact | Weighted kappa | Gemini 3.1 ABSTAIN | Gemini 3.5 ABSTAIN |", "|---|---:|---:|---:|---:|---:|"]
    for item in result["by_risk_band"]:
        lines.append(f"| {item['group']} | {item['n']} | {item['exact_agreement']:.6f} | {item['weighted_kappa_numeric_non_abstain'] if item['weighted_kappa_numeric_non_abstain'] is not None else 'UNAVAILABLE'} | {item['gemini31_abstain_rate']:.6f} | {item['gemini35_abstain_rate']:.6f} |")
    lines += ["", "## Confusion matrix", "", "Rows = Gemini 3.1; columns = Gemini 3.5.", "", "| Gemini 3.1 \\ Gemini 3.5 | 0 | 1 | 2 | 3 | ABSTAIN |", "|---|---:|---:|---:|---:|---:|"]
    matrix = result["confusion_matrix"]
    for row in LABELS:
        lines.append(f"| {row} | " + " | ".join(str(matrix.get(column, {}).get(row, 0)) for column in LABELS) + " |")
    lines += ["", "## Student State variation diagnostic", "", "Thresholds below are diagnostic heuristics only; they are not a label-selection gate.", "", "| Feature | Numeric-label mean range | Coverage range | Variation observed |", "|---|---:|---:|---|"]
    for item in result["state_variation"]:
        lines.append(f"| {item['feature']} | {item['numeric_label_mean_range']:.6f} | {item['numeric_coverage_range']:.6f} | {item['variation_observed']} |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "artifacts/recommendation/labeling/raw/progress_monitoring_gemini31.jsonl")
    parser.add_argument("--jobs", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/progress_monitoring_gemini31_jobs.jsonl")
    parser.add_argument("--gemini35", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/a4_replacement_gemini_labels.parquet")
    parser.add_argument("--state", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--normalized-output", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/progress_monitoring_gemini31_labels.parquet")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/recommendation/PROGRESS_MONITORING_GEMINI31_VALIDATION.md")
    args = parser.parse_args()
    if not args.raw.exists():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("# A4 Progress Monitoring: Gemini 3.1 vs Gemini 3.5\n\n`RAW_NOT_AVAILABLE` — no API call was made during implementation.\n", encoding="utf-8")
        print("RAW_NOT_AVAILABLE")
        return 2
    gemini31 = load_gemini31(args.raw, args.jobs)
    gemini35 = load_gemini35(args.gemini35)
    state = pd.read_parquet(args.state)
    result = evaluate(gemini31, gemini35, state)
    args.normalized_output.parent.mkdir(parents=True, exist_ok=True)
    _write_normalized_gemini31(gemini31, args.normalized_output)
    write_report(result, args.report)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
