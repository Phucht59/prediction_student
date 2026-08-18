"""Build Phase 6 LF diagnostics without Snorkel or API calls."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL_ACTIONS = ("assessment_recovery", "re_engagement", "study_planning", "progress_monitoring", "retrieval_practice")
SOURCE_NAMES = ("LF_GEMINI", "LF_GEMMA", "LF_BEHAVIOR")
SOURCE_TO_LEGACY = {"assessment_recovery": "A1", "re_engagement": "A2", "study_planning": "A3", "retrieval_practice": "A5"}


def _load_behavior(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)[["case_id", "action_id", "label", "lf_name"]].copy()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["lf_name"] = "LF_BEHAVIOR"
    return frame


def _load_gemini(path: Path, a4_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)[["case_id", "action_id", "label"]].copy()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["action_id"] = frame["action_id"].map(SOURCE_TO_LEGACY).fillna(frame["action_id"])
    frame["action_id"] = frame["action_id"].map({"A1": "assessment_recovery", "A2": "re_engagement", "A3": "study_planning", "A5": "retrieval_practice"}).fillna(frame["action_id"])
    replacement = pd.read_parquet(a4_path)[["case_id", "action_id", "label"]].copy()
    replacement = replacement[replacement["action_id"] == "B1_PROGRESS_MONITORING"].rename(columns={"action_id": "_source"})
    replacement["action_id"] = "progress_monitoring"
    frame = pd.concat([frame, replacement[["case_id", "action_id", "label"]]], ignore_index=True)
    frame["lf_name"] = "LF_GEMINI"
    return frame


def _load_gemma(path: Path, a4_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)[["case_id", "action_id", "label"]].copy()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["action_id"] = frame["action_id"].map({"A1": "assessment_recovery", "A2": "re_engagement", "A3": "study_planning", "A5": "retrieval_practice"}).fillna(frame["action_id"])
    if not a4_path.exists():
        return frame.assign(lf_name="LF_GEMMA")
    a4 = pd.read_parquet(a4_path)[["case_id", "action_id", "label"]]
    a4 = a4[a4["action_id"] == "progress_monitoring"].copy()
    frame = pd.concat([frame, a4], ignore_index=True)
    frame["lf_name"] = "LF_GEMMA"
    return frame


def _kappa(left: pd.Series, right: pd.Series) -> float | None:
    mask = ~left.eq("ABSTAIN") & ~right.eq("ABSTAIN")
    if not mask.any():
        return None
    try:
        from sklearn.metrics import cohen_kappa_score
        value = float(cohen_kappa_score(pd.to_numeric(left[mask]), pd.to_numeric(right[mask]), labels=[0, 1, 2, 3], weights="quadratic"))
        return None if math.isnan(value) else value
    except Exception:
        return None


def _metrics(left: pd.Series, right: pd.Series) -> dict:
    exact = left.eq(right)
    return {"n": len(left), "exact": int(exact.sum()), "agreement": float(exact.mean()) if len(left) else None,
            "conflict_rate": float((~exact).mean()) if len(left) else None, "weighted_kappa": _kappa(left, right),
            "overlap": len(left)}


def _dist(series: pd.Series) -> dict[str, int]:
    return {label: int(series.eq(label).sum()) for label in ("0", "1", "2", "3", "ABSTAIN")}


def _source_metrics(frame: pd.DataFrame) -> dict:
    return {"n": len(frame), "cases": frame["case_id"].nunique(), "coverage": float((~frame["label"].eq("ABSTAIN")).mean()) if len(frame) else 0.0,
            "abstain_rate": float(frame["label"].eq("ABSTAIN").mean()) if len(frame) else 0.0, "distribution": _dist(frame["label"])}


def _confusion(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    labels = ["0", "1", "2", "3", "ABSTAIN"]
    return pd.crosstab(left, right).reindex(index=labels, columns=labels, fill_value=0)


def build(behavior_path: Path, gemini_path: Path, gemini_a4_path: Path, gemma_path: Path, gemma_a4_path: Path, panel_a_path: Path, panel_b_path: Path, output: Path) -> dict:
    panel_a = set(pd.read_parquet(panel_a_path)["case_id"].astype(str))
    panel_b = set(pd.read_parquet(panel_b_path)["case_id"].astype(str))
    if panel_a & panel_b:
        raise ValueError("Panel A and Panel B overlap")
    sources = {"LF_BEHAVIOR": _load_behavior(behavior_path), "LF_GEMINI": _load_gemini(gemini_path, gemini_a4_path), "LF_GEMMA": _load_gemma(gemma_path, gemma_a4_path)}
    for name, frame in sources.items():
        if set(frame["case_id"]) - panel_a or set(frame["case_id"]) & panel_b:
            raise ValueError(f"{name} contains non-Panel-A cases")
    report = {"sources": {name: _source_metrics(frame) for name, frame in sources.items()}, "pairwise": {}, "a4_gemma_ready": gemma_a4_path.exists()}
    for action_id in FINAL_ACTIONS:
        for left_name, right_name in (("LF_GEMINI", "LF_GEMMA"), ("LF_GEMINI", "LF_BEHAVIOR"), ("LF_GEMMA", "LF_BEHAVIOR")):
            left = sources[left_name].query("action_id == @action_id")[["case_id", "label"]].rename(columns={"label": "left"})
            right = sources[right_name].query("action_id == @action_id")[["case_id", "label"]].rename(columns={"label": "right"})
            merged = left.merge(right, on="case_id", how="inner", validate="one_to_one")
            if len(merged):
                report["pairwise"][f"{action_id}:{left_name}:{right_name}"] = _metrics(merged["left"], merged["right"])
    lines = ["# Phase 6 LF diagnostics", "", "Behavioral LF is weak supervision, not ground truth. Gemini repeat/robustness runs are excluded as independent LFs. Feasibility is not an LF.", "", "## Final action contract", "", "- Active: `assessment_recovery`, `re_engagement`, `study_planning`, `progress_monitoring`, `retrieval_practice`.", "- Retired: `content_review`.", "- Not selected: `academic_help_seeking`.", "", "## Behavioral LF rules", "", "- A1 Assessment Recovery: feasible-only; monotonic missing-assessment and inverse completion evidence.", "- A2 Re-engagement: feasible/VLE-available; multi-signal inactivity, active-ratio, recent-activity, and negative-trend score.", "- A3 Study Planning: conservative participating-and-not-strongly-disengaged gate; distinct from A2; no study_regularness proxy.", "- A4 Progress Monitoring: progress-gap formula disabled because course_progress is a stage indicator; ABSTAIN rather than invent a gap.", "- A5 Retrieval Practice: positive quiz activity is evidence; zero activity means availability unknown and ABSTAIN.", ""]
    lines += ["## Source diagnostics", "", "| Source | Rows | Cases | Coverage | ABSTAIN rate | Distribution |", "|---|---:|---:|---:|---:|---|"]
    for name in SOURCE_NAMES:
        item = report["sources"][name]
        lines.append(f"| {name} | {item['n']} | {item['cases']} | {item['coverage']:.4f} | {item['abstain_rate']:.4f} | `{item['distribution']}` |")
    lines += ["", "## Source diagnostics by action", "", "| Action | Source | Cases | Coverage | ABSTAIN rate | Distribution |", "|---|---|---:|---:|---:|---|"]
    for action_id in FINAL_ACTIONS:
        for name in SOURCE_NAMES:
            item = _source_metrics(sources[name].query("action_id == @action_id"))
            lines.append(f"| {action_id} | {name} | {item['cases']} | {item['coverage']:.4f} | {item['abstain_rate']:.4f} | `{item['distribution']}` |")
    lines += ["", "## Pairwise agreement", "", "| Action | Pair | Overlap | Exact agreement | Conflict rate | Weighted kappa |", "|---|---|---:|---:|---:|---:|"]
    for key, item in report["pairwise"].items():
        action, left, right = key.split(":")
        kappa = "UNAVAILABLE" if item["weighted_kappa"] is None else f"{item['weighted_kappa']:.6f}"
        lines.append(f"| {action} | {left} vs {right} | {item['overlap']} | {item['agreement']:.4f} | {item['conflict_rate']:.4f} | {kappa} |")
    lines += ["", "## A5 confusion matrices", ""]
    for left_name, right_name in (("LF_GEMINI", "LF_GEMMA"), ("LF_GEMINI", "LF_BEHAVIOR"), ("LF_GEMMA", "LF_BEHAVIOR")):
        left = sources[left_name].query("action_id == 'retrieval_practice'")[["case_id", "label"]].rename(columns={"label": "left"})
        right = sources[right_name].query("action_id == 'retrieval_practice'")[["case_id", "label"]].rename(columns={"label": "right"})
        merged = left.merge(right, on="case_id", how="inner")
        lines.append(f"### {left_name} vs {right_name}")
        lines.append("")
        lines.append(_confusion(merged["left"], merged["right"]).to_markdown()) if len(merged) else lines.append("No overlap available.")
        lines.append("")
    lines += ["## Gates", "", f"- A4 Gemma normalized labels: `{'READY' if report['a4_gemma_ready'] else 'WAITING_FOR_A4_GEMMA_LABELS'}`.", "- A5: `REVIEW`; retain in the five-action architecture.", "- No Content Review labels, Academic Help-Seeking labels, Panel B rows, or robustness runs are used."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet")
    parser.add_argument("--gemini", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/gemini_supported_labels.parquet")
    parser.add_argument("--gemini-a4", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/a4_replacement_gemini_labels.parquet")
    parser.add_argument("--gemma", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/gemma_supported_labels.parquet")
    parser.add_argument("--gemma-a4", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/progress_monitoring_gemma_labels.parquet")
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/recommendation/PHASE6_LF_DIAGNOSTICS.md")
    args = parser.parse_args()
    report = build(args.behavior, args.gemini, args.gemini_a4, args.gemma, args.gemma_a4, args.panel_a, args.panel_b, args.output)
    print({"a4_gemma_ready": report["a4_gemma_ready"], "sources": report["sources"]})


if __name__ == "__main__":
    main()
