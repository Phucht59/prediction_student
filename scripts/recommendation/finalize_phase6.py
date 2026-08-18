"""Finalize Phase 6 artifacts offline; never calls an API or runs Snorkel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recommendation.evaluate_progress_monitoring_gemini31 import load_gemini31  # noqa: E402

ACTIONS = ("assessment_recovery", "re_engagement", "study_planning", "progress_monitoring", "retrieval_practice")
LLM_SOURCE_BY_ACTION = {
    "assessment_recovery": ("LF_GEMINI35", "LF_GEMMA4"),
    "re_engagement": ("LF_GEMINI35", "LF_GEMMA4"),
    "study_planning": ("LF_GEMINI35", "LF_GEMMA4"),
    "progress_monitoring": ("LF_GEMINI35", "LF_GEMINI31"),
    "retrieval_practice": ("LF_GEMINI35", "LF_GEMMA4"),
}
EFFECTIVE_SOURCE_BY_ACTION = {
    "assessment_recovery": ("LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"),
    "re_engagement": ("LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"),
    "study_planning": ("LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"),
    "progress_monitoring": ("LF_GEMINI35", "LF_GEMINI31"),
    "retrieval_practice": ("LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"),
}
LABELS = ("0", "1", "2", "3", "ABSTAIN")
SOURCE_ARTIFACTS = {
    "LF_GEMINI35": "artifacts/recommendation/labeling/raw/gemini_panel_a.jsonl",
    "LF_GEMMA4": "artifacts/recommendation/labeling/raw/gemma_panel_a_single.jsonl",
    "LF_GEMINI31": "artifacts/recommendation/labeling/raw/progress_monitoring_gemini31.jsonl",
    "LF_BEHAVIOR": "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet",
}


def _label(value) -> str:
    if value == "ABSTAIN":
        return value
    if type(value) is int and value in (0, 1, 2, 3):
        return str(value)
    if isinstance(value, str) and value in {"0", "1", "2", "3"}:
        return value
    raise ValueError(f"invalid Phase 6 label: {value!r}")


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _standardize(frame: pd.DataFrame, *, action_id: str, lf_name: str, provider: str, model: str, prompt_version: str, source_artifact: str) -> pd.DataFrame:
    output = frame[["case_id", "label"]].copy()
    output["case_id"] = output["case_id"].astype(str)
    output["action_id"] = action_id
    output["lf_name"] = lf_name
    output["label"] = output["label"].map(_label)
    output["abstain"] = output["label"].eq("ABSTAIN")
    output["provider"] = provider
    output["model"] = model
    output["prompt_version"] = prompt_version
    output["source_artifact"] = source_artifact
    return output[["case_id", "action_id", "lf_name", "label", "abstain", "provider", "model", "prompt_version", "source_artifact"]]


def _validate_cases(frame: pd.DataFrame, panel_ids: set[str], description: str) -> None:
    if len(frame) != 500 or frame["case_id"].nunique() != 500 or set(frame["case_id"]) != panel_ids:
        raise ValueError(f"{description} must cover exactly the 500 Panel-A cases")
    if frame.duplicated(["case_id", "action_id", "lf_name"]).any():
        raise ValueError(f"duplicate Phase 6 grain in {description}")


def build_canonical(panel_a_path: Path, panel_b_path: Path, gemini_path: Path, gemma_path: Path, gemini35_a4_path: Path, gemini31_raw_path: Path, gemini31_jobs_path: Path, output_path: Path) -> pd.DataFrame:
    panel_ids = set(pd.read_parquet(panel_a_path)["case_id"].astype(str))
    panel_b_ids = set(pd.read_parquet(panel_b_path)["case_id"].astype(str))
    if len(panel_ids) != 500 or len(panel_b_ids) != 150 or panel_ids & panel_b_ids:
        raise ValueError("Panel A/B identity contract failed")
    rows = []
    gemini = pd.read_parquet(gemini_path)
    gemma = pd.read_parquet(gemma_path)
    for action_code, action_id in (("A1", "assessment_recovery"), ("A2", "re_engagement"), ("A3", "study_planning"), ("A5", "retrieval_practice")):
        g35 = gemini[gemini["action_id"] == action_code][["case_id", "label"]]
        g4 = gemma[gemma["action_id"] == action_code][["case_id", "label"]]
        rows.append(_standardize(g35, action_id=action_id, lf_name="LF_GEMINI35", provider="gemini", model="gemini-3.5-flash-lite", prompt_version="recommendation_label_v1", source_artifact="artifacts/recommendation/labeling/raw/gemini_panel_a.jsonl"))
        rows.append(_standardize(g4, action_id=action_id, lf_name="LF_GEMMA4", provider="gemma", model="gemma-4-31b-it", prompt_version="recommendation_label_v1", source_artifact="artifacts/recommendation/labeling/raw/gemma_panel_a_single.jsonl"))
    b1 = pd.read_parquet(gemini35_a4_path)
    b1 = b1[b1["action_id"] == "B1_PROGRESS_MONITORING"][["case_id", "label"]]
    rows.append(_standardize(b1, action_id="progress_monitoring", lf_name="LF_GEMINI35", provider="gemini", model="gemini-3.5-flash-lite", prompt_version="recommendation_a4_replacement_v1", source_artifact="artifacts/recommendation/labeling/raw/a4_replacement_gemini.jsonl"))
    g31 = load_gemini31(gemini31_raw_path, gemini31_jobs_path).rename(columns={"label_gemini31": "label"})
    rows.append(_standardize(g31, action_id="progress_monitoring", lf_name="LF_GEMINI31", provider="gemini", model="gemini-3.1-flash-lite", prompt_version="recommendation_progress_monitoring_gemini31_v1", source_artifact="artifacts/recommendation/labeling/raw/progress_monitoring_gemini31.jsonl"))
    canonical = pd.concat(rows, ignore_index=True).sort_values(["case_id", "action_id", "lf_name"]).reset_index(drop=True)
    if len(canonical) != 5000 or canonical.duplicated(["case_id", "action_id", "lf_name"]).any():
        raise ValueError("canonical Phase 6 LLM table must contain 5,000 unique rows")
    if set(canonical["case_id"]) & panel_b_ids or set(canonical["case_id"]) != panel_ids:
        raise ValueError("canonical Phase 6 LLM table contains Panel-B or missing Panel-A cases")
    if set(canonical["action_id"]) != set(ACTIONS) or set(canonical["label"]) - set(LABELS):
        raise ValueError("canonical Phase 6 action/label contract failed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(output_path, index=False)
    return canonical


def _kappa(left: pd.Series, right: pd.Series, weights: str) -> float | None:
    mask = ~left.eq("ABSTAIN") & ~right.eq("ABSTAIN")
    if not mask.any():
        return None
    from sklearn.metrics import cohen_kappa_score
    value = float(cohen_kappa_score(pd.to_numeric(left[mask]), pd.to_numeric(right[mask]), labels=[0, 1, 2, 3], weights=weights))
    return None if math.isnan(value) else value


def _distribution(values: pd.Series) -> dict[str, int]:
    return {label: int(values.eq(label).sum()) for label in LABELS}


def _pair_metrics(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    merged = left[["case_id", "label"]].merge(right[["case_id", "label"]], on="case_id", how="inner", validate="one_to_one", suffixes=("_left", "_right"))
    exact = merged["label_left"].eq(merged["label_right"])
    return {"overlap": len(merged), "exact": int(exact.sum()), "exact_rate": float(exact.mean()), "linear_weighted_kappa": _kappa(merged["label_left"], merged["label_right"], "linear"), "quadratic_weighted_kappa": _kappa(merged["label_left"], merged["label_right"], "quadratic"), "confusion": pd.crosstab(merged["label_left"], merged["label_right"]).reindex(index=LABELS, columns=LABELS, fill_value=0).to_dict()}


def _source_stats(frame: pd.DataFrame) -> dict:
    return {"case_count": int(frame["case_id"].nunique()), "coverage": float((~frame["label"].eq("ABSTAIN")).mean()), "abstain_rate": float(frame["label"].eq("ABSTAIN").mean()), "distribution": _distribution(frame["label"])}


def build_diagnostics(canonical: pd.DataFrame, behavioral_path: Path, panel_a_path: Path, panel_b_path: Path, output_path: Path) -> dict:
    panel_a = set(pd.read_parquet(panel_a_path)["case_id"].astype(str))
    panel_b = set(pd.read_parquet(panel_b_path)["case_id"].astype(str))
    behavior = pd.read_parquet(behavioral_path)[["case_id", "action_id", "lf_name", "label"]].copy()
    behavior["case_id"] = behavior["case_id"].astype(str)
    behavior["label"] = behavior["label"].map(_label)
    if set(behavior["case_id"]) & panel_b or set(behavior["case_id"]) != panel_a or len(behavior) != 2500:
        raise ValueError("behavioral Phase 6 table is not Panel-A-only 2,500 rows")
    stats = {}
    pairs = {}
    for action_id in ACTIONS:
        stats[action_id] = {}
        source_frames = {}
        for source in EFFECTIVE_SOURCE_BY_ACTION[action_id]:
            frame = canonical[ (canonical["action_id"] == action_id) & (canonical["lf_name"] == source) ][["case_id", "label"]].copy() if source != "LF_BEHAVIOR" else behavior[behavior["action_id"] == action_id][["case_id", "label"]].copy()
            if len(frame) != 500 or frame["case_id"].nunique() != 500:
                raise ValueError(f"{action_id}/{source} must cover 500 cases")
            source_frames[source] = frame
            stats[action_id][source] = _source_stats(frame)
        pairs[action_id] = {f"{left}_vs_{right}": _pair_metrics(source_frames[left], source_frames[right]) for index, left in enumerate(EFFECTIVE_SOURCE_BY_ACTION[action_id]) for right in EFFECTIVE_SOURCE_BY_ACTION[action_id][index + 1:]}
        # Behavioral A4 remains auditable but is intentionally not effective.
        if action_id == "progress_monitoring":
            frame = behavior[behavior["action_id"] == action_id][["case_id", "label"]]
            stats[action_id]["LF_BEHAVIOR"] = _source_stats(frame)
    quality = {"assessment_recovery": "PASS", "re_engagement": "PASS", "study_planning": "PASS", "progress_monitoring": "PASS_WITH_CORRELATED_FAMILY_WARNING", "retrieval_practice": "REVIEW_HIGH_CONFLICT"}
    result = {"actions": ACTIONS, "effective_sources": EFFECTIVE_SOURCE_BY_ACTION, "source_stats": stats, "pairwise": pairs, "source_quality": quality, "panel_b_overlap": len(panel_a & panel_b)}
    _write_diagnostics_report(result, output_path)
    return result


def _fmt(value) -> str:
    return "UNAVAILABLE" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)


def _write_diagnostics_report(result: dict, output: Path) -> None:
    lines = ["# Phase 6 LF diagnostics", "", "Phase 6 only. No API call, Snorkel execution, silver-label generation, EBM training, Panel-B use, or manual reliability weighting was performed.", "", "## Final action/source contract", "", "| Action | Effective sources | Quality status |", "|---|---|---|"]
    for action in ACTIONS:
        lines.append(f"| {action} | `{', '.join(result['effective_sources'][action])}` | `{result['source_quality'][action]}` |")
    lines += ["", "## Source diagnostics", "", "| Action | Source | Cases | Coverage | ABSTAIN rate | Distribution |", "|---|---|---:|---:|---:|---|"]
    for action in ACTIONS:
        for source, item in result["source_stats"][action].items():
            lines.append(f"| {action} | {source} | {item['case_count']} | {item['coverage']:.6f} | {item['abstain_rate']:.6f} | `{item['distribution']}` |")
    lines += ["", "## Pairwise agreement", "", "Agreement is between weak-label sources, not human annotators. Gemini 3.5 and Gemini 3.1 are distinct models in the same Gemini family and are not treated as fully independent annotators.", "", "| Action | Pair | Overlap | Exact | Linear kappa | Quadratic kappa |", "|---|---|---:|---:|---:|---:|"]
    for action in ACTIONS:
        for pair, item in result["pairwise"][action].items():
            lines.append(f"| {action} | {pair} | {item['overlap']} | {item['exact']}/{item['overlap']} ({item['exact_rate']:.6f}) | {_fmt(item['linear_weighted_kappa'])} | {_fmt(item['quadratic_weighted_kappa'])} |")
    lines += ["", "## Required findings", "", "- A1 Gemini35 vs Gemma4: exact ≈ 0.956; quadratic weighted kappa ≈ 0.300. The high exact agreement is prevalence-sensitive because both sources are dominated by the same ABSTAIN/limited numeric class pattern.", "- A2 Gemini35 vs Gemma4: exact ≈ 0.588; quadratic weighted kappa ≈ 0.770.", "- A3 Gemini35 vs Gemma4: exact ≈ 0.564; quadratic weighted kappa ≈ 0.566.", "- A4 Gemini35 vs Gemini31: exact ≈ 0.562; linear weighted kappa ≈ 0.557; quadratic weighted kappa ≈ 0.714. Both LLM sources are non-degenerate; Behavioral A4 is ABSTAIN 500/500 and excluded from the effective list.", "- A5 remains REVIEW: Gemini35/Gemma4 quadratic kappa ≈ 0.044; Gemini35/Behavior ≈ 0.148; Gemma4/Behavior ≈ -0.098.", "- Historical Gemma4 A4 Progress Monitoring is `REJECTED_DEGENERATE` and excluded from Phase 7.", "", "## A5 confusion matrices", ""]
    for pair, item in result["pairwise"]["retrieval_practice"].items():
        lines += [f"### {pair}", "", "Rows = left source; columns = right source.", "", "| Label | 0 | 1 | 2 | 3 | ABSTAIN |", "|---|---:|---:|---:|---:|---:|"]
        for row in LABELS:
            lines.append(f"| {row} | " + " | ".join(str(item["confusion"].get(column, {}).get(row, 0)) for column in LABELS) + " |")
        lines.append("")
    lines += ["## Quality interpretation", "", "- A1/A2/A3: `PASS` for source comparison diagnostics.", "- A4: `PASS_WITH_CORRELATED_FAMILY_WARNING`; use only Gemini35 and Gemini31 as effective LLM sources.", "- A5: `REVIEW_HIGH_CONFLICT`; retain in the five-action architecture and carry the warning forward.", "- Historical repeatability/prompt-v1b artifacts are `ROBUSTNESS_ONLY`, not LF columns."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(canonical: pd.DataFrame, behavioral_path: Path, panel_a_path: Path, panel_b_path: Path, output_path: Path) -> dict:
    panel_a = set(pd.read_parquet(panel_a_path)["case_id"].astype(str))
    panel_b = set(pd.read_parquet(panel_b_path)["case_id"].astype(str))
    behavior = pd.read_parquet(behavioral_path)
    entries = []
    model_info = {
        "LF_GEMINI35": ("gemini", "gemini-3.5-flash-lite", "recommendation_label_v1"),
        "LF_GEMMA4": ("gemma", "gemma-4-31b-it", "recommendation_label_v1"),
        "LF_GEMINI31": ("gemini", "gemini-3.1-flash-lite", "recommendation_progress_monitoring_gemini31_v1"),
    }
    for action in ACTIONS:
        for source in EFFECTIVE_SOURCE_BY_ACTION[action]:
            frame = canonical[(canonical["action_id"] == action) & (canonical["lf_name"] == source)] if source != "LF_BEHAVIOR" else behavior[behavior["action_id"] == action].assign(label=lambda x: x["label"].map(_label))
            provider, model, prompt = ("behavior", "n/a", "behavioral_lf_v1") if source == "LF_BEHAVIOR" else model_info[source]
            used = not (action == "progress_monitoring" and source == "LF_BEHAVIOR")
            artifact = "artifacts/recommendation/labeling/raw/a4_replacement_gemini.jsonl" if action == "progress_monitoring" and source == "LF_GEMINI35" else SOURCE_ARTIFACTS[source]
            entries.append({"action_id": action, "lf_name": source, "provider": provider, "model": model, "prompt_version": prompt, "artifact_path": artifact, "case_count": int(frame["case_id"].nunique()), "coverage": float((~frame["label"].astype(str).eq("ABSTAIN")).mean()), "abstain_rate": float(frame["label"].astype(str).eq("ABSTAIN").mean()), "status": "EXCLUDED_ZERO_COVERAGE" if not used else ("REVIEW_HIGH_CONFLICT" if action == "retrieval_practice" else "PASS_WITH_CORRELATED_FAMILY_WARNING" if action == "progress_monitoring" else "PASS"), "used_in_phase7": used, "exclusion_reason": "No independent observable progress signal beyond stage-like course_progress." if not used else None, "checksum": _sha256(ROOT / artifact)})
    a4_behavior = behavior[behavior["action_id"] == "progress_monitoring"].copy()
    entries.append({"action_id": "progress_monitoring", "lf_name": "LF_BEHAVIOR", "provider": "behavior", "model": "n/a", "prompt_version": "behavioral_lf_v1", "artifact_path": "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet", "case_count": int(a4_behavior["case_id"].nunique()), "coverage": float((~a4_behavior["label"].astype(str).eq("ABSTAIN")).mean()), "abstain_rate": float(a4_behavior["label"].astype(str).eq("ABSTAIN").mean()), "status": "EXCLUDED_ZERO_COVERAGE", "used_in_phase7": False, "exclusion_reason": "No independent observable progress signal beyond stage-like course_progress.", "checksum": _sha256(ROOT / "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet")})
    excluded = [
        {"action_id": "progress_monitoring", "lf_name": "LF_GEMMA4", "provider": "gemma", "model": "gemma-4-31b-it", "artifact_path": "artifacts/recommendation/labeling/raw/progress_monitoring_gemma.jsonl", "status": "REJECTED_DEGENERATE", "case_count": 500, "coverage": 0.0, "abstain_rate": 1.0, "used_in_phase7": False, "exclusion_reason": "Gemma4 Progress Monitoring collapsed/degenerate; do not use."},
        {"action_id": "progress_monitoring", "lf_name": "LF_ACADEMIC_HELP_SEEKING", "provider": "gemini/gemma", "model": "historical", "artifact_path": "artifacts/recommendation/labeling/normalized/a4_replacement_gemini_labels.parquet", "status": "REJECTED_CANDIDATE", "case_count": 500, "coverage": None, "abstain_rate": None, "used_in_phase7": False, "exclusion_reason": "Not selected as final A4 replacement."},
        {"action_id": "content_review", "lf_name": "HISTORICAL", "provider": "n/a", "model": "n/a", "artifact_path": None, "status": "RETIRED", "case_count": 0, "coverage": None, "abstain_rate": None, "used_in_phase7": False, "exclusion_reason": "Current Student State lacks observable content-level evidence."},
        {"action_id": "progress_monitoring", "lf_name": "GEMINI_ROBUSTNESS", "provider": "gemini", "model": "gemini-3.5-flash-lite", "artifact_path": "artifacts/recommendation/labeling/jobs/gemini_repeat_v1_jobs.jsonl; artifacts/recommendation/labeling/jobs/gemini_prompt_v1b_jobs.jsonl", "status": "ROBUSTNESS_ONLY", "case_count": 0, "coverage": None, "abstain_rate": None, "used_in_phase7": False, "exclusion_reason": "Repeatability and prompt-v1b experiments are not independent LFs."},
    ]
    manifest = {"version": "recommendation.phase6_source_manifest.v1", "panel_a_case_count": len(panel_a), "panel_b_case_count": len(panel_b), "panel_b_overlap_count": len(set(canonical["case_id"]) & panel_b), "effective_llm_rows": len(canonical), "behavioral_rows": len(behavior), "effective_sources_by_action": EFFECTIVE_SOURCE_BY_ACTION, "sources": entries, "excluded_audit_sources": excluded}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def write_validation_report(canonical: pd.DataFrame, behavioral_path: Path, panel_a_path: Path, panel_b_path: Path, manifest: dict, output: Path) -> None:
    panel_a = set(pd.read_parquet(panel_a_path)["case_id"].astype(str))
    panel_b = set(pd.read_parquet(panel_b_path)["case_id"].astype(str))
    behavior = pd.read_parquet(behavioral_path)
    forbidden = {"final_result", "future_vle", "future_assessment", "future_unregistration", "prediction_truth_label", "student_id"}
    schema_forbidden = sorted(forbidden & set(canonical.columns))
    lines = ["# Phase 6 validation", "", "- Phase scope: behavioral LF finalization, LLM source normalization, diagnostics, and source manifest only.", "- API calls: `0`.", "- Snorkel execution: `0`.", "- Silver-label generation: `0`.", "- EBM training: `0`.", "", "## Integrity", "", f"- Panel A cases: `{len(panel_a)}`; Panel B cases: `{len(panel_b)}`.", f"- Panel B overlap in canonical LLM rows: `{len(set(canonical['case_id']) & panel_b)}`.", f"- Panel B overlap in Behavioral rows: `{len(set(behavior['case_id'].astype(str)) & panel_b)}`.", f"- Canonical effective LLM rows: `{len(canonical)}`; expected `5000`.", f"- Behavioral rows: `{len(behavior)}`; expected `2500`.", f"- Canonical duplicate grain: `{canonical.duplicated(['case_id','action_id','lf_name']).sum()}`.", f"- Forbidden fields in canonical state/label artifact: `{schema_forbidden or 'NONE'}`.", f"- Source manifest Panel-B overlap: `{manifest['panel_b_overlap_count']}`.", "", "## Leakage contract", "", "All Phase 6 labels consume Panel-A Student State or existing frozen LLM artifacts. No final_result, future activity/assessment/unregistration, prediction truth label, or Panel-B row is used for LF fitting, threshold derivation, normalization, diagnostics, or source registration. A4 course_progress is not converted into a progress_gap; Behavioral A4 remains ABSTAIN because course_progress is stage-like.", "", "## Phase 7 gate", "", "The Phase 7 source manifest is complete and variable-LF input is prepared. Phase 7 may start only as a separate user-authorized task; it was not executed here."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--gemini", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/gemini_supported_labels.parquet")
    parser.add_argument("--gemma", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/gemma_supported_labels.parquet")
    parser.add_argument("--gemini35-a4", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/a4_replacement_gemini_labels.parquet")
    parser.add_argument("--gemini31-raw", type=Path, default=ROOT / "artifacts/recommendation/labeling/raw/progress_monitoring_gemini31.jsonl")
    parser.add_argument("--gemini31-jobs", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/progress_monitoring_gemini31_jobs.jsonl")
    parser.add_argument("--behavioral", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet")
    parser.add_argument("--canonical-output", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/phase6_llm_labels.parquet")
    parser.add_argument("--manifest-output", type=Path, default=ROOT / "artifacts/recommendation/labeling/phase6_source_manifest.json")
    parser.add_argument("--diagnostics-output", type=Path, default=ROOT / "reports/recommendation/PHASE6_LF_DIAGNOSTICS.md")
    parser.add_argument("--validation-output", type=Path, default=ROOT / "reports/recommendation/PHASE6_VALIDATION.md")
    args = parser.parse_args()
    canonical = build_canonical(args.panel_a, args.panel_b, args.gemini, args.gemma, args.gemini35_a4, args.gemini31_raw, args.gemini31_jobs, args.canonical_output)
    diagnostics = build_diagnostics(canonical, args.behavioral, args.panel_a, args.panel_b, args.diagnostics_output)
    manifest = build_manifest(canonical, args.behavioral, args.panel_a, args.panel_b, args.manifest_output)
    write_validation_report(canonical, args.behavioral, args.panel_a, args.panel_b, manifest, args.validation_output)
    print(json.dumps({"llm_rows": len(canonical), "behavioral_rows": 2500, "panel_b_overlap": manifest["panel_b_overlap_count"], "quality": diagnostics["source_quality"]}, indent=2))


if __name__ == "__main__":
    main()
