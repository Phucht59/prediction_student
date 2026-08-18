"""Build per-action weak-label matrices from frozen Phase 6 sources.

A1/A2/A3/A5 use Gemini 3.5, Gemma 4, and Behavior. A4 Progress Monitoring
uses only Gemini 3.5 and Gemini 3.1. Gemma A4, Content Review, Academic
Help-Seeking, and Gemini robustness runs are never accepted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

FINAL_ACTIONS = (
    "assessment_recovery",
    "re_engagement",
    "study_planning",
    "progress_monitoring",
    "retrieval_practice",
)
BASE_SOURCES = ("LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR")
SOURCES_BY_ACTION = {
    "assessment_recovery": BASE_SOURCES,
    "re_engagement": BASE_SOURCES,
    "study_planning": BASE_SOURCES,
    "progress_monitoring": ("LF_GEMINI35", "LF_GEMINI31"),
    "retrieval_practice": BASE_SOURCES,
}
SOURCES = BASE_SOURCES
ABSTAIN_VALUE = -1
LEGACY_TO_FINAL = {
    "A1": "assessment_recovery",
    "A2": "re_engagement",
    "A3": "study_planning",
    "A5": "retrieval_practice",
}
FINAL_TO_LEGACY = {value: key for key, value in LEGACY_TO_FINAL.items()}
FINAL_TO_LEGACY["progress_monitoring"] = "A4"
FORBIDDEN_ACTIONS = frozenset({"content_review", "academic_help_seeking"})
FORBIDDEN_LF_NAMES = frozenset({"LF_ACADEMIC_HELP_SEEKING", "GEMINI_ROBUSTNESS", "HISTORICAL"})
FORBIDDEN_COLUMNS = frozenset({
    "final_result",
    "future_vle",
    "future_activity",
    "future_assessment",
    "future_unregistration",
    "prediction_truth_label",
    "target",
})
CANONICAL_LLM_RELATIVE = "artifacts/recommendation/labeling/normalized/phase6_llm_labels.parquet"
CANONICAL_BEHAVIOR_RELATIVE = "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet"
PANEL_COLUMNS = ("case_id", "stage")


class A4SourceGateError(RuntimeError):
    """A4 source completeness/identity gate."""


A4GemmaGateError = A4SourceGateError


def _read_panel_ids(path: Path) -> tuple[list[str], set[str], pd.Series]:
    schema = pd.read_parquet(path).columns
    if "case_id" not in schema:
        raise ValueError(f"{path} is missing case_id")
    if set(schema) & FORBIDDEN_COLUMNS:
        raise ValueError(f"{path} contains forbidden leakage columns")
    columns = [column for column in PANEL_COLUMNS if column in schema]
    frame = pd.read_parquet(path, columns=columns)
    case_ids = frame["case_id"].astype(str)
    if "stage" in frame.columns and frame["stage"].astype(str).eq("FINAL-100").any():
        raise ValueError(f"{path} contains FINAL stage cases")
    ordered = sorted(case_ids.tolist())
    unique = set(ordered)
    if len(unique) != len(ordered):
        raise ValueError(f"{path} contains duplicate case_id values")
    return ordered, unique, case_ids


def _normalize_label(value) -> int:
    if value == "ABSTAIN" or value == -1:
        return ABSTAIN_VALUE
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid weak label: {value!r}") from exc
    if integer not in (0, 1, 2, 3):
        raise ValueError(f"invalid weak label: {value!r}")
    return integer


def _labels(path: Path, source: str, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path)[["case_id", "action_id", "label"]].copy()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["action_id"] = frame["action_id"].map(mapping or {}).fillna(frame["action_id"])
    frame["lf_name"] = source
    return frame[["case_id", "action_id", "label", "lf_name"]]


def _validate_panel_cases(frame: pd.DataFrame, panel_a_ids: set[str], description: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["case_id"] = frame["case_id"].astype(str)
    if len(frame) != 500 or frame["case_id"].nunique() != 500:
        raise A4GemmaGateError(f"WAITING_FOR_A4_GEMINI31_LABELS: {description} is not 500 unique cases")
    if set(frame["case_id"]) != panel_a_ids:
        raise A4GemmaGateError(f"WAITING_FOR_A4_GEMINI31_LABELS: {description} cases do not match Panel A")
    frame["label"] = frame["label"].map(_normalize_label)
    return frame


def _reject_forbidden_frame(frame: pd.DataFrame, description: str) -> None:
    if "lf_name" in frame.columns and set(frame["lf_name"].astype(str)) & FORBIDDEN_LF_NAMES:
        raise ValueError(f"{description} contains a forbidden labeling function")
    if "action_id" in frame.columns and set(frame["action_id"].astype(str)) & FORBIDDEN_ACTIONS:
        raise ValueError(f"{description} contains retired or rejected actions")
    if set(frame.columns) & FORBIDDEN_COLUMNS:
        raise ValueError(f"{description} contains forbidden leakage columns")


def expected_sources_by_action() -> dict[str, list[str]]:
    return {action: list(sources) for action, sources in SOURCES_BY_ACTION.items()}


def panel_case_ids(path: Path) -> tuple[list[str], set[str]]:
    ordered, unique, _unused = _read_panel_ids(path)
    return ordered, unique


def validate_source_manifest(manifest_path: Path, panel_a_path: Path, panel_b_path: Path) -> dict:
    """Validate the Phase 6 manifest before any Phase 7 source is consumed."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = expected_sources_by_action()
    actual = {action: list(sources) for action, sources in manifest.get("effective_sources_by_action", {}).items()}
    if actual != expected:
        raise ValueError("Phase 6 source manifest does not match the variable-LF action contract")
    _, panel_a_ids, _ = _read_panel_ids(panel_a_path)
    _, panel_b_ids, _ = _read_panel_ids(panel_b_path)
    if panel_a_ids & panel_b_ids:
        raise ValueError("Panel A and Panel B overlap")
    if manifest.get("panel_a_case_count") != len(panel_a_ids) or manifest.get("panel_b_case_count") != len(panel_b_ids):
        raise ValueError("Phase 6 source manifest panel counts do not match current Panel A/B artifacts")
    if manifest.get("panel_b_overlap_count") != 0:
        raise ValueError("Phase 6 source manifest records Panel-B overlap")
    effective_pairs = {
        (entry["action_id"], entry["lf_name"])
        for entry in manifest.get("sources", [])
        if entry.get("used_in_phase7")
    }
    expected_pairs = {(action, source) for action, sources in SOURCES_BY_ACTION.items() for source in sources}
    if effective_pairs != expected_pairs:
        raise ValueError("Phase 6 source manifest effective sources are incomplete or unexpected")
    excluded = manifest.get("sources", []) + manifest.get("excluded_audit_sources", [])
    for entry in excluded:
        used = bool(entry.get("used_in_phase7"))
        action_id = entry.get("action_id")
        lf_name = entry.get("lf_name")
        if used and (action_id in FORBIDDEN_ACTIONS or lf_name in FORBIDDEN_LF_NAMES):
            raise ValueError(f"forbidden source marked used_in_phase7: {action_id}/{lf_name}")
        if used and action_id == "progress_monitoring" and lf_name == "LF_GEMMA4":
            raise ValueError("Gemma4 Progress Monitoring is rejected and cannot be used in Phase 7")
        if used and lf_name == "GEMINI_ROBUSTNESS":
            raise ValueError("Gemini robustness artifacts cannot be used as Phase 7 labeling functions")
    return manifest


def validate_phase7_authority(
    manifest_path: Path,
    phase7_input_path: Path,
    panel_a_path: Path,
    panel_b_path: Path,
    *,
    weak_supervision_path: Path | None = None,
) -> tuple[dict, dict]:
    """Fail Phase 7 if the frozen Phase 6 manifest and Phase 7 input disagree."""
    manifest = validate_source_manifest(manifest_path, panel_a_path, panel_b_path)
    phase7 = yaml.safe_load(phase7_input_path.read_text(encoding="utf-8"))
    if not isinstance(phase7, dict):
        raise ValueError("phase7_input.yaml is not a mapping")
    expected = expected_sources_by_action()
    actual = {action: list(sources) for action, sources in (phase7.get("actions") or {}).items()}
    if actual != expected:
        raise ValueError("phase7_input.yaml actions do not match the Phase 6 effective-source contract")
    if actual != {action: list(sources) for action, sources in manifest.get("effective_sources_by_action", {}).items()}:
        raise ValueError("phase7_input.yaml and phase6_source_manifest.json disagree on effective sources")
    declared = str(phase7.get("source_manifest") or "").replace("\\", "/")
    if declared != "artifacts/recommendation/labeling/phase6_source_manifest.json":
        raise ValueError("phase7_input.yaml does not point at the frozen Phase 6 source manifest")
    if phase7.get("panel_b_allowed") is not False:
        raise ValueError("phase7_input.yaml must forbid Panel B")
    if phase7.get("final_stage_allowed") is not False:
        raise ValueError("phase7_input.yaml must forbid FINAL")
    if phase7.get("manual_reliability_weights") is not False:
        raise ValueError("phase7_input.yaml must forbid manual reliability weights")
    if phase7.get("variable_lf_count") is not True:
        raise ValueError("phase7_input.yaml must declare variable LF counts")
    if weak_supervision_path is not None:
        config = yaml.safe_load(weak_supervision_path.read_text(encoding="utf-8"))
        config_sources = {action: list(sources) for action, sources in (config.get("action_source_lfs") or {}).items()}
        if config_sources != expected:
            raise ValueError("weak_supervision.yaml action_source_lfs do not match Phase 6 authority")
        if config.get("phase6_source_manifest_version") != manifest.get("version"):
            raise ValueError("weak_supervision.yaml source-manifest version does not match Phase 6")
        if config.get("label_cardinality") != 4:
            raise ValueError("weak_supervision.yaml cardinality must be 4")
        if config.get("gate", {}).get("reject_panel_b") is not True:
            raise ValueError("weak_supervision.yaml must reject Panel B")
    return manifest, phase7


def validate_a4_gemini35(path: Path, panel_a_ids: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise A4GemmaGateError("WAITING_FOR_A4_GEMINI35_LABELS")
    frame = _labels(path, "LF_GEMINI35")
    frame = frame[frame["action_id"] == "B1_PROGRESS_MONITORING"].copy()
    frame["action_id"] = "progress_monitoring"
    return _validate_panel_cases(frame, panel_a_ids, "Gemini 3.5 Progress Monitoring source")


def validate_a4_gemini31(path: Path, panel_a_ids: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise A4GemmaGateError("WAITING_FOR_A4_GEMINI31_LABELS")
    frame = _labels(path, "LF_GEMINI31")
    frame = frame[frame["action_id"] == "progress_monitoring"].copy()
    return _validate_panel_cases(frame, panel_a_ids, "Gemini 3.1 Progress Monitoring source")


def load_sources(
    behavior_path: Path,
    gemini_path: Path,
    gemini35_a4_path: Path,
    gemini31_a4_path: Path,
    gemma_path: Path,
    panel_a_path: Path,
    panel_b_path: Path,
) -> dict[str, pd.DataFrame]:
    _, panel_a_ids, _ = _read_panel_ids(panel_a_path)
    _, panel_b_ids, _ = _read_panel_ids(panel_b_path)
    if panel_a_ids & panel_b_ids:
        raise ValueError("Panel A and Panel B overlap")
    gemini35 = validate_a4_gemini35(gemini35_a4_path, panel_a_ids)
    gemini31 = validate_a4_gemini31(gemini31_a4_path, panel_a_ids)
    behavior = _labels(behavior_path, "LF_BEHAVIOR")
    behavior_a4 = behavior[behavior["action_id"] == "progress_monitoring"].copy()
    _validate_panel_cases(behavior_a4, panel_a_ids, "Progress Monitoring Behavioral LF")
    gemini = _labels(gemini_path, "LF_GEMINI35", LEGACY_TO_FINAL)
    gemma = _labels(gemma_path, "LF_GEMMA4", LEGACY_TO_FINAL)
    sources = {"LF_BEHAVIOR": behavior, "LF_GEMINI35": gemini, "LF_GEMMA4": gemma, "LF_GEMINI31": gemini31}
    for source, frame in sources.items():
        _reject_forbidden_frame(frame, source)
        if set(frame["case_id"]) - panel_a_ids or set(frame["case_id"]) & panel_b_ids:
            raise ValueError(f"{source} contains non-Panel-A cases")
        frame["label"] = frame["label"].map(_normalize_label)
    if ((gemini["action_id"] == "progress_monitoring") & (gemini["lf_name"] == "LF_GEMMA4")).any():
        raise ValueError("Gemma4 Progress Monitoring leaked into the Gemini source table")
    return sources


def load_canonical_sources(
    llm_path: Path,
    behavior_path: Path,
    panel_a_path: Path,
    panel_b_path: Path,
) -> dict[str, pd.DataFrame]:
    """Load only the frozen Phase 6 canonical label tables."""
    _, panel_a_ids, _ = _read_panel_ids(panel_a_path)
    _, panel_b_ids, _ = _read_panel_ids(panel_b_path)
    if panel_a_ids & panel_b_ids:
        raise ValueError("Panel A and Panel B overlap")
    llm = pd.read_parquet(llm_path, columns=["case_id", "action_id", "lf_name", "label"])
    behavior = pd.read_parquet(behavior_path, columns=["case_id", "action_id", "lf_name", "label"])
    for frame, description in ((llm, "canonical Phase 6 LLM labels"), (behavior, "canonical Phase 6 behavioral labels")):
        _reject_forbidden_frame(frame, description)
        frame["case_id"] = frame["case_id"].astype(str)
        if set(frame["case_id"]) - panel_a_ids or set(frame["case_id"]) & panel_b_ids:
            raise ValueError(f"{description} contain Panel B or non-Panel-A cases")
    if len(llm) != 5000 or llm.duplicated(["case_id", "action_id", "lf_name"]).any():
        raise ValueError("canonical Phase 6 LLM table must contain 5,000 unique rows")
    if len(behavior) != 2500 or behavior.duplicated(["case_id", "action_id"]).any():
        raise ValueError("canonical Phase 6 behavioral table must contain 2,500 unique rows")
    gemma_a4 = (llm["action_id"] == "progress_monitoring") & (llm["lf_name"] == "LF_GEMMA4")
    if gemma_a4.any():
        raise ValueError("canonical Phase 6 LLM table includes rejected Gemma4 Progress Monitoring")
    behavior = behavior.copy()
    behavior["lf_name"] = "LF_BEHAVIOR"
    combined = pd.concat([llm, behavior], ignore_index=True)
    combined["label"] = combined["label"].map(_normalize_label)
    sources: dict[str, pd.DataFrame] = {}
    for source in sorted(set(combined["lf_name"])):
        sources[source] = combined.loc[combined["lf_name"] == source, ["case_id", "action_id", "label", "lf_name"]].copy()
    required = {source for sources_for_action in SOURCES_BY_ACTION.values() for source in sources_for_action}
    missing = required - set(sources)
    if missing:
        raise ValueError(f"canonical Phase 6 tables are missing effective sources: {sorted(missing)}")
    return sources


def load_matrices(matrix_dir: Path) -> dict[str, pd.DataFrame]:
    matrices = {}
    for action_id in FINAL_ACTIONS:
        path = matrix_dir / f"{action_id}.parquet"
        if not path.exists():
            raise ValueError(f"missing weak-label matrix: {path}")
        frame = pd.read_parquet(path)
        sources = SOURCES_BY_ACTION[action_id]
        expected_columns = ["case_id", *sources]
        if list(frame.columns) != expected_columns:
            raise ValueError(f"{action_id} matrix columns {list(frame.columns)} != {expected_columns}")
        if len(frame) != 500 or frame["case_id"].astype(str).nunique() != 500:
            raise ValueError(f"{action_id} matrix is not 500 unique cases")
        if action_id == "progress_monitoring" and "LF_GEMMA4" in frame.columns:
            raise ValueError("A4 matrix must not include a Gemma4 column")
        if action_id == "progress_monitoring" and frame.shape != (500, 3):
            raise ValueError("A4 matrix must be 500x2 sources plus case_id")
        if action_id != "progress_monitoring" and frame.shape != (500, 4):
            raise ValueError(f"{action_id} matrix must be 500x3 sources plus case_id")
        values = set(frame[list(sources)].stack().unique())
        if values - {-1, 0, 1, 2, 3}:
            raise ValueError(f"invalid matrix values in {action_id}")
        matrices[action_id] = frame
    case_orders = [matrices[action]["case_id"].astype(str).tolist() for action in FINAL_ACTIONS]
    if any(order != case_orders[0] for order in case_orders[1:]):
        raise ValueError("per-action matrices do not share a deterministic case order")
    return matrices


def build_matrices(sources: dict[str, pd.DataFrame], panel_a_path: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    panel_a_ids, panel_a_set, _ = _read_panel_ids(panel_a_path)
    if set(panel_a_ids) != panel_a_set:
        raise ValueError("Panel A case order is internally inconsistent")
    for source, frame in sources.items():
        _reject_forbidden_frame(frame, source)
        if set(frame["action_id"].astype(str)) & FORBIDDEN_ACTIONS:
            raise ValueError(f"{source} includes a retired or rejected action")
    if "LF_GEMMA4" in sources:
        gemma = sources["LF_GEMMA4"]
        if (gemma["action_id"] == "progress_monitoring").any():
            raise ValueError("do not pad or accept Gemma4 as an A4 labeling function")
    matrices = {}
    for action_id in FINAL_ACTIONS:
        action_sources = SOURCES_BY_ACTION[action_id]
        matrix = pd.DataFrame({"case_id": panel_a_ids})
        for source in action_sources:
            if source not in sources:
                raise ValueError(f"missing source {source} for {action_id}")
            source_rows = sources[source].query("action_id == @action_id")[["case_id", "label"]].copy()
            source_rows["case_id"] = source_rows["case_id"].astype(str)
            if source_rows.duplicated("case_id").any():
                raise ValueError(f"{source}/{action_id} contains duplicate case_id values")
            if len(source_rows) != 500 or source_rows["case_id"].nunique() != 500:
                raise ValueError(f"{source}/{action_id} does not cover exactly 500 cases")
            if set(source_rows["case_id"]) != panel_a_set:
                raise ValueError(f"{source}/{action_id} cases do not match Panel A")
            matrix[source] = matrix["case_id"].map(source_rows.set_index("case_id")["label"])
        if list(matrix.columns) != ["case_id", *action_sources]:
            raise ValueError(f"{action_id} matrix has unexpected columns")
        if action_id == "progress_monitoring" and len(action_sources) != 2:
            raise ValueError("A4 must not be padded with a fake third labeling function")
        if matrix[list(action_sources)].isna().any().any():
            raise ValueError(f"missing weak labels in {action_id}")
        if set(matrix[list(action_sources)].stack().unique()) - {-1, 0, 1, 2, 3}:
            raise ValueError(f"invalid matrix values in {action_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix.to_parquet(output_dir / f"{action_id}.parquet", index=False)
        matrices[action_id] = matrix
    case_index = pd.DataFrame({"row_index": range(len(panel_a_ids)), "case_id": panel_a_ids})
    case_index.to_parquet(output_dir / "case_index.parquet", index=False)
    return matrices
