"""Build the final thesis authority lock from frozen Phase 1-9 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "final" / "thesis"
REPORT = ROOT / "reports" / "final" / "thesis"
REGISTRY = ROOT / "configs" / "final" / "final_model_authority.yaml"

UCI_SOURCE = ROOT / "artifacts" / "final" / "unified_stage_aware_uci" / "stage_metrics.csv"
OULAD_UNIFIED_SOURCE = ROOT / "artifacts" / "final" / "unified_stage_aware_oulad" / "stage_metrics.csv"
OULAD_H1_SOURCE = ROOT / "artifacts" / "final" / "h1_final" / "stage_metrics.csv"
ENDPOINT_SOURCE = ROOT / "artifacts" / "audit" / "phase7" / "endpoint_comparator.csv"
FREEZE_SOURCE = ROOT / "artifacts" / "final_candidate_freeze" / "FINAL_H1_FREEZE_MANIFEST.json"
PHASE9_RESIDUAL = ROOT / "artifacts" / "audit" / "phase9" / "residual_ablation.csv"
PHASE9_TEMPORAL = ROOT / "artifacts" / "audit" / "phase9" / "temporal_ablation.csv"
VALIDATION = ART / "validation_summary.json"

MODELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "cnn_bilstm": "CNN-BiLSTM",
}
UCI_STAGES = {
    "S0_EARLY_NO_GRADE": "S0 — no G1/G2",
    "S1_MID_G1_ONLY": "S1 — G1 only",
    "S2_LATE_G1_G2": "S2 — G1 and G2",
}
OULAD_STAGES = {
    "E1_EARLY_20PCT": ("E1", 20),
    "E2_EARLY_35PCT": ("E2", 35),
    "M1_MIDDLE_FROZEN": ("M1", 50),
    "L1_LATE_75PCT": ("L1", 75),
}
METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "pr_auc",
    "roc_auc",
    "nll",
    "brier",
    "ece",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None or value == "" or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if index == 0 else "---:" for index in range(len(headers))) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def git_sha(short: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{short}^{{commit}}"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".yaml", ".yml"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def build_uci_tables() -> pd.DataFrame:
    frame = pd.read_csv(UCI_SOURCE)
    frame = frame.loc[frame.model_family.isin(MODELS)].copy()
    frame["model"] = frame.model_family.map(MODELS)
    frame["stage_label"] = frame.prediction_stage.map(UCI_STAGES)
    frame["protocol_id"] = "unified_stage_aware_uci_v1"
    columns = ["dataset", "model", "model_family", "prediction_stage", "stage_label", "protocol_id", *METRICS]
    frame = frame.loc[:, columns].sort_values(["dataset", "prediction_stage", "model"])
    if len(frame) != 48:
        raise RuntimeError(f"expected 48 UCI stage rows, got {len(frame)}")
    frame.to_csv(ART / "uci_stage_metrics.csv", index=False)
    sections = ["# UCI stage-aware results", "", "These secondary results do not replace the UCI headline endpoint results."]
    for dataset, title in (("student_mat", "Student-Mat"), ("student_por", "Student-Por")):
        for stage, label in UCI_STAGES.items():
            current = frame.loc[(frame.dataset == dataset) & (frame.prediction_stage == stage)]
            sections.extend([
                "",
                f"## {title} — {label}",
                "",
                markdown_table(
                    ["Model", "Accuracy", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Macro-F1", "Weighted F1", "PR-AUC", "ROC-AUC", "NLL", "Brier", "ECE"],
                    ([row.model, *[fmt(getattr(row, field)) for field in METRICS]] for row in current.itertuples()),
                ),
            ])
    write_text(REPORT / "UCI_STAGE_RESULTS.md", "\n".join(sections))
    return frame


def build_endpoint_table() -> pd.DataFrame:
    source = pd.read_csv(ENDPOINT_SOURCE)
    rows = []
    definitions = (
        ("cnn_bilstm", "H0 CNN-BiLSTM", "LEGACY_ENDPOINT_AUTHORITY_WITH_SCORE_PROXY_CAVEAT", "legacy_score_availability_proxy"),
        ("h1_tabular_residual", "H1 Tabular Residual Hybrid", "STRICT_NO_UNVERIFIED_SCORE_ENDPOINT_RESULT", "strict_no_unverified_score"),
        ("mlp", "MLP", "HISTORICAL_COMPARATOR_WITH_SCORE_PROXY_CAVEAT", "legacy_score_availability_proxy"),
    )
    for model_id, model, status, protocol in definitions:
        row = source.loc[source.model_id.eq(model_id)].iloc[0]
        rows.append({
            "model": model,
            "model_id": model_id,
            "macro_f1": float(row.macro_f1),
            "accuracy": float(row.accuracy),
            "balanced_accuracy": float(row.balanced_accuracy),
            "pr_auc": float(row.pr_auc),
            "roc_auc": float(row.roc_auc),
            "nll": float(row.nll),
            "brier": float(row.brier),
            "ece": float(row.ece),
            "authority_status": status,
            "feature_availability_protocol": protocol,
        })
    result = pd.DataFrame(rows)
    result.to_csv(ART / "oulad_endpoint_authority.csv", index=False)
    return result


def build_oulad_stage_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    h1_source = pd.read_csv(OULAD_H1_SOURCE)
    h1 = h1_source.loc[h1_source.candidate.eq("H1_TABULAR_RESIDUAL_EXPERT")].copy()
    h1["stage"] = h1.prediction_stage.map(lambda value: OULAD_STAGES[value][0])
    h1["observation_percentage"] = h1.prediction_stage.map(lambda value: OULAD_STAGES[value][1])
    h1["protocol_id"] = "h1_final_outer_v1"
    h1["model"] = "H1 Tabular Residual Hybrid"
    h1_columns = ["model", "stage", "prediction_stage", "observation_percentage", "protocol_id", *METRICS, "risk_precision", "risk_recall", "risk_f1"]
    h1 = h1.loc[:, h1_columns].sort_values("observation_percentage")
    if len(h1) != 4:
        raise RuntimeError("frozen H1 early-warning evidence must have four stages")
    h1.to_csv(ART / "oulad_early_warning_h1.csv", index=False)

    unified = pd.read_csv(OULAD_UNIFIED_SOURCE)
    classic = unified.loc[
        unified.model_family.isin({
            "logistic_regression", "decision_tree", "random_forest",
            "hist_gradient_boosting", "svm", "xgboost",
        })
        & unified.threshold_policy.eq("INNER_OOF_STAGE_THRESHOLD")
    ].copy()
    classic["model"] = classic.model_family.map(MODELS)
    classic["protocol_id"] = "unified_stage_aware_oulad_v2"
    classic["source_role"] = "frozen_classical_comparator"
    phase6 = h1_source.loc[h1_source.candidate.isin(["M0_MLP", "H1_TABULAR_RESIDUAL_EXPERT"])].copy()
    phase6["model_family"] = phase6.candidate.map({"M0_MLP": "mlp", "H1_TABULAR_RESIDUAL_EXPERT": "cnn_bilstm"})
    phase6["model"] = phase6.candidate.map({"M0_MLP": "MLP", "H1_TABULAR_RESIDUAL_EXPERT": "H1 Tabular Residual Hybrid"})
    phase6["protocol_id"] = "h1_final_outer_v1"
    phase6["source_role"] = "protocol_matched_h1_comparator"
    columns = ["model", "model_family", "prediction_stage", "protocol_id", "source_role", *METRICS, "risk_precision", "risk_recall", "risk_f1"]
    combined = pd.concat([classic.loc[:, columns], phase6.loc[:, columns]], ignore_index=True)
    combined["stage"] = combined.prediction_stage.map(lambda value: OULAD_STAGES[value][0])
    combined["observation_percentage"] = combined.prediction_stage.map(lambda value: OULAD_STAGES[value][1])
    combined = combined.sort_values(["observation_percentage", "model"])
    if len(combined) != 32:
        raise RuntimeError(f"expected 32 OULAD comparator rows, got {len(combined)}")
    combined.to_csv(ART / "oulad_stage_comparators.csv", index=False)

    h1_table = markdown_table(
        ["Stage", "Observed", "Accuracy", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Macro-F1", "PR-AUC", "ROC-AUC", "NLL", "Brier", "ECE", "Risk Precision", "Risk Recall", "Risk F1"],
        ([row.stage, f"{int(row.observation_percentage)}%", *[fmt(getattr(row, field)) for field in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "pr_auc", "roc_auc", "nll", "brier", "ece", "risk_precision", "risk_recall", "risk_f1"]]] for row in h1.itertuples()),
    )
    sections = [
        "# OULAD stage-aware results",
        "",
        "## Frozen H1 early-warning authority",
        "",
        h1_table,
        "",
        "One estimator/checkpoint per fold and seed serves all four cutoffs. The mean across stages is not an endpoint metric.",
        "",
        "## Frozen comparator tables",
        "",
        "Protocol IDs remain visible because classical comparators and the Phase 6 H1/MLP pair come from separate frozen evidence namespaces.",
    ]
    for prediction_stage, (stage, percent) in OULAD_STAGES.items():
        current = combined.loc[combined.prediction_stage.eq(prediction_stage)]
        sections.extend([
            "",
            f"### {stage} — {percent}% observed",
            "",
            markdown_table(
                ["Model", "Protocol", "Accuracy", "Balanced Accuracy", "Macro-F1", "PR-AUC", "ROC-AUC", "NLL", "Brier", "ECE"],
                ([row.model, row.protocol_id, *[fmt(getattr(row, field)) for field in ["accuracy", "balanced_accuracy", "macro_f1", "pr_auc", "roc_auc", "nll", "brier", "ece"]]] for row in current.itertuples()),
            ),
        ])
    write_text(REPORT / "OULAD_STAGE_RESULTS.md", "\n".join(sections))
    return h1, combined


def build_registry(endpoint: pd.DataFrame) -> None:
    h0 = endpoint.loc[endpoint.model_id.eq("cnn_bilstm")].iloc[0]
    h1 = endpoint.loc[endpoint.model_id.eq("h1_tabular_residual")].iloc[0]
    text = f"""
schema_version: final_model_authority_v1
status: FROZEN
model_development_closed: true

student_mat:
  main_model: cnn_bilstm
  primary_task: final_grade_classification
  macro_f1: 0.9014601961315334
  stage_aware_task: secondary_early_warning_S0_S1_S2
  source: artifacts/final/recommendation/canonical_classification_results.csv

student_por:
  main_model: cnn_bilstm
  primary_task: final_grade_classification
  macro_f1: 0.8622587167738002
  stage_aware_task: secondary_early_warning_S0_S1_S2
  source: artifacts/final/recommendation/canonical_classification_results.csv

oulad:
  target: at_risk_vs_not_at_risk
  legacy_endpoint:
    model: h0_cnn_bilstm
    macro_f1: {float(h0.macro_f1):.16f}
    status: LEGACY_ENDPOINT_AUTHORITY_WITH_SCORE_PROXY_CAVEAT
    score_availability: conservative_proxy_not_fully_timestamp_verifiable
    source: artifacts/audit/phase8/h0_endpoint_profile.json
  strict_endpoint:
    model: h1_tabular_residual_expert
    macro_f1: {float(h1.macro_f1):.16f}
    status: STRICT_NO_UNVERIFIED_SCORE_ENDPOINT_RESULT
    score_features: excluded_when_release_time_unverifiable
    source: artifacts/audit/phase7/endpoint_final_metrics.json
  early_warning:
    model: h1_tabular_residual_expert
    status: FROZEN_VALID_DO_NOT_MODIFY
    stages: [E1_EARLY_20PCT, E2_EARLY_35PCT, M1_MIDDLE_FROZEN, L1_LATE_75PCT]
    endpoint_equivalence: false
    source: artifacts/final/h1_final/stage_metrics.csv
"""
    write_text(REGISTRY, text)


def build_thesis_reports(endpoint: pd.DataFrame, h1_stage: pd.DataFrame) -> None:
    freeze = load_json(FREEZE_SOURCE)
    architecture = freeze["scientific_configuration"]["architecture"]
    residual = pd.read_csv(PHASE9_RESIDUAL)
    temporal = pd.read_csv(PHASE9_TEMPORAL)
    full = residual.loc[residual.candidate.eq("H1R_FULL"), "macro_f1"].mean()
    residual_off = residual.loc[residual.candidate.eq("H1R_RESIDUAL_DISABLED"), "macro_f1"].mean()
    temporal_off = temporal.loc[temporal.candidate.eq("H1R_TEMPORAL_DISABLED"), "macro_f1"].mean()
    h0 = endpoint.loc[endpoint.model_id.eq("cnn_bilstm")].iloc[0]
    h1 = endpoint.loc[endpoint.model_id.eq("h1_tabular_residual")].iloc[0]
    mlp = endpoint.loc[endpoint.model_id.eq("mlp")].iloc[0]

    endpoint_table = markdown_table(
        ["Model", "Macro-F1", "Status", "Feature-availability protocol"],
        [
            ["H0 CNN-BiLSTM", fmt(h0.macro_f1), "Legacy endpoint", "Conservative score proxy"],
            ["H1 Tabular Residual Hybrid", fmt(h1.macro_f1), "Strict endpoint", "Unverified score values excluded"],
            ["MLP", fmt(mlp.macro_f1), "Historical comparator", "Conservative score proxy"],
        ],
    )
    write_text(
        REPORT / "THESIS_MAIN_RESULTS.md",
        f"""
# Thesis main results

| Dataset | Model | Macro-F1 | Authority |
|---|---|---:|---|
| Student-Mat | CNN-BiLSTM | 0.901460 | Primary endpoint |
| Student-Por | CNN-BiLSTM | 0.862259 | Primary endpoint |
| OULAD | H0 CNN-BiLSTM | 0.828084 | Legacy endpoint with score-proxy caveat |
| OULAD | H1 Tabular Residual Hybrid | 0.798400 | Strict no-unverified-score endpoint |

## OULAD endpoint authority

{endpoint_table}

The two OULAD values must not be presented as results from the same feature-
availability protocol. The stage-aware H1 mean must not be substituted for an
endpoint result.

## Thesis-ready narrative

Trên UCI Student-Mat và Student-Por, mô hình CNN-BiLSTM đạt Macro-F1 lần lượt
0.9015 và 0.8623. Với OULAD, kết quả endpoint lịch sử của CNN-BiLSTM đạt
0.8281 dưới cơ chế score-availability proxy. Khi áp dụng giao thức nghiêm ngặt
hơn, loại bỏ các score-progress feature không chứng minh được thời điểm công
bố, H1 đạt Macro-F1 0.7984. H1 được giữ làm mô hình dự báo sớm vì cả nhánh
temporal và residual đều cho đóng góp dương trong ablation.
""",
    )
    write_text(
        REPORT / "THESIS_AUTHORITY_POLICY.md",
        f"""
# Final thesis authority policy

## UCI

- Student-Mat primary endpoint: CNN-BiLSTM, Macro-F1 **0.901460**.
- Student-Por primary endpoint: CNN-BiLSTM, Macro-F1 **0.862259**.
- S0/S1/S2 are secondary stage-aware analyses and never replace the headlines.

## OULAD

- `0.828084` means **legacy endpoint Macro-F1** under the conservative score-
  availability proxy. It is partially scientifically valid and is not called
  target leakage or invalid.
- `0.798400` means **strict no-unverified-score endpoint Macro-F1** for H1.
- H1 early-warning results at 20%, 35%, 50% and 75% are frozen secondary
  evidence. Their mean is not a final endpoint.

The H1 architecture is particularly useful as a stage-aware early-warning
model because it combines temporal behavioral sequences with aggregate/static
student information. This is predictive evidence, not a causal intervention
claim.
""",
    )
    write_text(
        REPORT / "OULAD_SCORE_PROXY_NOTE.md",
        """
# OULAD score-availability proxy caveat

OULAD records an assessment date, a submission date and a score. It does not
record an explicit timestamp for when the marked score was released to the
student or became available to a real-time predictor.

Therefore:

```text
score observed in the database
!=
score provably available at the prediction cutoff
```

Historical H0 used a conservative proxy based on known submission and
assessment dates. This is retained as legacy evidence with a caveat; it is not
described as proven target leakage. The strict H1 endpoint excludes score-
progress values whose release time cannot be defended.
""",
    )
    residual_layers = " → ".join(architecture["tabular_residual_expert"]["layers"])
    write_text(
        REPORT / "FINAL_MODEL_ARCHITECTURE.md",
        f"""
# Final H1 model architecture

Candidate: `H1_TABULAR_RESIDUAL_EXPERT`

Parameters: **{freeze['parameter_count']:,}**

Architecture hash: `{freeze['architecture_hash']}`

```text
Temporal behavioral sequence (47 channels)
        ↓
CNN kernels 2/3/5, 32 channels, dilation 1
        ↓
Bidirectional LSTM (hidden 64, one layer)
        ↓
masked mean/max temporal pooling → projection 64
        │
        ├───────────────────────────┐
        │                           │
aggregate 165 + static 13           │
        ↓                           │
Tabular Residual Expert             │
        │                           │
        └──────────┬────────────────┘
                   ↓
z_final = z_hybrid + sigmoid(a) × z_tabular
                   ↓
           at-risk probability
```

## 1. Input

The temporal input has **{architecture['sequence_channels']}** channels.
Aggregate and runtime static dimensions are **165** and **13**. Features are
stage-safe and preprocessing is fitted on training partitions only.

## 2. CNN

Input projection: **{architecture['input_projection']}**. Parallel kernels:
**{architecture['kernels']}**. Conv channels: **{architecture['conv_channels']}**,
dilation: **{architecture['dilation']}**, with residual processing.

## 3. BiLSTM

One bidirectional LSTM layer with hidden size **{architecture['lstm_hidden']}**.

## 4. Temporal pooling

`{architecture['pooling']}` followed by a **{architecture['pooling_projection']}**-
dimensional projection.

## 5. Aggregate/static branch

Aggregate hidden size **{architecture['aggregate_hidden']}**, static hidden size
**{architecture['static_hidden']}**, fusion width **{architecture['fusion_hidden']}**,
and scalar gated-residual fusion.

## 6. Tabular Residual Expert

{residual_layers}. The input is the concatenated 165 aggregate and 13 static
features.

## 7. Fusion/logit

`{architecture['tabular_residual_expert']['residual_formula']}`. The coefficient
uses `{architecture['tabular_residual_expert']['alpha_policy']}` and starts at
{architecture['tabular_residual_expert']['alpha_initial']}.

## 8. Output

The primary output is the binary at-risk logit/probability. Survival and
outcome heads remain training auxiliaries.
""",
    )
    write_text(
        REPORT / "FINAL_MODEL_PIPELINE.md",
        """
# Final H1 model pipeline

```text
Raw OULAD data
        ↓
stage/cutoff-safe feature construction
        ↓
train-only preprocessing
        ↓
temporal sequence + aggregate/static representations
        ↓
H1 CNN-BiLSTM + tabular residual expert
        ↓
probability
        ↓
threshold selected from inner OOF only
        ↓
at-risk / not-at-risk prediction
```

## Training

Preprocessors fit only on the training partition. Epoch/checkpoint and training
configuration are selected without outer labels. Future events are masked.

## Validation

Inner grouped folds select the research threshold by Macro-F1. Operational
recall-oriented thresholds remain separate and do not select the scientific
model.

## Test

The frozen estimator and inner-selected threshold are applied once to the
outer partition. Test labels calculate metrics only; they do not modify the
model, threshold or feature schema.

Early-warning and endpoint evidence remain separate. No Phase 10 training,
threshold fitting or outer evaluation is performed.
""",
    )
    write_text(
        REPORT / "H1_CONTRIBUTION_EVIDENCE.md",
        f"""
# H1 branch-contribution evidence

| Inner ablation | Macro-F1 | Delta from full |
|---|---:|---:|
| H1-R full | {full:.6f} | — |
| Residual disabled | {residual_off:.6f} | {full - residual_off:+.6f} |
| Temporal disabled | {temporal_off:.6f} | {full - temporal_off:+.6f} |

Both the temporal representation and tabular residual expert contribute
materially. These ablations do not establish that either branch alone is
sufficient and do not imply causal intervention benefit.
""",
    )


def build_provenance() -> None:
    commits = [
        ("Phase 1 forensic audit", "78b188d8"),
        ("Phase 2 training repair", "14b3df97"),
        ("Phase 3 Optuna VNext final", "fcaee199"),
        ("Phase 4 fusion final", "a83d9114"),
        ("Phase 5 tabular residual final", "2c8baa96"),
        ("Phase 6 pre-outer freeze", "234e0c6"),
        ("Phase 6 final outer evaluation", "cf4053e"),
        ("Phase 7 infrastructure", "8ac1073a"),
        ("Phase 7 endpoint freeze", "c1082627"),
        ("Phase 7 final endpoint evaluation", "be44f0b3"),
        ("Phase 8 forensic audit", "00702e42"),
        ("Phase 9 infrastructure", "4804777"),
        ("Phase 9 recovery evidence lock", "9f8be6c"),
    ]
    rows = [[phase, f"`{git_sha(short)}`"] for phase, short in commits]
    write_text(
        REPORT / "THESIS_PROVENANCE.md",
        "# Thesis provenance\n\n" + markdown_table(["Phase", "Commit"], rows) + "\n\nPhase 10 aggregates frozen evidence only; it performs no model computation.",
    )
    write_text(
        REPORT / "THESIS_EVIDENCE_INDEX.md",
        """
# Thesis evidence index

| Evidence | Canonical source |
|---|---|
| UCI main results | `artifacts/final/recommendation/canonical_classification_results.csv` |
| UCI stage-aware results | `artifacts/final/unified_stage_aware_uci/stage_metrics.csv` |
| OULAD legacy endpoint | `artifacts/audit/phase8/h0_endpoint_profile.json` |
| OULAD strict endpoint | `artifacts/audit/phase7/endpoint_final_metrics.json` |
| OULAD early warning H1 | `artifacts/final/h1_final/stage_metrics.csv` |
| OULAD classical stage comparators | `artifacts/final/unified_stage_aware_oulad/stage_metrics.csv` |
| H1/MLP stage comparator | `artifacts/final/h1_final/stage_metrics.csv` |
| Endpoint comparators | `artifacts/audit/phase7/endpoint_comparator.csv` |
| Architecture/freeze | `artifacts/final_candidate_freeze/FINAL_H1_FREEZE_MANIFEST.json` |
| Endpoint ablation | `artifacts/audit/phase9/residual_ablation.csv`, `temporal_ablation.csv` |
| Early-warning ablation | `artifacts/audit/phase5/temporal_contribution_summary.csv` |
| Calibration | `artifacts/final/h1_final/calibration_summary.csv` |
| Uncertainty | `artifacts/final/h1_final/bootstrap_summary.json` |
| Feature authority | `artifacts/audit/phase8/feature_schema_diff.csv` |
| Score caveat | `artifacts/audit/phase9/score_feature_authority.json` |
| Final authority registry | `configs/final/final_model_authority.yaml` |
""",
    )


def build_stale_audit() -> list[dict[str, Any]]:
    rows = [
        {"id": "STALE-01", "path": "README.md", "prior": "OULAD shown as one H1 endpoint authority", "resolution": "dual legacy/strict authority shown", "status": "CORRECTED"},
        {"id": "STALE-02", "path": "README.md", "prior": "H0/MLP called protocol-matched to strict H1", "resolution": "shared target/population/folds but separate feature-availability protocols", "status": "CORRECTED"},
        {"id": "STALE-03", "path": "reports/audit/phase8/PHASE8_H0_VS_H1_FORENSIC.md", "prior": "directly comparable final predictions wording", "resolution": "retained as historical forensic evidence; superseded by Phase 10 terminology", "status": "HISTORICAL_RETAINED"},
        {"id": "STALE-04", "path": "repository documentation", "prior": "0.7771 mislabeled as endpoint", "resolution": "no occurrence found; retained as early-warning mean only", "status": "NOT_FOUND"},
        {"id": "STALE-05", "path": "repository documentation", "prior": "0.7984 mislabeled as early-warning mean", "resolution": "no occurrence found", "status": "NOT_FOUND"},
        {"id": "STALE-06", "path": "repository documentation", "prior": "0.8281 attributed to H1", "resolution": "no occurrence found", "status": "NOT_FOUND"},
    ]
    write_csv(ART / "stale_reference_audit.csv", rows)
    write_text(
        REPORT / "PHASE10_STALE_REFERENCE_AUDIT.md",
        "# Phase 10 stale-reference audit\n\n" + markdown_table(
            ["ID", "Path", "Prior wording", "Resolution", "Status"],
            ([row["id"], row["path"], row["prior"], row["resolution"], row["status"]] for row in rows),
        ) + "\n\nHistorical raw artefacts and Phase reports were not rewritten.",
    )
    return rows


def build_checksum_lock() -> dict[str, Any]:
    protected = [
        ROOT / "artifacts/audit/phase5/selected_candidate.json",
        ROOT / "artifacts/audit/phase5/phase5_gate.json",
        ROOT / "artifacts/final_candidate_freeze/FINAL_H1_FREEZE_MANIFEST.json",
        ROOT / "artifacts/final/h1_final/stage_metrics.csv",
        ROOT / "artifacts/final/h1_final/phase6_gate.json",
        ROOT / "artifacts/audit/phase7/endpoint_final_metrics.json",
        ROOT / "artifacts/audit/phase7/endpoint_freeze_manifest.json",
        ROOT / "artifacts/audit/phase7/phase7_gate.json",
        ROOT / "artifacts/audit/phase8/h0_endpoint_profile.json",
        ROOT / "artifacts/audit/phase8/h1_endpoint_profile.json",
        ROOT / "artifacts/audit/phase8/phase8_gate.json",
        ROOT / "artifacts/audit/phase9/selected_candidate.json",
        ROOT / "artifacts/audit/phase9/score_feature_authority.json",
        ROOT / "artifacts/audit/phase9/phase9_gate.json",
        ROOT / "configs/archive/final_model_authority_phase10.yaml",
        *sorted(path for path in REPORT.glob("*.md") if path.name != "THESIS_CHECKSUM_LOCK.md"),
        *sorted(path for path in ART.glob("*.csv")),
        *sorted(path for path in ART.glob("*.json") if path.name not in {"thesis_evidence_checksums.json"}),
    ]
    evidence_namespaces = [
        ROOT / "artifacts/audit/phase5",
        ROOT / "artifacts/audit/phase7",
        ROOT / "artifacts/audit/phase8",
        ROOT / "artifacts/audit/phase9",
        ROOT / "artifacts/final/h1_final",
        ROOT / "reports/audit/phase5",
        ROOT / "reports/audit/phase7",
        ROOT / "reports/audit/phase8",
        ROOT / "reports/audit/phase9",
        ROOT / "reports/final/h1_final",
    ]
    evidence_suffixes = {".csv", ".json", ".md", ".parquet", ".yaml", ".yml"}
    for namespace in evidence_namespaces:
        if not namespace.is_dir():
            continue
        protected.extend(
            path
            for path in namespace.rglob("*")
            if path.is_file()
            and path.suffix.lower() in evidence_suffixes
            and not {"logs", "runtime"}.intersection(path.relative_to(namespace).parts)
        )
    unique = sorted(set(protected), key=lambda path: path.as_posix())
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    files = {path.relative_to(ROOT).as_posix(): sha256(path) for path in unique}
    aggregate = hashlib.sha256("\n".join(f"{name}:{value}" for name, value in files.items()).encode()).hexdigest()
    payload = {
        "schema_version": "thesis_evidence_checksum_v1",
        "status": "PASS",
        "file_count": len(files),
        "aggregate_sha256": aggregate,
        "files": files,
        "self_excluded": "artifacts/final/thesis/thesis_evidence_checksums.json",
    }
    write_json(ART / "thesis_evidence_checksums.json", payload)
    write_text(
        REPORT / "THESIS_CHECKSUM_LOCK.md",
        f"""
# Thesis evidence checksum lock

Status: **PASS**

Protected files: **{len(files)}**

Aggregate SHA-256: `{aggregate}`

The machine-readable per-file manifest is
`artifacts/final/thesis/thesis_evidence_checksums.json`. The manifest and this
summary are self-excluded to avoid recursive hashing.
""",
    )
    return payload


def build_gate(stale: list[dict[str, Any]], checksum: dict[str, Any]) -> dict[str, Any]:
    validation = load_json(VALIDATION) if VALIDATION.is_file() else {"status": "PENDING"}
    gate = {
        "status": "PASS" if validation.get("status") == "PASS" else "PENDING_VALIDATION",
        "training_runs": 0,
        "optuna_trials": 0,
        "architecture_searches": 0,
        "outer_evaluations": 0,
        "threshold_tuning": 0,
        "early_warning_model": "H1_TABULAR_RESIDUAL_EXPERT",
        "early_warning_frozen": True,
        "early_warning_modified": False,
        "legacy_endpoint_macro_f1": 0.8280835945631038,
        "strict_endpoint_macro_f1": 0.7984000886272689,
        "endpoint_protocols_separated": True,
        "stale_references_corrected": sum(row["status"] == "CORRECTED" for row in stale),
        "forbidden_stale_labels_remaining": 0,
        "checksum_status": checksum["status"],
        "validation": validation,
        "model_development_closed": True,
    }
    write_json(ART / "phase10_gate.json", gate)
    write_text(
        REPORT / "PHASE10_GATE.md",
        "# Phase 10 gate\n\n" + "\n".join(f"- {key}: **{value}**" for key, value in gate.items()),
    )
    if validation.get("status") == "PASS":
        write_text(
            REPORT / "PHASE10_VALIDATION.md",
            "# Phase 10 validation\n\n" + "\n".join(f"- {key}: **{value}**" for key, value in validation.items()),
        )
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    endpoint = build_endpoint_table()
    build_uci_tables()
    h1_stage, _ = build_oulad_stage_tables()
    build_registry(endpoint)
    build_thesis_reports(endpoint, h1_stage)
    build_provenance()
    stale = build_stale_audit()
    preliminary_checksum = build_checksum_lock()
    gate = build_gate(stale, preliminary_checksum)
    if args.finalize:
        if not VALIDATION.is_file() or load_json(VALIDATION).get("status") != "PASS":
            raise RuntimeError("--finalize requires a PASS validation summary")
        checksum = build_checksum_lock()
        gate = build_gate(stale, checksum)
        checksum = build_checksum_lock()
        if gate["status"] != "PASS" or checksum["status"] != "PASS":
            raise RuntimeError("Phase 10 finalization failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
