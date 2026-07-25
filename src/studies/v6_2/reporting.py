from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contract import ARTIFACT_ROOT, REPORT_ROOT, ROOT, SCHEMA_VERSION, atomic_json, atomic_text


def build_regression_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in ("student_mat", "student_por"):
        path = ROOT / f"artifacts/v5_1/{dataset}/final_metrics.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        official = value["metrics"]
        if "rmse" in official and "r2" in official:
            rows.append(
                {
                    "dataset": dataset,
                    "candidate": value["candidate"],
                    "status": "OFFICIAL_POOLED_OOF",
                    "rmse": official["rmse"],
                    "r2": official["r2"],
                    "records": official["records"],
                    "aggregation": "five-seed probability/regression ensemble",
                    "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "scientific_scope": "secondary G3 regression metric; classification remains primary",
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset,
                    "candidate": value["candidate"],
                    "status": "NOT_POOLABLE",
                    "rmse": None,
                    "r2": None,
                    "records": official["records"],
                    "aggregation": "not computed for official mixed-objective ensemble",
                    "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "scientific_scope": (
                        "outer folds selected zero-weight regression in part of the "
                        "ensemble; reporting an official pooled value would mix "
                        "trained and untrained heads"
                    ),
                }
            )
        diagnostics = pd.DataFrame(value.get("seed_metrics", []))
        if {"candidate", "rmse", "r2"}.issubset(diagnostics):
            diagnostics = diagnostics.dropna(subset=["rmse", "r2"])
            for candidate, group in diagnostics.groupby("candidate", sort=True):
                rows.append(
                    {
                        "dataset": dataset,
                        "candidate": candidate,
                        "status": "SEED_LEVEL_DIAGNOSTIC_NOT_OFFICIAL_POOLED",
                        "rmse": float(group["rmse"].mean()),
                        "r2": float(group["r2"].mean()),
                        "records": int(group["records"].max()),
                        "aggregation": f"mean across {len(group)} available seed rows",
                        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "scientific_scope": (
                            "diagnostic only; not substituted for the official "
                            "classification result"
                        ),
                    }
                )
    output = pd.DataFrame(rows)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_csv(ARTIFACT_ROOT / "uci_regression_metrics.csv", index=False)
    lines = [
        "# UCI regression metrics",
        "",
        "Classification is the primary thesis task. RMSE and R² are secondary",
        "continuous-G3 diagnostics and are reported only where the registered",
        "regression head produced a scientifically valid aggregate.",
        "",
        "| Dataset | Candidate | Status | RMSE | R² | Scope |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        rmse_text = "" if row["rmse"] is None else f"{row['rmse']:.4f}"
        r2_text = "" if row["r2"] is None else f"{row['r2']:.4f}"
        lines.append(
            f"| {row['dataset']} | {row['candidate']} | {row['status']} | "
            f"{rmse_text} | {r2_text} | "
            f"{row['scientific_scope']} |"
        )
    atomic_text(REPORT_ROOT / "UCI_REGRESSION_METRICS.md", "\n".join(lines))
    return rows


def build_classification_table() -> list[dict[str, Any]]:
    source = pd.read_csv(ROOT / "artifacts/final/final_results.csv")
    selected: list[pd.DataFrame] = []
    for dataset, group in source.groupby("dataset", sort=False):
        deep = group[group["model_id"].isin(["cnn_bilstm", "cnn_only", "bilstm_only"])]
        classical = group[~group["model_id"].isin(["cnn_bilstm", "cnn_only", "bilstm_only"])]
        best = classical.sort_values(
            ["macro_f1", "model_id"], ascending=[False, True]
        ).head(1)
        selected.extend([deep, best])
    frame = pd.concat(selected, ignore_index=True)
    frame["role"] = frame["model_id"].map(
        {
            "cnn_bilstm": "OFFICIAL_CNN_BILSTM",
            "cnn_only": "KEY_ABLATION",
            "bilstm_only": "KEY_ABLATION",
        }
    ).fillna("STRONGEST_CLASSICAL_BY_MACRO_F1")
    frame["source_version"] = frame["dataset"].map(
        {
            "student_mat": "V5.1",
            "student_por": "V5.1",
            "oulad": "V6 frozen final",
        }
    )
    frame["evaluation_scope"] = frame["dataset"].map(
        {
            "student_mat": "complete OOF, 5 outer folds",
            "student_por": "complete OOF, 5 outer folds",
            "oulad": "grouped OOF, 3 outer folds",
        }
    )
    frame["seed_aggregation"] = "registered five-seed probability ensemble"
    columns = [
        "dataset",
        "model_id",
        "model",
        "role",
        "source_version",
        "result_scope",
        "evaluation_scope",
        "seed_aggregation",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "pr_auc",
        "risk_precision",
        "risk_recall",
        "risk_f1",
        "brier",
        "ece",
    ]
    frame[columns].to_csv(
        ARTIFACT_ROOT / "canonical_classification_results.csv", index=False
    )
    lines = [
        "# Canonical final classification results",
        "",
        "These are consolidated references to existing frozen OOF evidence. V6.2",
        "does not retrain, tune, or replace any prediction model.",
        "",
        "| Dataset | Role | Model | Macro-F1 | Balanced accuracy | PR-AUC | Risk F1 | Brier |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame[columns].to_dict(orient="records"):
        def fmt(name: str) -> str:
            value = row[name]
            return "" if pd.isna(value) else f"{float(value):.4f}"

        lines.append(
            f"| {row['dataset']} | {row['role']} | {row['model']} | "
            f"{fmt('macro_f1')} | {fmt('balanced_accuracy')} | "
            f"{fmt('pr_auc')} | {fmt('risk_f1')} | {fmt('brier')} |"
        )
    atomic_text(REPORT_ROOT / "FINAL_CLASSIFICATION_RESULTS.md", "\n".join(lines))
    return frame[columns].to_dict(orient="records")


def write_scientific_documents() -> None:
    atomic_text(
        REPORT_ROOT / "V6_1_ERRATA.md",
        """# V6.1 errata and interpretive addendum

The checksum-protected V6.1 artifact and report are not edited. The blank H5
evidence sentence is clarified here:

**H5 data limitation: PARTIAL.** Parameter matching, dilation correction, a
direct CNN skip, and a parallel CNN/BiLSTM candidate did not produce a stable
development-gate improvement over the full serial control. Temporal-order
destruction reduced Macro-F1, so chronology contains signal, but the incremental
local CNN signal beyond the BiLSTM and aggregate/static representation remained
small. This supports a bounded data/representation explanation, not a claim that
OULAD contains no temporal information.
""",
    )
    atomic_text(
        REPORT_ROOT / "ARCHITECTURE_AUTHORITY.md",
        """# Architecture authority

## UCI Student Performance

The authoritative Student-Mat and Student-Por model is the frozen V5.1
CNN–BiLSTM family. It treats G1/G2 as a two-step sequence, combines the temporal
encoder with the registered context branch, and uses fold-selected fusion and
auxiliary objectives. The final evidence is complete OOF aggregation over five
outer folds and the fixed seed ensemble; classification is primary.

## OULAD

The authoritative final OULAD prediction model remains the frozen V6 serial
architecture: 47 temporal channels → projection → multi-kernel CNN → residual →
bidirectional LSTM → masked pooling, combined with aggregate/static branches
through gated residual fusion. V6.1 tested parameter-matched CNN, dilation,
serial skip, and parallel CNN || BiLSTM candidates on the permitted development
partition. No candidate passed the preregistered gate, so V6.2 does **not**
switch the final architecture to parallel and performs no new outer evaluation.
""",
    )
    atomic_text(
        ROOT / "docs/VERSION_AUTHORITY.md",
        """# Version authority

| Version | Authoritative role | Supersession boundary |
|---|---|---|
| V5 | Historical controlled model evidence | Retained; not the canonical final release |
| V5.1 | Canonical UCI prediction evidence and frozen OULAD reference | Prediction artifacts remain immutable |
| V5.2–V5.4 | Diagnostic/extension evidence | Do not silently replace canonical V5.1/V6 results |
| V6 | Canonical integrated OULAD prediction/risk-profile evidence | Prediction OOF, checkpoints and model registry remain frozen |
| V6.1 | OULAD architecture diagnosis | Negative development gate retained; no new final model |
| V6.2 | Recommendation scientific validation, expert package, claim/database audit | Evaluation-only; no training, model selection, outer-test opening or Future OULAD |

The canonical result source for cross-model classification comparison is
`artifacts/final/final_results.csv`. V6.2 may reference it, but never rewrite
its metrics.
""",
    )
    atomic_text(
        REPORT_ROOT / "THESIS_CLAIM_MATRIX.md",
        """# Thesis claim matrix

| Claim | Evidence status | Allowed wording | Prohibited wording |
|---|---|---|---|
| OULAD risk prediction | Supported on historical grouped OOF | The frozen model estimates historical OOF risk | Proven future or production performance |
| CNN incremental value | Limited/partial | Capacity and dilation explain a small part; local incremental signal is limited | CNN is proven essential |
| Temporal order | Supported diagnostically | Order destruction reduced development Macro-F1 | Causal long-term dependency proof |
| Recommendation technical validity | Supported by deterministic rules, lineage, abstention and workload checks | Technically valid decision-support prototype | Effective intervention |
| Expert validity | PENDING_REAL_EXPERT_LABELS | Expert package is ready; metrics pending | Expert-approved or clinically/educationally effective |
| Withdrawal mechanism | Not reliable | Exploratory prediction output only | Engagement mechanism or withdrawal-specific action trigger |
| Causal impact | Not evaluated | No causal effectiveness claim | Prevents dropout or improves outcomes |
| External/Future OULAD generalization | Not evaluated; locked | Future OULAD remained unaccessed | Generalizes to future cohorts |
""",
    )
    atomic_text(
        REPORT_ROOT / "PROPOSAL_COMPLIANCE_MATRIX.md",
        """# Proposal compliance matrix

| Requirement | Status | Evidence / justification |
|---|---|---|
| Student performance/risk prediction on registered datasets | FULL | Frozen V5.1/V6 OOF artifacts and canonical classification table |
| CNN–BiLSTM implementation and comparison | FULL | Official models, CNN-only/BiLSTM-only and classical comparators |
| OULAD architecture diagnosis | FULL | V6.1 controlled development-only experiments; negative gate preserved |
| R² and RMSE reporting | PARTIAL_WITH_JUSTIFICATION | Portuguese official aggregate exists; Math official mixed-objective ensemble is not poolable |
| Recommendation generation | FULL | V6.2 grounded pre-cutoff feature lineage and bounded rule policy |
| Recommendation expert evaluation | PENDING | Blind 60-case, two-reviewer package ready; no real labels supplied |
| Recommendation effectiveness | NOT_APPLICABLE_TO_FINAL | Requires prospective/outcome evaluation; not claimed |
| Withdrawal-specific intervention | NOT_APPLICABLE_TO_FINAL | Head fails reliability gate and is disabled for mechanism/action use |
| Database-backed evidence | PARTIAL_WITH_JUSTIFICATION | Additive migration and audit supplied; application requires explicit disposable DSN |
| Future OULAD evaluation | NOT_APPLICABLE_TO_FINAL | Explicitly locked by protocol |
| Reproducibility and checksums | FULL | One-command non-training validator and V6.2-only checksum manifest |
""",
    )


def build_claim_audit() -> dict[str, Any]:
    records = [
        {
            "claim": "recommendation_effectiveness",
            "status": "NOT_ESTABLISHED",
            "authoritative_boundary": "technical validity and expert-evaluation readiness only",
        },
        {
            "claim": "causal_student_outcome_improvement",
            "status": "PROHIBITED",
            "authoritative_boundary": "no prospective or randomized intervention evidence",
        },
        {
            "claim": "cnn_incremental_contribution_oulad",
            "status": "LIMITED_PARTIAL",
            "authoritative_boundary": "V6.1 development diagnosis; no new outer gate",
        },
        {
            "claim": "withdrawal_specific_mechanism",
            "status": "DISABLED",
            "authoritative_boundary": "near-zero recall; exploratory output only",
        },
        {
            "claim": "future_oulad_generalization",
            "status": "NOT_EVALUATED_LOCKED",
            "authoritative_boundary": "historical grouped OOF only",
        },
    ]
    value = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_BOUNDED",
        "claims": records,
        "causal_effectiveness_claimed": False,
        "expert_approval_claimed": False,
        "future_generalization_claimed": False,
    }
    atomic_json(ARTIFACT_ROOT / "claim_audit.json", value)
    return value


def build_reports() -> dict[str, Any]:
    regression = build_regression_summary()
    classification = build_classification_table()
    write_scientific_documents()
    claim = build_claim_audit()
    return {
        "regression_rows": len(regression),
        "classification_rows": len(classification),
        "claim_audit": claim["status"],
    }


__all__ = [
    "build_claim_audit",
    "build_classification_table",
    "build_regression_summary",
    "build_reports",
    "write_scientific_documents",
]
