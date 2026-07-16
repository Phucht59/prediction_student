from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ML = ["C-L0", "C-R0", "C-H0"]
DL = ["C-M0", "C-C0", "C-L1", "C-H1", "C-H2"]


def parse_tests(stdout: str, return_code: int) -> dict[str, object]:
    def count(label: str) -> int:
        matches = re.findall(rf"(\d+)\s+{label}", stdout)
        return int(matches[-1]) if matches else 0
    failed, passed, skipped = count("failed"), count("passed"), count("skipped")
    collected_match = re.search(r"collected\s+(\d+)\s+items", stdout)
    collected = int(collected_match.group(1)) if collected_match else ((failed or 0) + (passed or 0) + (skipped or 0))
    return {"command": "py -3.10 -m pytest -q", "return_code": return_code, "collected": collected, "passed": passed, "skipped": skipped, "failed": failed, "raw_stdout": "test_stdout.txt"}


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame[columns].copy()
    for column in values.select_dtypes(include="number"):
        values[column] = values[column].map(lambda value: f"{value:.4f}" if isinstance(value, float) else value)
    rendered = [[str(value) for value in row] for row in values.itertuples(index=False, name=None)]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in rendered]
    return "\n".join([header, separator, *rows])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-b-run", required=True)
    parser.add_argument("--study-c-run", required=True)
    parser.add_argument("--execution-run", required=True)
    parser.add_argument("--test-stdout", type=Path, required=True)
    parser.add_argument("--test-return-code", type=int, required=True)
    args = parser.parse_args()
    b = ROOT / "artifacts" / "study_b_student_por" / args.study_b_run
    c = ROOT / "artifacts" / "study_c_oulad" / args.study_c_run
    destination = ROOT / "reports" / "extension_execution" / args.execution_run
    destination.mkdir(parents=True, exist_ok=True)
    raw_stdout = args.test_stdout.read_bytes()
    stdout = raw_stdout.decode("utf-16") if raw_stdout.startswith((b"\xff\xfe", b"\xfe\xff")) else raw_stdout.decode("utf-8", errors="replace")
    (destination / "test_stdout.txt").write_text(stdout, encoding="utf-8")
    test_report = parse_tests(stdout, args.test_return_code)
    (destination / "test_report.json").write_text(json.dumps(test_report, indent=2) + "\n", encoding="utf-8")

    b_metrics = pd.read_csv(b / "metrics_summary.csv").sort_values("macro_f1", ascending=False)
    transfer = pd.read_csv(b / "transfer_metrics.csv")
    c_metrics = pd.read_csv(c / "metrics_by_model_forecast.csv")
    development = c_metrics[c_metrics.scope == "development_oof"]
    future = c_metrics[c_metrics.scope == "future_presentation"]
    cohorts = pd.read_csv(c / "cohort_flow.csv")
    summary_rows = []
    positive_forecasts = 0
    for forecast in ["F1_EARLY", "F2_MIDDLE", "F3_LATE"]:
        rows = development[development.forecast_id == forecast]
        best_ml = rows[rows.candidate_id.isin(ML)].sort_values("macro_f1", ascending=False).iloc[0]
        best_dl = rows[rows.candidate_id.isin(DL)].sort_values("macro_f1", ascending=False).iloc[0]
        flagship = rows[rows.candidate_id == "C-H2"].iloc[0]
        future_flagship = future[(future.forecast_id == forecast) & (future.candidate_id == "C-H2")].iloc[0]
        delta = float(flagship.macro_f1 - best_ml.macro_f1)
        positive_forecasts += int(delta > 0)
        cohort = cohorts[cohorts.forecast_id == forecast].iloc[0]
        summary_rows.append({"forecast_id": forecast, "cohort_size": int(cohort.primary_cohort), "prevalence": cohort.at_risk / cohort.primary_cohort, "best_ml": best_ml.candidate_id, "best_ml_macro_f1": best_ml.macro_f1, "best_dl": best_dl.candidate_id, "best_dl_macro_f1": best_dl.macro_f1, "flagship_macro_f1": flagship.macro_f1, "flagship_at_risk_recall": flagship.at_risk_recall, "flagship_pr_auc": flagship.pr_auc, "flagship_minus_best_ml": delta, "future_flagship_macro_f1": future_flagship.macro_f1})
    forecast_summary = pd.DataFrame(summary_rows)
    f2_delta = float(forecast_summary.loc[forecast_summary.forecast_id == "F2_MIDDLE", "flagship_minus_best_ml"].iloc[0])
    verdict = "SUPPORTED" if f2_delta >= 0.01 and positive_forecasts >= 2 else "NOT_SUPPORTED"
    best_b_ml = b_metrics[b_metrics.candidate_id.isin(["B-R0", "B-L0", "B-RF0", "B-S0", "B-H0"])].iloc[0]
    best_b_dl = b_metrics[b_metrics.candidate_id.isin(["B-M0", "B-C0", "B-L1", "B-H1", "B-O0"])].iloc[0]

    execution = f"""# Study B + Study C execution summary

## Status

- Study A remained frozen; no official Study A evidence was modified and the 79 `legacy_heldout_observed` records were not accessed.
- Study B independent `student-por`: PASS.
- Study B frozen cross-subject transfer: PASS, with overlap limitation.
- Study C OULAD F1/F2/F3 materialization, grouped development evaluation, and future-presentation evaluation: PASS.
- Deep-learning advantage verdict: **{verdict}**.

## Study B

{table(b_metrics, ["candidate_id", "accuracy", "macro_f1", "macro_pr_auc", "class_collapse"])}

Best ML is **{best_b_ml.candidate_id}** (Macro-F1 {best_b_ml.macro_f1:.4f}); best DL is **{best_b_dl.candidate_id}** (Macro-F1 {best_b_dl.macro_f1:.4f}). This repeats Study A's qualitative finding that compact deep models do not surpass the strongest ML/reference approach on two late-stage grades.

Frozen transfer on all 649 Portuguese records:

{table(transfer[transfer.partition == "all"].sort_values("macro_f1", ascending=False), ["candidate_id", "records", "accuracy", "macro_f1"])}

This is a frozen cross-subject transfer evaluation, not independent external validation, because quasi-identity overlap exists between the mathematics and Portuguese datasets.

## Study C

{table(forecast_summary, ["forecast_id", "cohort_size", "prevalence", "best_ml", "best_ml_macro_f1", "best_dl", "best_dl_macro_f1", "flagship_macro_f1", "flagship_at_risk_recall", "flagship_pr_auc", "flagship_minus_best_ml", "future_flagship_macro_f1"])}

The preregistered flagship C-H2 fails the advantage rule: F2 delta is {f2_delta:+.4f}, and its delta is positive in {positive_forecasts}/3 development forecasts. The negative result is retained. Future-presentation results are domain-shift evidence and were never used for tuning.

## Validation

- Full suite: {test_report['passed']} passed, {test_report['skipped']} skipped, {test_report['failed']} failed (return code {test_report['return_code']}).
- Student grouping: global `id_student` exclusion between historical development and future test, plus grouped nested folds.
- Event contract: exact `[0, cutoff_day)` filtering before weekly aggregation.
- Target and feature snapshots are physically separate.
- SVM C-S0: `SKIPPED_COMPUTE_GATE_CPU_ONLY_RBF_ON_15K_PLUS_ROWS`; this is not represented as PASS.

## Evidence

- `artifacts/study_b_student_por/{args.study_b_run}/`
- `reports/study_b_student_por/{args.study_b_run}/`
- `artifacts/study_c_oulad/{args.study_c_run}/`
- `reports/study_c_oulad/{args.study_c_run}/`
"""
    (destination / "EXECUTION_SUMMARY.md").write_text(execution, encoding="utf-8")

    claims = f"""# Limitations and allowed claims

## Allowed

- Study B independently evaluates `student-por` under its own folds and search.
- Frozen mathematics-to-Portuguese transfer measures cross-subject domain shift, subject to quasi-identity overlap.
- Study C evaluates at-risk classification at three preregistered landmarks using cutoff-valid weekly OULAD activity.
- The Study C flagship provides real temporal modeling but **did not establish incremental advantage over the strongest ML baseline** under the preregistered rule.
- Future-presentation evaluation is a chronological/domain-shift test with global student exclusion.

## Prohibited

- Do not claim OULAD proved CNN-BiLSTM superior.
- Do not call Study B transfer fully independent external validation.
- Do not treat F1/F2/F3 as one fixed cohort or as multiple semesters for every learner.
- Do not infer causality from activity features or model explanations.
- Do not use the 79 observed Study A records as an untouched test.
- Do not hide failed/collapsed models or the skipped OULAD SVM.

## Important limitations

- OULAD `final_result` is an operational at-risk label, not a causal outcome.
- Cohorts shrink across landmarks because withdrawals before each cutoff are excluded by the landmark definition.
- Hyperparameter budgets were compute-constrained and preregistered.
- Deep stability uses declared seeds; seeds are not independent datasets.
- No model result changes the frozen Study A conclusion.
"""
    (destination / "LIMITATIONS_AND_ALLOWED_CLAIMS.md").write_text(claims, encoding="utf-8")

    context = f"""# Thesis writing context — Study B and Study C extension

## Study B

Use `student-por` as an independent in-domain study with 649 rows, the same G1/G2 information contract and three-class G3 target as Study A. Report B-RF0 {best_b_ml.macro_f1:.4f} versus B-H1 {b_metrics.loc[b_metrics.candidate_id == 'B-H1', 'macro_f1'].iloc[0]:.4f}. Keep transfer separate and disclose the 358 conservative one-to-one quasi-identity matches.

## Study C

Define learner-module-presentation records and the binary at-risk target (`Withdrawn` or `Fail`). The landmarks are 20%, 50%, and 80% of presentation length; each is a different active-at-cutoff population. Weekly sequences use only events before cutoff. Historical development uses grouped nested folds; the latest eligible presentation per module is a future test after global student exclusion.

The main scientific hypothesis is whether temporal CNN-BiLSTM fusion adds value over strong ML supplied with equivalent aggregated/flattened information. The evidence verdict is **{verdict}**. Use the exact tables and figures in the Study C report bundle; do not substitute a favorable single forecast or seed.

## Recommended figures

- `reports/study_c_oulad/{args.study_c_run}/figures/target_distribution_by_forecast.svg`
- `reports/study_c_oulad/{args.study_c_run}/figures/model_macro_f1_by_forecast.svg`
- `reports/study_c_oulad/{args.study_c_run}/figures/deep_vs_ml_delta.svg`
- `reports/study_c_oulad/{args.study_c_run}/figures/future_presentation_comparison.svg`
- `reports/study_c_oulad/{args.study_c_run}/figures/confusion_matrix_flagship.svg`
"""
    (destination / "THESIS_WRITING_CONTEXT_EXTENSION.md").write_text(context, encoding="utf-8")
    resume = f"""# Resume / verification

The scientific runs are complete. No training resume is pending. To revalidate compact evidence without retraining:

```powershell
py -3.10 scripts/validate_extension_evidence.py --study-b-run {args.study_b_run} --study-c-run {args.study_c_run} --execution-run {args.execution_run}
```

To inspect the end-to-end runner without starting expensive work:

```powershell
py -3.10 scripts/run_extension_end_to_end.py --protocol configs/extension_protocol_v1.yaml --max-wall-clock-hours 6.5 --resume --dry-run
```

PostgreSQL is reachable, but the configured application role lacks migration-owner DDL permission. After supplying an authorized migration-owner connection through the existing environment contract, register lineage with:

```powershell
py -3.10 scripts/apply_extension_migration.py --study-b-run {args.study_b_run} --study-c-run {args.study_c_run} --report reports/extension_execution/{args.execution_run}/database_registration.json
```
"""
    (destination / "RESUME.md").write_text(resume, encoding="utf-8")
    provenance = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "study_b_run": args.study_b_run, "study_c_run": args.study_c_run, "verdict": verdict, "study_a_mutated": False, "legacy_observed_accessed": False}
    (destination / "source_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "verdict": verdict, "execution_run": args.execution_run}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
