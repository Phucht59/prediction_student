"""Generate Phase 9 reports only from completed structured evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "audit" / "phase9"
REPORT = ROOT / "reports" / "audit" / "phase9"


def load(name: str):
    return json.loads((ART / name).read_text(encoding="utf-8"))


def metric(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def write(name: str, body: str) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(body.strip() + "\n", encoding="utf-8")


def main() -> int:
    gate = load("phase9_gate.json")
    if gate.get("status") != "PASS":
        raise RuntimeError("Phase 9 reports require a PASS structured gate")
    score = load("score_feature_authority.json")
    holdout = load("holdout_availability_audit.json")
    stage_a = load("stage_a_summary.json")
    stability = load("stability_summary.json")
    selected = load("selected_candidate.json")
    freeze = load("H1_R_ENDPOINT_FREEZE_MANIFEST.json")
    confirmation = load("confirmation_metrics.json")
    comparators = load("development_comparator_context.json")
    residual = pd.read_csv(ART / "residual_ablation.csv")
    temporal = pd.read_csv(ART / "temporal_ablation.csv")
    stage_b = pd.read_csv(ART / "stage_b_attribution.csv")
    stage_c = pd.read_csv(ART / "stage_c_trials.csv")
    models = stability["models"]
    control = models["H1_R0_PHASE7_CONTROL"]
    finalist = models[selected["candidate_id"]]
    full = residual.loc[residual.candidate.eq("H1R_FULL"), "macro_f1"].mean()
    residual_off = residual.loc[residual.candidate.eq("H1R_RESIDUAL_DISABLED"), "macro_f1"].mean()
    temporal_off = temporal.loc[temporal.candidate.eq("H1R_TEMPORAL_DISABLED"), "macro_f1"].mean()
    residual_delta = float(full - residual_off)
    temporal_delta = float(full - temporal_off)

    write(
        "PHASE9_SUMMARY.md",
        f"""# Phase 9 — OULAD H1-R endpoint recovery

Status: **{gate['status']}**  
Evidence scope: **RECOVERY DEVELOPMENT EVIDENCE**

The historical score proxy was rejected because OULAD does not record the
time at which a marked score became visible. The valid, score-free H0 recipe
did not improve H1 in the preregistered Stage A comparison:

- A0 Phase 7 recipe Macro-F1: {metric(stage_a['A0']['macro_f1'])}
- A1 valid H0 recipe Macro-F1: {metric(stage_a['A1']['macro_f1'])}
- Delta: {stage_a['delta_A1_minus_A0']['macro_f1']:+.6f}

Stage B and Stage C were not triggered. The selected development candidate is
**{selected['candidate_id']}** with classification
**{selected['recovery_classification']}**. No true untouched OULAD holdout
exists in repository evidence, so no new final confirmation claim is made.
""",
    )
    write(
        "PHASE9_SCORE_AUTHORITY.md",
        f"""# Phase 9 score-feature authority

Decision: **{score['decision']}**

Official endpoint semantics remain `F2_MIDDLE`: events are legal only when
`0 <= event_day < floor(module_presentation_length * 0.50)`.

Historical score availability used `{score['historical_proxy']}`. This proves
that submission and due dates precede cutoff, but not that a marker released
the score before cutoff. Raw OULAD has no score-release timestamp. Therefore
score values and the two score-based pretraining tasks are excluded. This
decision was made without using performance as an authorization criterion.
""",
    )
    write(
        "PHASE9_HOLDOUT_AUDIT.md",
        f"""# Phase 9 holdout audit

True untouched OULAD holdout available: **NO**

The split manifest contains {holdout['future_candidate_unique_records']}
unique `future_candidate` records, but Git history contains
{holdout['prior_future_evidence_count']} prior Future-presentation artefacts.
The Phase 7 outer population has also been observed. A new random split cannot
make either population untouched. Confirmation was therefore not executed.
""",
    )
    write(
        "PHASE9_RECIPE_RECONSTRUCTION.md",
        f"""# Phase 9 recipe reconstruction

| Candidate | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| A0 Phase 7 H1 control | {metric(stage_a['A0']['macro_f1'])} | {metric(stage_a['A0']['pr_auc'])} | {metric(stage_a['A0']['roc_auc'])} | {metric(stage_a['A0']['nll'])} | {metric(stage_a['A0']['brier'])} | {metric(stage_a['A0']['ece'])} |
| A1 valid H0 recipe | {metric(stage_a['A1']['macro_f1'])} | {metric(stage_a['A1']['pr_auc'])} | {metric(stage_a['A1']['roc_auc'])} | {metric(stage_a['A1']['nll'])} | {metric(stage_a['A1']['brier'])} | {metric(stage_a['A1']['ece'])} |

A1 preserves the H1 topology and 160,492 parameters. It uses H0's valid
score-free compact feature recipe, masked train-only sequence normalization,
legal score-free temporal pretraining, standard BCE with 0.15 auxiliary
weights, and fixed eight-epoch training. It did not pass the materiality gate.
""",
    )
    write(
        "PHASE9_COMPONENT_ATTRIBUTION.md",
        f"""# Phase 9 component attribution

Status: **{stage_b.iloc[0]['status']}**

Drop-one attribution was preregistered to run only after material Stage A
recovery. Since A1 did not improve A0, attribution was stopped. The score
ablation is independently `NOT_APPLICABLE` because the proxy was rejected.
""",
    )
    write(
        "PHASE9_TUNING.md",
        f"""# Phase 9 compact training tune

Status: **{stage_c.iloc[0]['status']}**  
Optuna trials: **{gate['optuna_trials']} / 24 maximum**

The Stage C trigger was not met. No architecture or training search was run.
""",
    )
    write(
        "PHASE9_STABILITY.md",
        f"""# Phase 9 stability

Two predefined seeds were evaluated across each of three development
outer-train partitions.

| Candidate | Macro-F1 mean | std | PR-AUC | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1-R0 Phase 7 control | {metric(control['macro_f1'])} | {metric(control['std_macro_f1'])} | {metric(control['pr_auc'])} | {metric(control['roc_auc'])} | {metric(control['nll'])} | {metric(control['brier'])} | {metric(control['ece'])} |
| Selected {selected['candidate_id']} | {metric(finalist['macro_f1'])} | {metric(finalist['std_macro_f1'])} | {metric(finalist['pr_auc'])} | {metric(finalist['roc_auc'])} | {metric(finalist['nll'])} | {metric(finalist['brier'])} | {metric(finalist['ece'])} |

Recovery classification: **{selected['recovery_classification']}**.

Historical development context (not used for selection) places score-proxy
H0 at {metric(comparators['h0_p1_masked_and_next_week']['macro_f1_mean'])}
and tabular MLP at {metric(comparators['mlp_tabular_full']['macro_f1'])} on
outer-training-fold-0 inner evidence. This comparison is only partial because
those historical candidates used the score proxy rejected by Phase 9.
""",
    )
    write(
        "PHASE9_HYBRID_ABLATION.md",
        f"""# Phase 9 hybrid contribution ablation

All ablations use inner development evidence only and evaluate the same
deterministically trained model recipe.

- Full H1-R Macro-F1: {full:.6f}
- Residual disabled: {residual_off:.6f} (delta full-minus-disabled {residual_delta:+.6f})
- Temporal disabled: {temporal_off:.6f} (delta full-minus-disabled {temporal_delta:+.6f})

These results describe branch contribution; they were not used to create a
new architecture candidate.
""",
    )
    write(
        "PHASE9_SELECTED_H1R.md",
        f"""# Phase 9 selected endpoint candidate

Selected: **{selected['candidate_id']}**  
Architecture: **H1_TABULAR_RESIDUAL_EXPERT**  
Parameter count: **{selected['parameter_count']}**  
Architecture changed: **NO**  
Score proxy: **{selected['score_proxy_decision']}**

Freeze hash: `{freeze['freeze_hash']}`

This is a development freeze, not a new final endpoint claim. The historical
H0 endpoint remains the official endpoint authority pending a genuinely new
external cohort or dataset.
""",
    )
    write(
        "PHASE9_CONFIRMATION.md",
        f"""# Phase 9 confirmation

Status: **{confirmation['status']}**

No confirmation was executed because no demonstrably untouched OULAD
population remains. Outer evaluations: **{confirmation['outer_evaluations']}**.
""",
    )
    validation = load("validation_summary.json") if (ART / "validation_summary.json").is_file() else {"status": "PENDING"}
    write(
        "PHASE9_VALIDATION.md",
        "# Phase 9 validation\n\n" + "\n".join(f"- {key}: **{value}**" for key, value in validation.items()),
    )
    write(
        "PHASE9_GATE.md",
        "# Phase 9 gate\n\n" + "\n".join(f"- {key}: **{value}**" for key, value in gate.items()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
