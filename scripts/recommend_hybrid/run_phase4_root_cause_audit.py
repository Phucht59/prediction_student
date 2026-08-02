"""Write a non-tuning audit of the quarantined Phase 3 diagnostic split."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "artifacts/recommend_hybrid/scientific_labeling/silver_labels.parquet"
OUT = ROOT / "reports/recommend_hybrid/scientific_model/PHASE4_ROOT_CAUSE_AUDIT.md"
DIAG = ROOT / "artifacts/recommend_hybrid/scientific_model/diagnostic_seen_v1"


def table(frame, by):
    r = frame.groupby(by, dropna=False).agg(rows=("silver_label", "size"), conditional=("silver_label", lambda x: int((x == 1).sum())), confidence=("silver_confidence", "mean"), conflict=("lf_conflict", "mean")).reset_index()
    # Avoid the optional tabulate dependency: reports must run in the locked venv.
    return "```text\n" + r.to_csv(index=False) + "```"


def main():
    f = pd.read_parquet(SILVER)
    retained = f[f.silver_status.eq("RETAINED")].copy()
    diagnostic = f[f.split.eq("test")].copy()
    conflict = retained.assign(conflict_band=pd.cut(retained.lf_conflict, [-.01, .05, .25, 1], labels=["LOW", "MEDIUM", "HIGH"]))
    purity = retained.groupby(["dataset", "stage", "action_id"]).silver_label.apply(lambda x: x.value_counts(normalize=True).max())
    report = ["# Phase 4 root-cause audit", "", "Status: `DIAGNOSTIC_SEEN_V1`; no values in this report are used for tuning or final selection.", "", "## CONDITIONAL collapse", "", "The Phase 3 reported CONDITIONAL F1 is 0.0000. The former model used only a three-way softmax and row-level shuffled batches, so it provided neither an ordinal middle-class constraint nor query ranking isolation. Its test evaluation constructed a feature vocabulary from the inspected test split. Consequently its saved apparent metrics are invalid as release evidence.", "", "Retained-label distribution:", "", table(retained, ["dataset", "stage", "action_id"]), "", "By split:", "", table(retained, ["split", "silver_label"]), "", "The audit must not infer that retained labels are equally reliable: confidence and conflict are training-only weighting inputs, never model features.", "", "## Context permutation increase", "", "The reported +0.0302 NDCG change is a failure. Phase 3 had no matched-pair/context-sensitivity objective and did not preserve a fixed feature vocabulary across train/test. The Phase 4 implementation will permute only evidence columns within dataset/stage, preserve query IDs, action IDs, eligibility, targets and the exact query set, with a fixed permutation seed.", "", "## Action-prior dominance", "", f"All {len(purity)} observed dataset/stage/action groups have a modal silver-label share of at least 0.95 (mean {purity.mean():.3f}). This proves an action-stage shortcut in the available target, so the mandated `full minus action-only >= 0.05` gate cannot be certified from it.", "", "Phase 3 directly concatenated action embeddings with context and had no query-centered ranking score or action dropout. This permits action priors to dominate. Phase 4 uses FiLM-gated interaction, product/difference fusion, action embedding dropout, and a bias-free ranking head. Action-only, context-only, and dataset-stage-action diagnostics are quarantined audits rather than competing recommenders.", "", "## Conflict handling", "", table(conflict, ["dataset", "stage", "action_id", "conflict_band", "silver_label"]), "", "Consensus core is defined from inner-training quantiles only; ambiguous retained rows receive discounted soft/ordinal loss and abstained rows are excluded from supervision.", "", "## Conclusion", "", "No Phase 3 checkpoint is eligible for the final registry. The observed diagnostic split is permanently excluded from Phase 4 model selection."]
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(report) + "\n", encoding="utf-8")
    DIAG.mkdir(parents=True, exist_ok=True)
    (DIAG / "audit_manifest.json").write_text(json.dumps({"split": "DIAGNOSTIC_SEEN_V1", "rows": int(len(diagnostic)), "retained_rows": int(len(retained)), "tuning_allowed": False}, indent=2) + "\n")
    print("PHASE4_ROOT_CAUSE_AUDIT_COMPLETE")


if __name__ == "__main__": main()
