# Gemini label portability

No provider was called. Frozen records were read only.

## Counts (verified)

| Panel | Cases | Review records | Models | Prompt SHA | Abstain |
|---|---:|---:|---|---|---|
| A | 300 | 1117 | flash 77 + flash-lite 1040 | `f7edfaac…24624f3` | 0 |
| B | 150 | 557 | flash-lite 557 | same | 0 |

Matches `PANEL_A_FREEZE_MANIFEST.json` and `PANEL_B_FINAL_HELDOUT_METRICS.json`. The “1500 action rows” figure is the weak-label matrix (300×5), not Gemini rows. Eligible/external-review keys = 1117.

## What Gemini saw

Verified from `17b519b` raw request `.../panel_a_batch_01/raw_requests/case_e0f9d4369e88a2e49acbb575.json`:

Present in user JSON:

- `stage`, `cutoff_day`
- `risk_band` (example `BORDERLINE`)
- `uncertainty_band` (example `HIGH`)
- `availability_flags`
- `observed_pre_cutoff_evidence` (behavioral / assessment fields)
- `candidate_actions` (already filtered)
- `contraindications`

Absent:

- numeric `risk_probability`
- `hybrid_uncertainty` float
- `seed_disagreement`
- `final_result` / outcomes
- student/module identity (blinded)

`evidence_ids` schema enum is **only** behavioral / availability keys. Across all 1674 frozen rows, `risk_probability` / `hybrid_uncertainty` / `seed_disagreement` appear **0 times** in `evidence_ids`.

## Protocol conflict

`configs/recommend_hybrid/final/llm_annotation_protocol.yaml` sets `current_model_output_visible: false`.

The live prompt still injected H1-derived **bands**. Reviewers were instructed to use only supplied evidence and could not cite risk in `evidence_ids`. Material effect of `risk_band` on scores is **UNVERIFIED**.

## Portability rule applied

```text
PORTABLE              — not used: every prompt contained H1 risk_band/uncertainty_band
CONDITIONALLY_PORTABLE — all 1674 records
NON_PORTABLE          — not used: no numeric P(risk) and evidence_ids never cite risk
UNKNOWN               — not used
```

Reasons:

1. Action catalog unchanged (five canonical IDs).
2. Judgments cite behavioral/feasibility evidence only.
3. Case payload is still H1-band-conditioned.
4. Same student-stage case is **not reconstructable from `main`** (landmark + query tables live on `17b519b`).
5. Phase4 cutoff-safe builder is not proven to reproduce `regularity_score`, `content_coverage`, `due_soon_count`, etc.
6. Panel B rows are additionally `HELDOUT_V2_ONLY`.

## Types that MAY need future labeling (do not label now)

- Cases whose C0 `risk_band`/`uncertainty_band` would differ from H1.
- Cases whose rebuilt cutoff-safe evidence moves a feasibility bit (especially `quiz_available`, assessment gaps).
- Rare cell: `TARGETED_CONTENT_REVIEW` (Panel A 115 reviews; 0 Panel A top-1).
- `EARLY_20` content-review (ineligible by policy — no Gemini needed unless policy changes).
- Any **new** action or evidence field.

Do not use Panel B as the sampling frame for those future labels.

CSV: `artifacts/recommend_hybrid/v3_context/GEMINI_LABEL_PORTABILITY.csv` (1674 rows).
