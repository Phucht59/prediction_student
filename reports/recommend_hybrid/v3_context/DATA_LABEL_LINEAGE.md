# Data and label lineage

## End-to-end V2 lineage (verified)

```text
canonical_v3 OULAD H1 checkpoints
  artifacts/canonical_v3/checkpoints/oulad_h1_{shared,final}/outer{fold}_seed{seed}.pt
        ↓ referenced by
artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json
        ↓ OOF inference (historical; artifacts not on main)
artifacts/recommend_hybrid/causal/input/landmark_rows.parquet   # 17b519b only
        ↓ data_builder.py
risk_probability = prediction_risk_probability
hybrid_uncertainty = H2(p) binary entropy
seed_disagreement = NA
        ↓ query_evidence.py (+ raw OULAD tables if present)
assessment / VLE availability / inactivity_streak
        ↓ sampling.py  seed=2026  300 / 150  student-disjoint
Panel A cases → Gemini (1117) → Snorkel OOF → EBM
Panel B cases → Gemini (557)  → heldout scores only
```

## Feature table (V2)

Definitions: `configs/recommend_hybrid/final/feature_contract.yaml`.
Runtime columns: `src/recommend_hybrid/final/ranker.py::FEATURE_COLUMNS`.

| Feature | Family | Source (contract) | Cutoff | FIT-only? | Feas | EBM | Router | Phase4 analog |
|---|---|---|---|---|---|---|---|---|
| risk_probability | prediction | H1 5-seed OOF mean | pre-cutoff by construction | n/a (frozen model) | no | yes | via risk policy | C0 `PredictionResult.risk_probability` — **different model** |
| hybrid_uncertainty | prediction | entropy of mean P | same | n/a | no | yes | yes | no matching C0 field |
| seed_disagreement | prediction | 5-seed std | same | n/a | no | **excluded** | threshold null | none |
| course_progress | schedule | cutoff / length | schedule | no | no | yes | no | `progress` on UnifiedHybridData |
| inactivity_streak | rec evidence | last VLE day → cutoff-1 | t < cutoff | no | no | yes | no | aggregate `days_since_last_activity` / streak — **not proven equal** |
| active_day_rate | rec evidence | active days / window | train window | window on train | yes | yes | no | temporal `active_days` — not same rate |
| regularity_score | rec evidence | weekly VLE regularity | train scale | yes | yes | yes | no | none identical |
| content_coverage | rec evidence | weeks with content | train-defined | yes | yes | yes | no | `content_activity` channel only |
| quiz_activity | rec evidence | weeks with quiz | train-defined | yes | no | yes | no | temporal `quiz_activity` |
| assessments_due | rec evidence | remaining scheduled | schedule | no | no | yes | no | `assessments_due_to_date` different meaning |
| missing_assessment_count | rec evidence | due-not-submitted | cutoff | no | yes | yes | no | `missed_due_count` |
| due_soon_count | rec evidence | due before next stage | schedule | no | yes | yes | no | **no C0 analog** |
| completion_rate | rec evidence | submitted/due | cutoff | no | no | yes | no | aggregate `completion_rate` |
| quiz/vle/material available | rec evidence | metadata | schedule | no | yes | yes | no | reconstructible from raw tables |
| stage | identity | landmark | n/a | no | yes (content review) | yes | no | alias 20pct…75pct |
| label_conflict | label-derived | vote disagreement | n/a | n/a | no | **forbidden** | intended | default 0.0 on runtime features |
| ood_score | eval-derived | train detector | n/a | train-only | no | **forbidden** | intended | never populated |

Leakage checks recorded in V2 freeze: `post_cutoff_violations=0`, `student_overlap=0` (`DEVELOPMENT_FREEZE_MANIFEST.json`). `data_builder` drops `followup__*`, `treatment*`, `outcome*`, `final_result`, `action_id`.

## Labels

### Gemini

Frozen jsonl on main. Envelopes / raw requests on `17b519b` only.

### Snorkel

`artifacts/recommend_hybrid/final/weak_labels/label_model_manifest.json`:

- 1500 rows, 1117 Gemini keys, 1116 eligible retained + 1 insufficient-support
- min families = 2 (1499 rows have 2; 1 row has 1)
- Gemini source weights ≈ 0.78–0.86 across folds
- Feasibility LF weight ≈ 0.95–1.0
- Fit: outer-fold train-only, 1000 epochs, seeds 2026/2027/2028

After C0 rebase: **refit later**, do not reuse probabilistic labels as if evidence were unchanged.

### EBM targets

`expected_relevance` in `[0,3]` from Snorkel OOF. Public score `clip(pred/3,0,1)`. Calibration: none.

## Stage / cutoff

V2 intervention stages stop at `LATE_75`. `FINAL_EVALUATION` is non-intervention (H1 “FINAL” checkpoints exist but V2 catalog forbids intervening).

Phase4 `100pct` ≈ Withdrawn-length confounder (prediction report). Rec V2 already refuses 100% intervention — keep that.
