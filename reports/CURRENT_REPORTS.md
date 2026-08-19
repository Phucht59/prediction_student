# Current Report Registry

This registry separates current authority from immutable historical evidence.
It is the starting point for thesis and repository readers.

## Current prediction authority

Thesis-final prediction model (2026-08 owner finalization):

| Topic | Current source |
|---|---|
| Public Hybrid C0 | `src/prediction/model/hybrid.py` |
| Config / registry | `configs/prediction/hybrid_final.json`, `configs/prediction/registry.json` |
| Canonical report | `reports/prediction/final/FINAL_PREDICTION_MODEL_REPORT.md` |
| Result tables | `reports/prediction/final/uci_final.csv`, `oulad_final.csv` |
| Decision manifest | `artifacts/prediction/final/FINALIZATION_DECISION.json` |

Older rows below remain historical provenance. They are not the active Hybrid prediction authority.

## Current authority (other subsystems)

| Topic | Current source |
|---|---|
| Historical pre-Phase4 prediction configs | `configs/final/final_model_authority.yaml` |
| Student-Mat and Student-Por configurations | `configs/final/cnn_bilstm_mat.yaml`, `configs/final/cnn_bilstm_por.yaml` |
| OULAD H1 configuration | `configs/final/h1_tabular_residual_oulad.yaml` |
| Thesis result set | `reports/final/thesis_v3/` |
| Final imbalance evidence | `reports/final/imbalance_evidence/` |
| Recommendation V2 release | `artifacts/recommend_hybrid/final/release/FINAL_RELEASE_MANIFEST.json` and `reports/recommend_hybrid/final/FINAL_RELEASE_SUMMARY.md` |
| Database schema and current-state validation | `database/final/FINAL_SCHEMA_CONTRACT.md`, `tests/database/test_schema_contract.py`, and a fresh read-only inventory |

## Required interpretation boundaries

- OULAD authority is `H1_TABULAR_RESIDUAL_EXPERT`, `STRICT_REAL_TIME`, 160,492 parameters, and Macro-F1 `0.8940709888551659`.
- The authority-equivalent OULAD class-weight challenge completed 15 jobs: its Macro-F1 was `0.8859829540176192` versus FIXED_NONE `0.8942454181505014`; it was not promoted.
- Recommendation V2 is runtime-authorized only for the hashes in its final release manifest. Historical development manifests that say `runtime_authorized=false` remain immutable provenance, not the current release decision.
- Do not cite the July 23 post-cutover audit (13 base tables, 2 views, and 75,576 rows) as the current database inventory. Migration 011 subsequently added three expert-review tables, while the current live schema test also includes `ml.prediction` and expects 17 base tables. The schema contract still says 16, so the exact live count must be refreshed with a read-only inventory before it is stated in the thesis. The 29-table/zero-row audit is a pre-cutover historical snapshot.

## Historical files retained for provenance

The following are intentionally retained and must not be presented as current state:

- `reports/final/DATABASE_CURRENT_STATE_AUDIT.md`: pre-cutover legacy snapshot.
- `reports/recommend_hybrid/final/FIVE_EBM_MODEL_REPORT.md`: historical Panel-A report. Its 14-feature/Isotonic description is superseded by the frozen final ranker manifest (16 features and `NONE_RAW_EBM_SELECTED`).
- `reports/final/thesis/`: checksum-locked historical thesis evidence. Use `reports/final/thesis_v3/` for the current thesis result set.

These files are retained because their paths and hashes are included in frozen checksum manifests. Editing, moving, or deleting them would invalidate reproducibility evidence.
