# Artifact authority

| Path | Category | Producer | Consumer | Reproducible |
| --- | --- | --- | --- | --- |
| `artifacts/canonical_v3/predictions/` | CANONICAL_PREDICTION | frozen canonical pipeline | prediction adapter | no retraining in release |
| `artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json` | CANONICAL_INPUT | checkpoint freeze | adapter/validator | checksum verified |
| `artifacts/final/recommendation/final_recommendation_registry.json` | FINAL_RECOMMENDATION | `build_final_recommendations.py` | final validation | yes |
| `artifacts/final/recommendation/recommendations.parquet` | FINAL_RECOMMENDATION | `build_final_recommendations.py` | audit/export | derived from canonical plans |
| `artifacts/recommend_hybrid/final/FINAL_METRICS.json` | FINAL_METRIC | `validate_phase5.py --generate-artifacts` | release validator | yes |
| `artifacts/recommend_hybrid/scientific_labeling/` | CANONICAL_ARTIFACT | Phase 1/2 scripts | diagnostic validators | protected |
| `archive/non_release_research/` | NON_RELEASE_DIAGNOSTIC | archived experiment | none | no release use |

Checksums for final recommendation files are in `artifacts/final/recommendation/checksums.sha256`.
