# Source traceability

| Claim | Value | Source | Verification |
| --- | --- | --- | --- |
| Dataset size | 395 | model selection summary, dataset version | row_count |
| Class counts | 130/192/73 | frozen evidence/dataset facts | target distribution |
| Final run | a2945d79-9845-4979-b148-159f4853eca3 | `run_manifest.json` | manifest ID |
| Selected config hash | cda384...678ad | run manifest/config | SHA-256 |
| Nested Macro-F1 | 0.8781 +/- 0.0448 | selected config nested result | fold aggregation |
| Locked Macro-F1 | 0.9262 | final run manifest | recompute predictions |
| G2 locked Macro-F1 | 0.9365 | baseline_results.csv | baseline row |
| HGB nested/locked | 0.8690 / 0.9463 | config/protocol/baseline CSV | protocol separation |
| Prediction checksum | d5b6f...1f4a74 | reproducibility manifest | SHA-256 |
| Recommendation validity | 1.0 | recommendation_evaluation.json | schema metrics |
| Tests | 62 passed, 0 skipped | final audit | credentialed pytest record |
| PostgreSQL status | migration applied; 395 source/target rows | final audit and DB-first evidence | live schema/count/integration checks |
