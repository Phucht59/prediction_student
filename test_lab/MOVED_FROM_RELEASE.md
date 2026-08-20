# Trees moved out of the release surface

Moved 2026-08-19. Bytes preserved. Not current Hybrid C0 or Recommendation V3.

| Original path | Why moved |
|---|---|
| `experiments/final_hybrid_imbalance_authority_equivalent` | Pre-C0 imbalance experiment; not the thesis Hybrid |
| `artifacts/final` | Old CNN-BiLSTM / H1 / unified-stage freeze |
| `artifacts/canonical_v3` | Historical H1 checkpoints/metrics |
| `artifacts/final_candidate_freeze` | Historical H1 freeze |
| `artifacts/final_release` | Historical registry/replay, not Phase4 C0 |
| `artifacts/hybrid_vnext` | Phase2–4 experiment outputs |
| `artifacts/recommendation` | Pre-V3 recommendation overlay |
| `artifacts/audit` | Phase1–9 research audits |
| `artifacts/migration` | Phase8 restore probes |
| `reports/final` | Historical thesis_v3 / pre-C0 reports |
| `reports/final_candidate` | Historical H1 freeze writeup |
| `reports/hybrid_vnext` | Phase2–4 reports |
| `reports/audit` | Phase research reports |
| `reports/migration` | Restore notes |
| `reports/project_cleanup` | Old file audit |
| `tests/hybrid_vnext` | Experiment contract tests (moved with hybrid_vnext artifacts) |
| `tests/recommendation/*` except `final/` | Pre-V3 recommendation tests |
| `scripts/hybrid_vnext` | Experiment runners |
| `scripts/recommendation` | Pre-V3 scripts |
| `configs/archive` | Phase10 archive YAML |

Left in the normal tree:

- `src/prediction`, `configs/prediction`, `artifacts/prediction`, `reports/prediction/final`
- `src/recommend_hybrid/v3`, `artifacts/recommend_hybrid/v3`, `reports/recommend_hybrid/v3`
- `artifacts/recommend_hybrid/final` (V2 frozen labels still used by V3 portability)
- `tests/prediction`, `tests/recommend_hybrid/v3`, `tests/recommendation/final`
- `experiments/hybrid_vnext` (C0 topology helper package; kept because it is small source, not a run dump)
