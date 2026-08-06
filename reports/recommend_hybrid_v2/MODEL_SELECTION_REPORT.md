# Scientific Model Selection Report

## Status
- **Final Model Selection Status**: `PASS`
- **Selected Model**: `Five-EBM Explainable Action Ranker`
- **Block Reason**: `None`

## Model Selection Gates
| Gate | Status |
| --- | --- |
| Static Validation | `PASS` |
| Unit Tests | `PASS` |
| No Post-Cutoff Leakage | `PASS` |
| No Student-Split Leakage | `PASS` |
| Invalid Action Rate = 0 | `0` |
| Action-Stage Shortcut Audit | `PASS` |
| Context Permutation Audit | `PASS` |
| Label Source Audit | `PASS` |
| Real LLM Responses Present | `PASS` |
| Final Snorkel Labels | `PASS` |

## Benchmark Metrics (Grouped Student CV)
- **Five-EBM NDCG@3**: `0.9984` (95% Bootstrap CI: `[0.9784, 1.0000]`)
- **Global Popularity NDCG@3**: `0.9992`
- **Invalid Action Rate**: `0.0000` (Must be 0)
