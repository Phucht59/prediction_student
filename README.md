# Student Outcome and Risk Prediction Thesis

This repository contains the final, reproducible evidence for a thesis on student outcome prediction and risk-informed decision support. The public release separates canonical final evidence from historical research material without changing frozen results.

## Thesis objective

The system predicts academic outcomes for Student-Mat and Student-Por and identifies at-risk learners in OULAD. It also produces bounded, human-reviewable recommendation plans from cutoff-safe observed features.

## Datasets

- **Student-Mat** and **Student-Por**: three-class outcome prediction.
- **OULAD**: early risk prediction at the registered F2_MIDDLE cutoff.

## Final CNN-BiLSTM prediction models

The final registry contains one frozen CNN-BiLSTM model for each dataset: `cnn_bilstm_mat`, `cnn_bilstm_por`, and `cnn_bilstm_oulad`. V5.1 remains the canonical UCI evidence and V6 remains the canonical OULAD risk-profile evidence. Their paths are intentionally retained for replay and checksum verification.

## Recommendation decision-support system

The recommendation layer consumes risk information and real pre-cutoff observations. It may abstain when evidence is incomplete and does not infer observed learner behaviour from prediction probabilities.

Recommendation is evaluated offline for semantic grounding, cutoff-safe lineage, determinism, consistency, abstention, workload and safety. Human/user evaluation and causal intervention effectiveness are outside the current study and remain future work.

## Final metrics

| Dataset | Final model | Macro-F1 | Balanced accuracy | PR-AUC |
|---|---|---:|---:|---:|
| Student-Mat | CNN-BiLSTM | 0.9015 | 0.9021 | 0.9442 |
| Student-Por | CNN-BiLSTM | 0.8623 | 0.8676 | 0.9147 |
| OULAD | CNN-BiLSTM | 0.8281 | 0.8203 | 0.8934 |

See [the final results report](reports/final/FINAL_MODEL_RESULTS.md) for the authoritative metric tables, uncertainty, provenance, and claim boundaries.

## Repository structure

- `src/models/`, `src/data/`, `src/evaluation/`, `src/recommendation/`, `src/database/`: final system code.
- `configs/final/`, `artifacts/final/`, `reports/final/`, `database/final/`: canonical release configuration and evidence.
- `docs/`: methodology, database, version authority, and operating guidance.
- `tests/`: replay, contract, leakage, and system validation.
- `lab/`: historical experiments, diagnostics, and future-evaluation material; it never supersedes canonical final results.

## Validation

```powershell
python project.py final status
python project.py final report
python project.py final validate
```

These commands validate existing evidence; they do not retrain models. Future OULAD remains locked.

## Scientific limitations

Results apply only to the registered targets, splits, seeds, data, and evaluation protocol. Prediction and recommendation outputs are not causal claims. Recommendation effectiveness requires future independent human/user evaluation and outcome-based intervention study. See [Claim Boundaries](reports/final/CLAIM_BOUNDARIES.md).
