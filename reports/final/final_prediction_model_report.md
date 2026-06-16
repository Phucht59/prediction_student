# Final xAPI Prediction Model

## Final Decision
The final xAPI deep model is reverted to `gated_fusion_v28` with `low_f1_tuned`.

| dataset | model | prediction mode | Macro F1 | Recall Low | F1 Low |
|---|---|---|---:|---:|---:|
| xAPI | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |

## Technical Notes
- The final model remains aligned with the CNN + BiLSTM thesis direction.
- The xAPI model uses gated context fusion because xAPI contains both behavioral sequence features and categorical/contextual features.
- Machine-learning baselines are comparison-only. They are not used as teachers, distillation sources, pseudo-label sources, baseline probability sources, or feature-importance sources.
- Locked test is used only for final evaluation.
- Threshold tuning uses CV/OOF probabilities, not locked test.
- ADASYN is not used.
- student-combine is not used.
- Regression head is not claimed.

## Baseline Comparison
| dataset | model type | model | Macro F1 | Recall Low | F1 Low |
|---|---|---|---:|---:|---:|
| xAPI | deep | gated_fusion_v28 | 0.7541 | 0.8846 | 0.8214 |
| xAPI | baseline | RandomForestClassifier | 0.8465 | not_available | not_available |

## Conclusion
The xAPI final model is `gated_fusion_v28 + low_f1_tuned`. Later experimental variants are not used as final models.
