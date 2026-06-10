# V19 Ablation Study Report

Protocol: strict train/validation/test split with paper features. Validation selects best epoch; test is evaluated after epoch selection.

| dataset | model | seed | best_epoch | val_f1_macro | test_accuracy | test_f1_macro |
| --- | --- | --- | --- | --- | --- | --- |
| student-mat | CNN-only | 42 | 17 | 0.6872 | 0.7333 | 0.7003 |
| student-mat | BiLSTM-only | 42 | 29 | 0.6298 | 0.6000 | 0.5729 |
| student-mat | CNN+BiLSTM | 42 | 27 | 0.6893 | 0.7667 | 0.7556 |
| student-por | CNN-only | 42 | 22 | 0.7600 | 0.7041 | 0.6770 |
| student-por | BiLSTM-only | 42 | 22 | 0.7092 | 0.6735 | 0.6805 |
| student-por | CNN+BiLSTM | 42 | 22 | 0.7637 | 0.6837 | 0.6495 |
| xapi | CNN-only | 42 | 12 | 0.7587 | 0.7222 | 0.7254 |
| xapi | BiLSTM-only | 42 | 25 | 0.7429 | 0.7361 | 0.7391 |
| xapi | CNN+BiLSTM | 42 | 24 | 0.7184 | 0.7361 | 0.7391 |

## Notes

- This ablation tests whether CNN+BiLSTM is consistently better than CNN-only or BiLSTM-only under the same strict split.
- Results must be read honestly; if CNN+BiLSTM does not win on a dataset, the report should say so.
