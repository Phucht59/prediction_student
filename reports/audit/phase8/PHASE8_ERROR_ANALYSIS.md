# Error analysis

Registered-threshold confusion counts:

| Model | TN | FP | FN | TP | Risk precision | Risk recall |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 8492 | 768 | 1691 | 4427 | 0.852166 | 0.723602 |
| H1 | 8225 | 1035 | 1859 | 4259 | 0.804496 | 0.696143 |

Prediction probability correlation is
0.915454. Error overlap:

- Both correct: 11,837
- H0 correct / H1 wrong: 1,082
- H0 wrong / H1 correct: 647
- Both wrong: 1,812

H0 recovers 1,082 cases that H1 misses, while H1 recovers 647 H0 errors. This
supports a loss of endpoint predictive signal rather than a pure threshold
shift.
