# 04 Temporal diagnostics

Inner VALID. UCI shuffle gap ~0 is expected (T≤2).
OULAD C0-R seed 42 (identity vs shuffle/reverse):

| Stage | identity AP | shuffle gap | reverse gap |
|---|---:|---:|---:|
| 20pct | 0.7575 | 0.0009 | 0.0023 |
| 35pct | 0.8102 | 0.0082 | 0.0129 |
| 50pct | 0.8584 | 0.0132 | 0.0217 |
| 75pct | 0.8980 | 0.0123 | 0.0219 |
| 100pct | 0.9239 | 0.0127 | 0.0244 |

OULAD warm shuffle gaps are **positive** (~0.008–0.013). Reverse gaps larger (~0.013–0.024).
Order/dynamics exist on OULAD; they are not large enough by themselves to clear the material margin vs XGB/LR.
See `runs/diagnose_*.json`.
