# 03 Joint screen

Fold 0, seeds 42/1201/2026, inner VALID. Constrained J (warm losses heavily penalized).

| Domain | Candidate | mean J | mean warm losses | mean min r |
|---|---|---:|---:|---:|
| uci | C0-R | -65.913 | 2.00 | -8.846 |
| uci | C1-R | -79.454 | 2.00 | -16.081 |
| uci | C2-S | -66.486 | 2.00 | -9.207 |
| uci | C3-G | -70.022 | 2.00 | -11.449 |
| oulad | C0-R | -24.959 | 1.00 | -0.065 |
| oulad | C1-R | -41.769 | 1.67 | -0.164 |
| oulad | C2-S | -50.106 | 2.00 | -0.160 |
| oulad | C3-G | -58.612 | 2.33 | -0.281 |

UCI: all four backbones lose both warm stages vs CatBoost 3×3 ceiling under this screen budget.
OULAD: **C0-R** is the least-bad backbone (fewest warm losses).
