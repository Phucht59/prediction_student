# FAIRNESS_BY_GROUP

Phân tích **mô tả**, không phải can thiệp sửa bias. OOF serving join studentInfo.

CSV: `C:/hufit/student/reports/research/hybrid_superiority_v2/fairness_by_group.csv`.
Hình 35%: `C:/hufit/student/reports/research/hybrid_superiority_v2/figures/fairness_ap_by_group_35pct.png`.

## Nhóm AP thấp hơn overall ≥ 0.05 (n≥200)

| attr | group | stage | n | AP | stage AP | Δ | FN rate |
|---|---|---|---:|---:|---:|---:|---:|
| imd_band | 70-80% | 20pct | 1622 | 0.699 | 0.756 | -0.057 | 0.235 |
| imd_band | 80-90% | 20pct | 1537 | 0.694 | 0.756 | -0.062 | 0.269 |
| code_module | AAA | 20pct | 477 | 0.544 | 0.756 | -0.211 | 0.448 |
| code_module | GGG | 20pct | 1639 | 0.551 | 0.756 | -0.204 | 0.053 |
| imd_band | 80-90% | 35pct | 1492 | 0.750 | 0.807 | -0.057 | 0.341 |
| code_module | AAA | 35pct | 465 | 0.616 | 0.807 | -0.191 | 0.416 |
| code_module | GGG | 35pct | 1620 | 0.740 | 0.807 | -0.067 | 0.298 |
| code_module | AAA | 50pct | 454 | 0.605 | 0.847 | -0.241 | 0.657 |
| code_module | AAA | 75pct | 432 | 0.630 | 0.888 | -0.259 | 0.675 |
