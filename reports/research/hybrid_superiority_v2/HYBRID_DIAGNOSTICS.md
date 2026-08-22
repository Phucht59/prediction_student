# Hybrid diagnostics (development)

Outer test **không** dùng. Đây là chẩn đoán, không phải confirmation.

## C0-R UCI (diagnose 12 epoch, 79 977 params)

| Stage | VALID AP | Shuffle gap |
|---|---:|---:|
| S0 | 0.416 | 0 |
| S1 | 0.616 | 0 |
| S2 | 0.694 | −0.003 |

Availability 4-case: **pass**. Shuffle gap ~0 vì T≤2 + pooling hoán vị bất biến. G1/G2 **không** vào tabular Hybrid.

Robust 3×3: S0 0.461 / S1 0.811 / S2 0.913. S1 material pass vs CatBoost; S2 fail material 0.004.

## C3-G UCI

Screen fold-0 thua C0-R (J −5.12 vs −3.15). Robust 3×3 S2 0.884 < CatBoost 0.907. Không phải survivor.

## OULAD C0-R SPEED (10 epoch, 6 screen trial, 3 robust seed, ~163k params)

Diagnose OULAD **bỏ** trong SPEED_FINISH (GPU dành cho screen).

Best screen trial AP: 20% 0.761 / 35% 0.809 / 50% 0.858 / 75% 0.897 / 100% 0.923. Robust mean gần giống. Trần XGB/LR 0.768–0.926. Hybrid bám trần, không vượt material.

100% operational: 22522 records, 94 Withdrawn. Shortcut length→Withdrawn là sensitivity trên full enrollment, không phải panel 100% chính.

## Kết luận nhánh

- Không đào CNN sâu hơn chỉ vì tên đề tài.
- UCI survivor = C0-R, chưa đủ S2 material.
- OULAD C0-R SPEED không thắng trần tree/linear.
- Capacity UCI ~80k; OULAD SPEED best ~163k (trên target 50–200k).
- Không promote serving Hybrid.
