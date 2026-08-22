# ABLATION — ưu thế Hybrid (single-stage, 9 run)

Protocol: inner 3 fold × 3 seed, outer fold 0 firewall, FIT-only, STOP AP. **Huấn luyện một mốc** (UCI S1 / OULAD 35%), khác bản khóa mixed-state (UCI S1 serving AP 0.821).

CSV: `artifacts/research/kltn_science_fix/ablation_raw.csv` (144 run_id unique).

## OULAD 35% — kiến trúc

| ablation | AP mean | std | n |
|---|---:|---:|---:|
| **full (Hybrid khóa topology)** | **0.8091** | 0.0064 | 9 |
| concat (thay cổng) | 0.8107 | 0.0071 | 9 |
| no_aggregate | 0.8055 | 0.0068 | 9 |
| tabular_only | 0.8039 | 0.0063 | 9 |
| bilstm_only | 0.7851 | 0.0072 | 9 |
| cnn_only | 0.7742 | 0.0087 | 9 |

**Đọc theo Hybrid:** full **vượt** tabular (+0.005), BiLSTM-only (+0.024), CNN-only (+0.035). Một nhánh CNN hoặc BiLSTM đơn **không** đủ. Cổng softmax **tương đương** concat (Δ −0.002, trong nhiễu) — không làm giảm AP so với concat.

## UCI S1 — kiến trúc (grade_mode = both, như serving)

| ablation | AP mean | std | n |
|---|---:|---:|---:|
| **full** | **0.7988** | 0.0392 | 9 |
| tabular_only | 0.7927 | 0.0394 | 9 |
| no_aggregate | 0.7910 | 0.0425 | 9 |
| concat | 0.7811 | 0.0400 | 9 |
| cnn_only | 0.7730 | 0.0470 | 9 |
| bilstm_only | 0.7721 | 0.0457 | 9 |

**Đọc theo Hybrid:** full là **cao nhất** trên panel này; hơn tabular, hơn concat, hơn CNN-only và BiLSTM-only.

## UCI S1 — G1/G2 (full topology)

| grade_mode | AP mean |
|---|---:|
| **both (serving)** | **0.7988** |
| temporal_only | 0.7910 |
| aggregate_only | 0.7901 |

Cách serving hiện tại (điểm vào temporal **và** tóm tắt aggregate cùng mốc) cho AP cao nhất trong 3 arm. Không phải rò G3.

## UCI S0 (không có chuỗi)

full 0.469 ≈ tabular 0.466 — đúng thiết kế: CNN/BiLSTM tắt. Không dùng S0 để so kiến trúc lai.

## H1

Trên panel ablation: Hybrid full **≥** mọi arm một-nhánh; trên UCI S1 full là max; trên OULAD 35% full vượt tabular/CNN/BiLSTM. Bản khóa mixed-state (Chương 4) còn cao hơn panel một-mốc này (S1 0.821).
