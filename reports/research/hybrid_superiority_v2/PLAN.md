# PLAN — Vá lỗ hổng khoa học (research only)

Nhánh: `research/hybrid-superiority-v2`. Không đụng `src/prediction` serving. Outer fold **không** mở.

Không có job `precision_fix` trên nhánh này (không thấy log/report). Ablation dùng **công thức khóa serving**: BCE with logits + `pos_weight` FIT-only, ngưỡng STOP F1→recall→`|t−0.5|`. Không đổi sang focal.

## Nguồn số đã có (không bịa)

| Việc | Nguồn | Script mới |
|---|---|---|
| P0.2 9-run Hybrid vs LR/RF | `test_lab/artifacts/hybrid_vnext/phase4/ROBUST_CONFIRMATION.csv` + `BASELINE_INNER_RESULTS.csv` | `research/kltn_science_fix/run_stats.py` |
| P0.2 hai số AP 0.821 vs 0.811 | `uci_final.csv` / `OVERFIT_AUDIT.json` vs hardcoded `generate_ch4_figures.py` | cùng file, mục dual-AP |
| P0.3 Fail/Withdrawn | OOF 66685 + `studentInfo.csv` | `run_label_split.py` |
| P0.4 nhóm | OOF + studentInfo | `run_fairness.py` |
| P0.5 cổng | `GATE_DIAGNOSTICS.csv` + forward 3 checkpoint serving | `run_gates.py` |
| P0.1 ablation | train mới, cùng split hash khóa | `run_ablation.py` |
| P1 PR/ROC/CM/reliability | OOF + nhãn join | `run_curves.py` |
| P1 Spearman FIT-only | UCI raw + split khóa | `run_spearman.py` |
| P1 survivorship | checkpoint serving, 100% trên id còn ở 20% | `run_survival.py` |

## Split khóa (sao chép vào artifacts research)

`C:\hufit\kltn\artifacts\hybrid\phase1\splits\` — hash khớp `SPLIT_MANIFEST.json`. Chỉ dùng `outer_fold==0` cho FIT/STOP/VALID.

## Ablation (GPU)

UCI S1 + OULAD 35%, 3 inner fold × 3 seed, HP khóa. Arm: tabular / CNN / BiLSTM / full / concat / no_aggregate; UCI thêm G1/G2 temporal_only, aggregate_only, both.

## Chương

Viết vào `reports/research/hybrid_superiority_v2/chapters/` sau khi có số. Không merge main cho đến khi user duyệt.
