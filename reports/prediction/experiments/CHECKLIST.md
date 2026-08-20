# Checklist khoa học — Hybrid CNN–BiLSTM (một mô hình, một spec)

Mọi số dưới đây là thực nghiệm. S0 và OULAD 20% thấp hơn baseline **được giữ**, không cherry-pick. Không bịa dataset/metric. Không push. Production không đổi.

**Một mô hình:** Hybrid CNN∥BiLSTM, `architecture_id=C0`, hparams chỉ `TRAINING_CONFIG.json`. Fold/seed = đánh giá spec đó, không phải model mới.

---

## Nested Cross-Validation / 5-fold Outer Test — DONE

- **UCI:** 5 outer fold chính thức. Train trên 4 fold kia; STOP tách trong train; **test = outer fold**. Threshold từ STOP. `hpo_on_outer=false`.
- **OULAD:** outer chính thức chỉ có **3** fold (không bịa thành 5). Cùng protocol nested.

Mean PR-AUC nested outer:

| Dataset | Level | Hybrid | LR | RF |
| --- | --- | ---: | ---: | ---: |
| UCI | S0 | 0.493 | 0.492 | **0.497** |
| UCI | S1 | **0.813** | 0.745 | 0.770 |
| UCI | S2 | **0.909** | 0.847 | 0.907 |
| OULAD | 20% | 0.749 | **0.754** | 0.753 |
| OULAD | 35% | **0.804** | 0.792 | 0.796 |
| OULAD | 50% | **0.847** | 0.835 | 0.841 |
| OULAD | 75% | **0.891** | 0.882 | 0.888 |
| OULAD | 100% | **0.925** | 0.911 | 0.917 |

File: `artifacts/experiments/cv5/nested_outer/nested_outer_metrics.csv`

Phát triển 5-fold (outer 0 loại khỏi train) vẫn là bảng canonical seed 42: `artifacts/experiments/cv5/hybrid_vs_baselines.csv`.

## Early Stopping đúng protocol — DONE

STOP macro PR-AUC, patience 8, max 24, AdamW, clip 1.0. Checkpoint tốt nhất trên STOP, không trên outer/test.

## Không dùng Outer-Test để HPO / chọn model / threshold — DONE

`TRAINING_CONFIG.json` đóng băng. Threshold F1→recall→|t−0.5| trên STOP. Sensitivity `used_for_selection=false`. Nested outer chỉ **chấm**, không search.

## Nhiều Random Seeds — DONE

Seeds khóa 42 / 1201 / 2026 (Phase 4 3×3 và UCI 5-fold diagnostic). Nested outer seed 42 (cùng spec). Không chọn seed đẹp.

## Leakage Audit toàn bộ pipeline — DONE

Split lấy từ `codex/backup-hybrid-phase8-2026-08-17`, hash khớp 4 file inner/outer. FIT/STOP/VALID không chồng; group không chồng; outer0 overlap = 0. G3 không predictor; OULAD `t < cutoff`.  
`LEAKAGE_FREE=true` — `artifacts/experiments/validation/LEAKAGE_OVERFIT_AUDIT.json`

## Overfitting Analysis theo fold, seed, stage — DONE

Locked Phase 4: UCI S0 gap **HIGH 0.125**; S1/S2 MODERATE; OULAD LOW.  
5-fold train−VALID gap (diagnostic): UCI S0 ~0.16; OULAD ~0.03.  
File: `OVERFIT_AUDIT.json`, `scientific/metrics.csv` cột `overfit_gap`.

## So sánh Baseline (RF, LR) — DONE

Cùng ID/nhãn/FIT-only/STOP threshold. LR/RF: tabular. Hybrid: CNN∥BiLSTM trên chuỗi.  
Macro 5-fold dev: UCI Hybrid **0.726** > RF 0.717; OULAD Hybrid **0.843** > RF 0.838.

## Statistical Significance

### Bootstrap CI — DONE
`scientific/stat_tests.csv`: `hybrid_pr_lo/hi`, `pr_delta_lo/hi`, 300 bootstrap.

### McNemar — DONE
Cùng file, `mcnemar_p`, paired hard labels.

### DeLong — DONE
`delong_p`, `delong_delta_roc`.

### Effect size + CI — DONE
ΔPR-AUC bootstrap CI; Cohen’s g từ McNemar.  
Ví dụ (mean across jobs, Hybrid−RF): UCI S1 ΔPR-AUC +0.025; OULAD 20% −0.001; OULAD 100% +0.005.

## Calibration Analysis

### Brier Score — DONE
Trong `metrics_valid.csv` / `cv5_metrics.csv` / nested outer (`brier`).

### ECE — DONE
Cùng bảng (`ece`). UCI S0 ECE cao (~0.20) — kết quả **không thuận lợi**, giữ.

### Calibration Plot — DONE
`artifacts/experiments/validation/plots/calib_*.png` (84 ảnh gồm CM).

## Ablation Study

### CNN-only / BiLSTM-only / Tabular-only / Hybrid — DONE

Không train 3 mạng riêng (sẽ phá “một mô hình”). Ablation = chấm **cùng Hybrid đã train**, chỉ một nhánh.

UCI: S0 mọi nhánh = tabular (0.46); S2 temporal (CNN/BiLSTM ~0.90) >> tabular (0.51); full Hybrid 0.91.  
OULAD: full Hybrid > từng nhánh; BiLSTM mạnh dần khi cutoff tăng.  
Permutation UCI fold 0: shuffle **temporal** làm S2 PR-AUC 0.885→0.321 (Δ 0.56); S0 temporal Δ=0 (đúng contract).  
Files: `ablation_branch.csv`, `nested_outer/permutation_uci_fold0.csv`.

## Robustness / Sensitivity Analysis — DONE

UCI fold 0: dropout {0.30, frozen 0.406, 0.50} và lr ×{0.5,1,2}. **Không chọn** cấu hình mới.  
`scientific/sensitivity_uci_fold0.csv`

## Error Analysis (FP/FN, case study) — DONE

Confusion mean + histogram theo TP/FP/FN/TN. Case study hashed `record_id` + score + threshold (không bịa hồ sơ).  
`nested_outer/error_case_studies.csv`, `plots/cm_*.png`, `figures/error_hist_*.png`

## Subgroup & Fairness Analysis — DONE

UCI: sex, school, subject. OULAD: gender, disability, imd_band, code_module, age_band. Gap max−min TPR/FPR/PR-AUC.  
`subgroup.csv`, `figures/fairness_*.png`

## Explainability

### Feature importance — DONE
Permutation block static / aggregate / temporal. Gate masses 3 nhánh.

### SHAP — DONE
KernelSHAP RF trên **cùng** packed UCI S2 (giải thích comparator tabular). Không DeepSHAP (tránh mô hình giải thích thứ hai).  
`shap_rf_uci_s2.csv`, `figures/shap_rf_uci_s2.png`, `figures/fusion_gate_*.png`

## Cross-Dataset Validation (UCI + OULAD) — DONE

Một architecture, hai dataset, nhiều information level.

## External Validation — NOT AVAILABLE

Không có dataset thứ ba hợp lệ trên máy. **Không tạo data giả.**

## Không cherry-pick / không bịa — DONE

S0 và 20% thấp hơn baseline được **báo cáo và chấp nhận**. Nested outer cùng pattern: UCI S0 Hybrid 0.493 vs RF 0.497; OULAD 20% Hybrid 0.749 vs LR 0.754.

## Kết luận (chỉ từ số trên)

Một Hybrid CNN–BiLSTM, một bộ tham số đóng băng, **thắng macro** UCI và OULAD so với LR/RF. Hai mốc sớm (S0, 20%) thấp hơn một ít — chấp nhận. Không nhân bản mô hình để vá hai mốc đó.
