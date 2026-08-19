# HYBRID_VNEXT_RESEARCH_AUDIT

**Phase:** nghiên cứu / forensic — không thiết kế kiến trúc cuối, không train, không Optuna, không outer test mới.  
**Authority khảo sát:** `main` + nguồn Phase8 `codex/backup-hybrid-phase8-2026-08-17` (`C:\hufit\kltn`).  
**Đối tượng:** một public architecture `Hybrid` (CNN + BiLSTM + tabular/context + fusion + binary risk head) dùng chung cho UCI Combined và OULAD.  
**Không phải đối tượng tuyên bố:** model thesis cũ 3-class `cnn_bilstm_*`, H0 F2 0.828, H1 residual 0.894.

Outer numbers bên dưới chỉ dùng cho **DIAGNOSIS**, không dùng cho model selection.

---

## A. Executive conclusion

Hybrid hiện tại **không có ưu thế rõ ràng** trước baseline. Trên outer đóng băng, Hybrid đứng hạng 4 UCI (thua RF/XGB/LR), hạng 3 OULAD early và FINAL-100 (thua XGB/RF). Bootstrap OULAD: Hybrid − XGB PR-AUC âm và CI không chứa 0.

**Đã xác nhận:** (1) topology thật là CNN ∥ BiLSTM song song, không phải CNN→BiLSTM; (2) `aggregate_available` đang khóa nhánh BiLSTM; (3) baseline nhận aggregate + last/mean/max của sequence — inductive advantage hợp lệ; (4) UCI chỉ có 0/1/2 timestep nên CNN/BiLSTM không có lợi thế inductive; (5) FINAL-100 có shortcut length≈Withdrawn (99.89% chuỗi ngắn là Withdrawn); (6) architecture hiện tại **chưa** được Optuna tune; (7) ~87% tham số nằm ở CNN+BiLSTM trong khi tín hiệu mạnh nằm ở tabular.

**Hypothesis mạnh, chưa đo trên đúng F3 hiện tại:** fusion sụp vào tabular; temporal order ít đóng góp; 16 epoch / LR 8e-4 chưa tối ưu; entropy floor 0.002 quá yếu.

**Điểm yếu cần sửa đầu tiên:** availability mapping + inductive fairness của fusion (để temporal được dùng đúng), không phải đào sâu CNN.

---

## B. Current pipeline map

Cùng một class `Hybrid` / `HybridConfig`. Dataset chỉ khác chiều input, preprocessor fit, checkpoint và threshold.

```text
RAW
  UCI: student-mat.csv + student-por.csv
  OULAD: studentInfo + registration + courses + VLE + assessments
        ↓
TARGET (binary risk)
  UCI: G3 < 10
  OULAD: final_result ∈ {Fail, Withdrawn}
        ↓
GROUP ID
  UCI: quasi-identity → global_student_group (662 nhóm, 366 trùng MAT∩POR)
  OULAD: id_student
        ↓
OUTER SPLIT  (frozen parquet, StratifiedGroupKFold, seed 42)
INNER SPLIT  (FIT / STOP / VALID; outer test không vào inner)
        ↓
PREPROCESSING  (FIT-only)
  static/context: median impute + StandardScaler + OneHot
  aggregate: mean/std trên FIT rows có aggregate_available=1
  OULAD temporal: MaskedStandardScaler trên FIT
  rồi D3_both_safe (OULAD) / D0 (UCI)
        ↓
FEATURE VIEWS
  static | temporal[T,C] + mask + lengths | aggregate + aggregate_available | progress
  UCI:  T=2, C=1 (G1/G2 /20); aggregate 5 kênh grade-summary; S0 aggregate_available=0
  OULAD early: T≈ tuần đến cutoff 20/35/50/75%; C=11; aggregate 13; available≡1
  OULAD FINAL-100: observation_end = min(course_end, unregistration)
        ↓
MODEL INPUT  →  temporal_adapter
                  ├─ ResidualCNN (k=2, dilations 1,2, ch=128) → mean+max → d=96
                  └─ BiLSTM (h=128, 1 layer, packed)          → mean+max → d=96
              static_projector  ─┐
              aggregate_projector ┴→ tabular = h_static + h_agg * aggregate_available
        ↓
FUSION F3  3-way softmax(tabular, h_cnn, h_bilstm) + entropy floor 0.002
        ↓
HEAD  LayerNorm → 128 → GELU → Dropout → 1 logit → sigmoid P(risk)
        ↓
LOSS  BCEWithLogits + FIT-only pos_weight + fusion_regularization
OPT   AdamW lr=8e-4, wd=2e-4, clip=1.0, batch=256, max_epochs=16, patience=5
        ↓
CHECKPOINT  inner: restore best STOP macro-PR-AUC
            final reconstruction: median(inner best epoch) fixed-epoch last state
        ↓
THRESHOLD  STOP-only; rank risk-F1 → recall → |t-0.5|
CALIBRATION  không có (không isotonic/temperature)
        ↓
OUTER PREDICTION (historical freeze) → PR-AUC / F1 / ECE
```

Hai fitted instance OULAD (`oulad_early`, `oulad_final`) **cùng topology**, khác cohort/endpoint và weights.

---

## C. Leakage audit

| Risk | Location | Status | Evidence | Severity | Required action |
|---|---|---|---|---|---|
| UCI G3 trong predictor | `uci_risk_target`; `UCI_FORBIDDEN_PREDICTORS` | PASS | G3 chỉ tạo nhãn; cấm G1/G2/G3/absences khỏi context | — | Giữ |
| UCI G1/G2 lọt S0 | `build_uci_phase7_view` / `build_uci_stage_view` | PASS | S0 temporal=0, mask=0, aggregate_available=0 | — | Giữ |
| UCI stage expansion trước split | `src/hybrid/data/splits.py` + Phase7 views | PASS | Split trên record/group; mỗi record có 3 view sau split | — | Giữ |
| UCI MAT/POR cùng học sinh | `UCI_QUASI_IDENTITY_FIELDS`; `uci_summary.json` | PASS (Phase8) / POTENTIAL (thesis cũ) | Phase8: 662 nhóm, 366 cross-subject, `StratifiedGroupKFold` trên `global_student_group`. Thesis MAT/POR riêng: outer historical không group-safe (1–6 nhóm proxy overlap) | Medium nếu trích dẫn số cũ | Không dùng split thesis cũ cho VNext |
| UCI quasi-ID collision ≠ cùng người | không có student id thật | UNVERIFIABLE | Proxy có thể gộp người khác hoặc tách cùng người | Low | Giữ conservative grouping |
| OULAD `id_student` overlap fold | `splits.py` `verify_split_disjointness`; Phase1 `split_audit.json` | PASS | train∩test group = ∅ mọi outer/inner | — | Giữ |
| OULAD event sau cutoff | `build_oulad_phase7_view`: `date < cutoff_day` và `date >= observation_start` | PASS | VLE và submission lọc nghiêm; tuần trước registration là padding, không phải inactivity 0 | — | Giữ |
| OULAD `final_result` / `score` / `date_unregistration` làm feature | forbidden lists + Phase7 builder không copy các cột này vào array | PASS | Unregistration chỉ eligibility / observation_end | — | Giữ |
| OULAD score không có timestamp phát hành | Phase7 chủ động bỏ `score` | PASS | Không dùng điểm bài tập | Low information loss | Không mở lại nếu không có release time |
| FINAL-100 sequence-length shortcut | `final100.py` audit | **FAIL (shortcut, không phải future-event leak)** | Withdrawn mean length 9.16 vs Pass/Distinction/Fail ≈ 37; `withdrawn_rate_in_short_history=0.9989`; `flagged_as_shortcut_risk=true` | **HIGH** | Tách FINAL-100 khỏi early-warning claim; kiểm soát length as confounder |
| Preprocessor fit toàn bộ data | `_scale` / `TabularContextPreprocessor` | PASS | Fit trên FIT ids; `assert_train_only_fit` | — | Giữ |
| Threshold/calibration outer label | `stage_threshold_metrics(..., stop, valid)` | PASS | Threshold từ STOP; VALID chỉ report | — | Giữ |
| Nested CV group leak | `verify_inner_group_disjointness` | PASS | Code raise nếu overlap | — | Giữ |
| Reconstruction OOF leak | `reconstruct_phase8_hybrid.py` | PASS | `OOF_LEAKAGE` assert; outer_test_used=false | — | Giữ |
| Derived temporal leak (delta/rolling trên tương lai) | Phase7 temporal 11 kênh + D3 | PASS | Chỉ dùng tuần đã mask; D3 chia theo `week_exposure_fraction` | — | Giữ |
| Assessment due sau cutoff | Phase7 `opportunities.date < cutoff` | PASS | Opportunity chỉ bài có hạn trước cutoff | — | Giữ |
| Eligibility thay đổi cohort theo stage | `phase7_eligible_oulad` | PASS (hợp đồng) | Unreg trước cutoff bị loại khỏi early-warning; không cố định cohort | Medium diễn giải | Báo cáo theo operational risk-set, không “cùng sinh viên 4 stage” trừ common-cohort artifact |
| Current `src/prediction` thiếu feature generator | `oulad.py` chỉ wrap array | POTENTIAL | Active tree không tự tạo temporal từ raw; phụ thuộc kltn/legacy | Medium reproducibility | Design phase phải đưa generator vào active tree |

---

## D. Overfitting / underfitting audit

### Evidence cụ thể

| Nguồn | Quan sát | Phân loại |
|---|---|---|
| UCI inner selection (`uci_selection.json`) | generalization_gap S0 0.172 / S1 0.065 / S2 0.046 | **Overfit S0**; S2 nhẹ hơn |
| UCI reconstruction STOP PR-AUC | fold0 0.690 / fold1 0.508 / fold2 0.590 | **Selection + seed/fold variance** rất lớn |
| UCI capacity | 513 287 params vs ~435–444 FIT records | Overparameterized |
| Phase6 learning curves | UCI max gap 0.137; OULAD 0.036; class `OVERFIT_REMAINS` | UCI overfit lịch sử |
| Phase7D A2 (arch khác) | OULAD best_epoch mean 75/90, gap 0.024 | OULAD **có thể undertrain** nếu chỉ 16 epoch |
| Phase8/reconstruction recipe | max 16, patience 5; OULAD early best 6/13/9 (median 9); FINAL 15/7/11 (median 11); UCI 8/16/6 (median 8) | OULAD early **có thể** dừng sớm; UCI fold1 chạy hết 16 |
| Reconstruction final refit | `stop_ids=[]` → train đúng median epoch, giữ last state | Epoch budget được propagate; **không** restore best-of-refit |
| Historical unified OULAD (arch khác) | outer luôn 4 epoch, bỏ inner best | Đã sửa ở Phase8 reconstruction; **không** còn là bug hiện tại |
| Train metric tuyệt đối | Không còn full train/val loss curve của F3 | Underfit OULAD F3: **HYPOTHESIS** |

Kết luận: **UCI đang overfit / variance cao**. **OULAD F3 không có bằng chứng overfit mạnh**; undertraining so với Phase7D (75 epoch) là giả thuyết hợp lệ nhưng không được đo trên đúng D3/F3. Không được kết luận “chỉ cần thêm depth”.

Selection-overfit rủi ro: D3/F3/P1 được chọn trên inner development; outer đã mở một lần (freeze). VNext **không** được chọn lại architecture bằng outer cũ.

---

## E. Baseline fairness audit

### Hybrid nhận gì

| Hạng mục | Hybrid |
|---|---|
| Raw sequence | Có (11 kênh OULAD / 1 kênh UCI) |
| Aggregate | Có, projector riêng rồi **cộng** vào static |
| Static/context | Có |
| Engineered last/mean/max sequence | Không explicit; CNN/BiLSTM phải học |
| Preprocess | FIT-only scaler/encoder |
| Class weight | FIT-only `neg/pos` trên BCE |
| HPO | **Không** trên topology hiện tại; hyper cố định |
| Epoch | Inner early-stop PR-AUC; final = median inner epoch |
| Threshold | STOP-only, risk-F1 |
| Calibration | Không |
| Seeds (outer freeze) | 42 / 1201 / 2026 |
| Params | 513 287 UCI / 514 247 OULAD |

### Baseline nhận gì (Phase8 `baseline_configs.json` + `build_phase7_baseline_frame`)

| Model | Features | Weighting | HPO trong outer freeze | Threshold | Ghi chú |
|---|---|---|---|---|---|
| LR | static + 13 agg + last/mean/max mỗi kênh temporal + progress | `class_weight=balanced`, C=1 | Không (fixed) | Cùng protocol STOP | Linear trên representation đã engineer |
| DT | như trên | balanced, depth=8, min_leaf=20 | Không | Cùng | |
| RF | như trên | balanced, 200 trees, min_leaf=2 | Không | Cùng | Thắng UCI |
| XGB | như trên | 200 / depth 5 / lr 0.05 / subsample 0.8 | Không | Cùng | Thắng OULAD |
| MLP sklearn | như trên | alpha 1e-4, (128,64), 250 iter | Không | Cùng | ~17–21k params |
| SVM | catalog hiện tại có; **không có trong outer freeze table** | balanced, probability=True | — | — | So sánh outer hiện tại **thiếu SVM** |
| CatBoost | từng là trần Phase2/7 | — | không vào Phase8 outer | — | Trần lịch sử cao hơn XGB ở một số stage |

**Câu hỏi bắt buộc — inductive advantage: CÓ.**

Baseline không nhìn raw order, nhưng nhận bản tóm tắt mạnh của cùng sequence (last/mean/max + 13 aggregate: cumulative, trend, streak, completion, late rate, …). Đây **không** phải leakage. Đây là **tabular inductive advantage**.

Định lượng gián tiếp:

- Phase7 `architecture_contribution.csv`: đổi data/aggregate giúp OULAD +0.0047 macro PR-AUC; thêm unified CNN/BiLSTM **−0.0017**.
- Phase4B: OULAD hybrid dưới baseline 0.006–0.012 PR-AUC mọi stage.
- Historical ablation (arch H0 serial): aggregate/static-only chỉ kém full hybrid **0.0035** Macro-F1.
- Outer Phase8: XGB ≥ Hybrid mọi OULAD stage; RF thắng UCI đặc biệt S2 (0.911 vs Hybrid 0.874).

Budget: baseline outer dùng **fixed strong config**, không phải full Optuna. Hybrid cũng không full-tune. Không thể nói baseline bị “tối ưu kém cố ý”. Lịch sử Phase2 từng HPO baseline rồi chuyển fixed vì compute. **Không công bằng theo hướng Hybrid thiếu tune architecture hiện tại**, không theo hướng baseline bị cắt.

Registry `configs/prediction/registry.json` **loại XGB** khỏi active comparators. Nếu VNext chỉ so LR/DT/RF/SVM/MLP sẽ **làm đẹp giả**. Giữ XGB (và ideally CatBoost trần lịch sử) trong protocol mới.

---

## F. Architecture forensic audit

Nguồn: `src/prediction/model/hybrid.py`, `components.py`, và bản gốc `src/hybrid/phase8/model.py` (byte-level cùng topology). Probe: `artifacts/audit/hybrid_vnext_forensic_probe.json`.

### Computational graph thật

```text
static  --ResidualProjector--> h_s
aggregate --ResidualProjector--> h_a
tabular = h_s + h_a * aggregate_available          # không phải nhánh fusion riêng

temporal * mask --Linear+LN--> adapted
adapted --proj--> ResidualCNN (2 block, k=2, d=1 then 2, GELU, residual, symmetric pad)
      --masked mean+max--> Linear --> h_cnn
adapted --pack BiLSTM--> masked mean+max --> Linear --> h_lstm
        # h_cnn KHÔNG đi vào BiLSTM

available = [1, lengths>0, aggregate_available]     # ← mapping lệch
gate([tabular, h_cnn, h_lstm, available, progress]) --> 3 logits
logits.masked_fill(~available, -inf) --> softmax
fused = w0*tabular + w1*h_cnn + w2*h_lstm
head(LayerNorm(fused)) --> 1 logit
```

### 1. Serial hay parallel?

**Parallel.** Cùng `adapted` vào CNN và BiLSTM. Không có `CNN → BiLSTM`. Gọi “CNN-BiLSTM” theo nghĩa nối tiếp là **sai**.

### 2. CNN

| Mục | Giá trị |
|---|---|
| Input | `[B, T, d_fuse=96]` sau adapter |
| Kernel / dilation / blocks | 2 / (1, 2) / 2 |
| Padding | **không nhân quả**: `F.pad` trái/phải trong cửa sổ đã cutoff |
| Receptive field | **7 timestep** (2 conv × (k−1)×d cho mỗi block) |
| Residual | Có, trong block |
| Pooling | masked mean ∥ masked max → 256 → Linear 96 |
| Mask | nhân `keep` sau mỗi conv |

Với OULAD 20% (~8 tuần) RF phủ gần hết chuỗi. Với 75%/FINAL (~28–39 tuần) CNN chỉ local; long-range phải nhờ BiLSTM. Với UCI T≤2, RF=7 > T: CNN chỉ trộn toàn bộ (hoặc 1) bước — **không có hierarchy**.

### 3. BiLSTM

Packed sequence, bỏ length=0. Hidden 128 × 2 hướng × mean+max → 512 → 96. Một lớp. Không residual quanh LSTM.

### 4. Static / aggregate

Giữ projector riêng rồi **cộng sớm** thành một vector tabular. Gate **không** tách static vs aggregate.

### 5. Fusion

| Mode | Hành vi |
|---|---|
| `equal` | logits 0 → softmax đều trên nhánh available |
| `global` | 3 scalar `nn.Parameter` |
| `adaptive` | MLP gate, không entropy floor |
| `adaptive_entropy` (F3, hiện tại) | adaptive + `0.002 * relu(0.35 log k − H)` |

### 6. Availability mapping — **CONFIRMED BUG**

Probe (untrained, forward 4 hàng):

| Hàng | temporal | aggregate | CNN weight | BiLSTM weight |
|---|---|---|---|---|
| 0 | 1 | 1 | 0.35 | 0.30 |
| 1 | 1 | 0 | 0.49 | **0.00** |
| 2 | 0 | 1 | **0.00** | **0.49** |
| 3 | 1 | 0 | 0.57 | **0.00** |

Hệ quả:

- UCI S0: CNN tắt (T=0) **và** BiLSTM tắt (aggregate_available=0) — đúng tình cờ.
- UCI S1/S2: cả hai bật vì aggregate_available=1.
- OULAD early/final: `aggregate_available ≡ 1` nên BiLSTM **không bị tắt**, nhưng tín hiệu available[2] mang nghĩa “có aggregate” trong khi nhân với **h_lstm**. Gate học nhầm semantics.
- `branch_mode="bilstm"` giữ available[:,2] = aggregate_available: ablation BiLSTM-only **phụ thuộc aggregate flag**, không phải temporal flag.

**Không sửa ở phase này.** Bắt buộc test ở design: `available = [1, temporal, temporal]` hoặc 4 nhánh `[static, aggregate, cnn, lstm]`.

### 7. Phân bổ tham số (cùng topology)

| Khối | UCI | OULAD |
|---|---:|---:|
| static projector | 20 832 | 19 296 |
| aggregate projector | 10 848 | 12 384 |
| temporal adapter | 384 | 1 344 |
| CNN | 168 672 | 168 672 |
| BiLSTM | 280 672 | 280 672 |
| gate | 18 950 | 18 950 |
| head + fusion LN | 12 929 | 12 929 |
| **Tổng** | **513 287** | **514 247** |

CNN+BiLSTM ≈ 87% params. Tabular ≈ 6%. Capacity đang đảo so với nơi tín hiệu mạnh.

### 8. Representation / gate collapse

Phase7 `branch_diagnostics.csv` (A2/A3 — họ gần, **không phải F3**):

- `‖h_static‖` ≈ `‖h_aggregate‖` ≈ 9.8
- `‖h_cnn‖` OULAD ≈ 4.0–5.9; `‖h_lstm‖` ≈ 2.5–4.2
- UCI S0: CNN/LSTM norm ≈ 0.34 / 0.25 (chết)
- cosine(CNN, BiLSTM) 0.45–0.73 → **redundant**

Phase8 F3 `gate_diagnostics` trên VALID: **UNVERIFIABLE** (thư mục fusion/temporal chỉ còn log load VLE, không còn CSV). Không tuyên bố “tabular weight ≈ 0.9” như fact.

Entropy floor 0.002 rất nhỏ; khó chống collapse.

---

## G. OULAD root-cause ranking

Outer diagnosis (không dùng để chọn model):

| Stage | Hybrid PR-AUC | Best baseline | Δ |
|---|---:|---|---:|
| 20% | 0.7545 | XGB 0.7656 | −0.0111 |
| 35% | 0.7978 | XGB 0.8054 | −0.0075 |
| 50% | 0.8430 | XGB 0.8482 | −0.0052 |
| 75% | 0.8861 | XGB 0.8932 | −0.0071 |
| macro early | 0.8204 | XGB 0.8281 | −0.0077 |
| FINAL-100 | 0.9805 | XGB 0.9830 | −0.0024 |

Bootstrap Hybrid−XGB early: mean −0.0077, 95% CI [−0.0091, −0.0064], `p(Δ>0)=0`.  
ECE Hybrid early 0.014–0.019 — **không** còn là thảm họa như unified cũ (0.05–0.13). XGB vẫn tốt hơn (0.004–0.007).

| # | Root cause | Evidence | Confidence | Expected impact | How to test |
|---|---|---|---|---|---|
| 1 | Tabular inductive advantage: aggregate + last/mean/max đã nén gần hết tín hiệu trees cần | Phase7 contribution; H0 agg-only −0.0035; XGB thắng mọi stage; Grinsztajn 2022 | **HIGH** | Hybrid khó thắng nếu fusion không residual-hóa tree features | Cùng feature set: Hybrid tabular-only vs XGB; Hybrid full vs tabular-only |
| 2 | Parallel CNN∥LSTM + cộng sớm tabular; mapping available lệch | Code + probe; cosine CNN–LSTM cao | **HIGH** (mapping), **MED** (impact metric) | Temporal không complementary | Fix mask; serial vs parallel; 4-way gate |
| 3 | 87% params ở temporal, ~6% ở tabular | Param breakdown | **HIGH** | Gradient/capacity lệch | Gradient group norms; widen tabular / shrink CNN |
| 4 | Temporal order ít được exploit | H0 shuffle/reverse chỉ −0.0036..−0.0066; Phase8 shuffle **UNVERIFIABLE** | **MED** | Nếu lặp lại trên F3: CNN/LSTM gần như bag-of-weeks | Shuffle / reverse / time-gap mask trên inner |
| 5 | FINAL-100 length shortcut | Audit 0.9989 withdrawn in short hist | **HIGH** cho FINAL, **LOW** cho 20–75 (unreg bị loại) | Mọi model đều ~0.98; không chứng minh temporal DL | Condition on length; early-warning only |
| 6 | HPO không cover F3 topology; Phase7D winner (lr 6e-5, 90 ep) khác recipe 8e-4 / 16 ep | `model_selection.json`; Phase7D report | **HIGH** | Có thể +0.002–0.005, **không** đảo inductive gap | Inner-only HPO *sau* khi khóa topology |
| 7 | Một checkpoint / nhiều stage (P1) tradeoff | P1 selected; Phase6H S2 grad ≫ S0 (arch cũ) | **MED** | Early kém nếu late dominate | So P1 vs P2_all_stage trên inner |
| 8 | CNN RF=7 vs chuỗi 30+ tuần | Tính RF; không có multi-kernel 2/3/5 ở F3 | **MED** | Thiếu multi-scale | Multi-kernel / dilation lớn hơn — **chỉ sau** khi (1)–(4) rõ |
| 9 | Loss = weighted BCE; Focal/CB không được trainer F3 consume | `execution.py` / reconstruct | **MED-LOW** | Thresholded F1 có thể lệch | Inner Focal vs BCE; không SMOTE raw tensor |
| 10 | Calibration không phải nguyên nhân chính OULAD F3 | ECE Hybrid 0.018 vs cũ 0.13 | **HIGH** (cleared as primary) | Ít impact PR-AUC | Giữ ECE/Brier; không kết luận yếu từ F1@0.5 |

---

## H. UCI root-cause ranking

Outer diagnosis (macro 3 stage): Hybrid PR-AUC 0.7056 vs RF 0.7232 (Δ −0.0176). Bootstrap vs RF: −0.018, CI [−0.040, +0.005], `p(Δ>0)=0.055`.

| Stage | Hybrid | Best | Ghi chú |
|---|---:|---|---|
| S0 | 0.468 | XGB 0.490 | Không temporal |
| S1 | 0.774 | RF 0.783 | 1 timestep |
| S2 | 0.874 | RF 0.911 | 2 timestep; **lỗ lớn nhất** |

| # | Root cause | Evidence | Confidence | Expected impact | How to test |
|---|---|---|---|---|---|
| 1 | T∈{0,1,2}: CNN/BiLSTM không có inductive advantage | View builder; RF=7>T; S0 nhánh temporal chết | **HIGH** | Không bao giờ “chứng minh temporal DL” bằng UCI | Unified model phải *tắt/bỏ qua* temporal khi T<2; không tạo Hybrid-UCI riêng |
| 2 | Overparam + overfit S0 | 513k / 440 rows; gap 0.17; fold STOP 0.51–0.69 | **HIGH** | Variance nuốt delta | Regularize / nhỏ head; report seed-all |
| 3 | Grade signal đã nằm ở aggregate (latest, mean, Δ) — trùng temporal | `UCI_PHASE7_AGGREGATE_CHANNELS` | **HIGH** | CNN/LSTM tái học G1/G2 | Ablation bỏ temporal UCI |
| 4 | Mapping: S0 tắt BiLSTM qua aggregate flag | Probe + S0 available=0 | **HIGH** (cơ chế), impact S0 = 0 vì T=0 anyway | — | Vẫn phải sửa cho thống nhất |
| 5 | Absences bị cấm | Policy: không timestamp | PASS policy | Trees cũng không có | Không mở lại |
| 6 | Threshold S0=0.12 (inner OOF reconstruction) | `uci/reconstruction_manifest.json` | **MED** | F1 nhạy; PR-AUC mới là ranking | Báo cáo cả PR-AUC và F1 |
| 7 | Combined MAT+POR, prevalence 0.22 (MAT 0.33 / POR 0.15) | `uci_summary.json` | **MED** | Một head binary gộp 2 môn | Subject như context (đã có); không tách model |

UCI **không phù hợp** để chứng minh sức mạnh temporal của CNN–BiLSTM. Vẫn giữ trong protocol unified vì khóa luận có hai dataset — architecture phải *thích nghi* (gate/mask), không fork.

---

## I. Unified architecture constraints

VNext **bắt buộc** giữ:

1. Một class, một topology công khai: `static + aggregate + CNN + BiLSTM + fusion + binary head`.
2. Không `Hybrid-UCI` / `Hybrid-OULAD` / `Hybrid-S0` / `Hybrid-20%`.
3. Không thay CNN bằng transformer chỉ trên OULAD rồi giữ CNN trên UCI.
4. Không đổi fusion family theo dataset (hyper số học được, topology không).
5. Input contract: `static, temporal, mask, lengths, aggregate, aggregate_available, progress`.
6. Stage/endpoint = availability view, không phải identity model.
7. Split group-safe; preprocess FIT-only; threshold/HPO inner-only.
8. Fitted weights / dims / checkpoint / threshold được khác nhau.
9. GPU CUDA bắt buộc khi train ở phase sau; fit 6 GB.

Được phép (không đổi topology): số channel/hidden, dilation trong CNN, dropout, LR, epoch, pos_weight, entropy coefficient, *cách tính* available (sửa bug), số chiều d_fuse.

Cấm diễn giải: “vì config X thắng XGB trên outer cũ nên chọn X”.

---

## J. Candidate research hypotheses

Chưa phải thiết kế cuối.

### H1 — Fusion / availability bottleneck

- **Why:** available[2] = aggregate; tabular cộng sớm; 87% params temporal.
- **Evidence:** probe; code; param table.
- **Minimal experiment:** inner 1-fold: (a) mask đúng `[1, temp, temp]`; (b) 4-way gate; (c) residual `tabular + gate*temporal` kiểu TFT GRN.
- **Expected:** Δ PR-AUC OULAD +0.002–0.006 nếu temporal từng bị dìm; UCI S2 có thể + nhỏ.
- **Reject:** |Δ| < 0.002 trên 3 fold × 3 seed inner, cùng recipe.

### H2 — Undertraining / LR mismatch (OULAD)

- **Why:** F3 dùng 16 ep / 8e-4; Phase7D A2 tốt hơn ở ~75 ep / 6e-5.
- **Evidence:** epoch logs reconstruction; Phase7D.
- **Minimal:** inner learning curve 40 ep, 2 LR {8e-4, 6e-5}, patience 8.
- **Expected:** STOP PR-AUC còn tăng sau epoch 16 ở 6e-5.
- **Reject:** plateau ≤ epoch 12 mọi fold.

### H3 — Temporal order không được exploit

- **Why:** trees thắng bằng summary; H0 shuffle gần như không đau.
- **Evidence:** historical −0.0036..−0.0066; Phase8 **UNVERIFIABLE**.
- **Minimal:** train F3, evaluate shuffle/reverse/identity trên STOP (hoặc train trên shuffled).
- **Expected:** nếu Δ < 0.003 → model không dùng order.
- **Reject:** Δ ≥ 0.01 PR-AUC khi phá order.

### H4 — Loss / imbalance không phải nút thắt chính

- **Why:** OULAD F3 ECE ổn; historical class-weight làm giảm Macro-F1 authority cũ (−0.008).
- **Evidence:** ECE table; `FINAL_IMBALANCE_EVIDENCE.md` (họ model khác).
- **Minimal:** BCE vs weighted BCE vs Focal, inner only.
- **Expected:** Focal không thắng PR-AUC.
- **Reject:** Focal +≥0.005 PR-AUC ổn định 3 fold.

### H5 — CNN depth/wider không giúp

- **Why:** H0 capacity/dilation gain ≤ 0.002; UCI T≤2; RF đã 7.
- **Evidence:** Phase1 historical ablation; RF tính toán.
- **Minimal:** cnn_channels 64 vs 128 vs 192, inner.
- **Expected:** flat.
- **Reject:** +≥0.005 ổn định.

### H6 — FINAL-100 metric bị length confounder

- **Why:** withdrawn ≈ short sequence.
- **Evidence:** flagged shortcut.
- **Minimal:** PR-AUC stratified by length; model với length bị permute.
- **Expected:** gap Hybrid–XGB và absolute PR-AUC giảm mạnh trên Fail-vs-Pass (loại Withdrawn).
- **Reject:** length permute không đổi PR-AUC.

### H7 — Comparator feature parity

- **Why:** trees được last/mean/max + agg.
- **Minimal:** (i) XGB chỉ static; (ii) XGB static+agg; (iii) XGB full summaries; (iv) Hybrid tabular-only; (v) Hybrid full.
- **Expected:** phần lớn XGB đã đến từ (ii); (v)−(iv) nhỏ.
- **Reject:** (v)−(iv) ≥ 0.01 trên OULAD inner.

---

## K. Literature findings

Chỉ cơ chế dính root cause đã quan sát. Không copy architecture paper.

1. **Grinsztajn, Oyallon, Varoquaux (NeurIPS 2022), “Why do tree-based models still outperform deep learning on tabular data?”**  
   Trees bền với feature không thông tin, giữ orientation, học hàm không trơn. Đúng mô tả OULAD: 13 aggregate + hàng chục last/mean/max, N ~ 10⁴. Giải thích XGB/RF cạnh tranh mà không cần leakage.

2. **Lim et al. (IJF / arXiv:1912.09363), Temporal Fusion Transformer**  
   Gating để **bỏ qua** nhánh không dùng + residual, variable selection, LSTM local + attention dài. Trả lời H1: VNext cần *skip unused temporal trên UCI S0* và *không cộng chết tabular*, không cần full TFT (nặng hơn 6 GB nếu copy nguyên).

3. **Lin et al. (ICCV 2017) Focal Loss; Cui et al. (CVPR 2019) Class-Balanced Loss**  
   Hữu ích khi easy-negative át gradient. OULAD early prevalence ~0.34–0.44, không cực đoan; evidence nội bộ class-weight từng làm **giảm** F1. Đặt Focal ở hàng thí nghiệm rẻ, không phải trục chính.

4. **CNN–BiLSTM educational papers trên OULAD (nhiều báo 2024–2026)**  
   Thường report accuracy cao, protocol lỏng, ít so sánh group-safe nested CV với XGB cùng aggregate. Không lấy số của họ làm trần. Bài học: hybrid “thắng” dễ khi baseline yếu hoặc leak cutoff.

5. **Calibration (Guo et al. 2017; ECE/Brier)**  
   Unified cũ lệch calibration nặng. F3 hiện tại ECE OULAD chấp nhận được. Không được kết luận architecture yếu chỉ từ F1 thresholded — PR-AUC mới là ranking (và Hybrid vẫn thua XGB trên PR-AUC).

6. **Nested CV / HPO overfit (Cawley & Talbot 2010; nhiều tài liệu Optuna)**  
   Optuna cũ tune **họ khác**. Current F3 chưa được tune. VNext: HPO chỉ trên inner; không reuse outer 2026-08-17 để chọn.

---

## L. Recommended experimental order

Chưa chạy. Rẻ / nhiều thông tin trước.

1. **Unit + 1-fold mapping test** — sửa *trong prototype*, không production: `available` đúng; assert BiLSTM tắt khi T=0, bật khi T>0 bất kể aggregate.  
2. **H7 feature-parity** (inner fold 0, 1 seed) — định lượng inductive advantage.  
3. **H3 shuffle/reverse** trên checkpoint F3 hiện tại (eval-only nếu được; else train 1 fold).  
4. **H1 fusion variants** (mask fix / residual tabular / 4-way), inner 3×3 nếu (1) pass.  
5. **H2 learning curve** chỉ trên fusion thắng (1).  
6. **H4 loss** chỉ nếu PR-AUC ổn nhưng F1 kém.  
7. **H5 depth/width** chỉ nếu (3) chứng minh order được dùng và RF thiếu.  
8. **H6 FINAL-100** song song, không trộn vào early-warning selection.  
9. **Không** outer test / không chọn seed đẹp / không giảm tune XGB.

---

## M. 14-hour compute plan draft

Máy: GTX 2060 6 GB. Mọi đề xuất dưới đây fit VRAM.

| Đại lượng | Ước lượng |
|---|---|
| Params | ~0.51 M → 2 MB fp32 / 1 MB fp16 |
| Activation (B=256, T=40, C=128 CNN + 256 LSTM) | ≪ 1 GB |
| Batch an toàn | 128–512; mặc định 256 |
| AMP | Tương thích; Phase8 execution hiện **không** AMP (V1 trainer cũ có) |
| Bottleneck | I/O VLE nếu rebuild view; còn lại backward BiLSTM |
| 1 inner fold OULAD 16 ep | vài phút GPU (reconstruction từng chạy cả CPU) |
| 3 fold × 3 seed × 16 ep | ~1–2 giờ / cấu hình |

Phân bổ 14 giờ (sau design):

| Slot | Giờ | Việc |
|---|---:|---|
| 0 | 0.5 | Unit mapping + smoke CUDA |
| 1 | 2 | H7 + H3 (rẻ) |
| 2 | 5 | H1 ≤ 4 fusion candidates × 3 fold × 3 seed (hoặc 3×1 rồi robust winner) |
| 3 | 3 | H2 curve trên 1–2 winner |
| 4 | 2 | H4/H6 nếu còn |
| 5 | 1.5 | Dự phòng OOM / rerun |
| **Cấm** | — | Full Optuna architecture + outer 3 dataset |

HPO strategy phù hợp 14h: **factorial nhỏ đã preregister**, không TPE 100 trial trên CNN/LSTM size. Mixed precision bật khi train. Không đề xuất Transformer đầy đủ / T>64 channel CNN rộng trên 6 GB.

---

## N. GO / NO-GO

```text
READY_FOR_DESIGN
```

Đủ bằng chứng để biết đang sửa **cái gì** và **thí nghiệm nào bác bỏ**. Không đủ để tuyên bố architecture tối ưu.

Còn thiếu (không chặn design, phải đo ở thí nghiệm rẻ):

- Gate weight / entropy **trên F3 đã fit** (OOF reconstruction có checkpoint — design phase được phép chạy eval-only).  
- Shuffle/reverse **đúng F3**.  
- Learning curve đầy đủ train vs STOP của F3.

Không thiếu: topology thật, leakage chính, FINAL-100 shortcut, inductive advantage, HPO lineage, epoch policy reconstruction, outer ranking (chỉ để chẩn đoán).

---

## Trả lời 20 câu bắt buộc

1. **Serial hay parallel?** Parallel. CNN và BiLSTM cùng `adapted`.  
2. **Aggregate/static fusion?** Projector riêng → `tabular = h_s + h_a * available`; một nhánh gate.  
3. **Availability mapping đáng nghi?** **Có, confirmed.** `available[2] = aggregate_available` nhưng nhân `h_lstm`.  
4. **OULAD XGBoost mạnh vì đâu?** Cùng tín hiệu đã được engineer thành bảng (agg + last/mean/max); bias cây trên tabular cỡ 10k mẫu; không cần học order.  
5. **Aggregate tóm tắt bao nhiêu temporal?** Gần hết phần trees dùng: historical agg-only −0.0035 vs full serial; Phase7 thêm CNN/LSTM **âm** 0.0017.  
6. **Optuna đúng architecture hiện tại?** **Không.** D3/F3/P1 chọn bằng screen; Optuna 7D/4A/6* là họ khác.  
7. **Final architecture train bao nhiêu epoch?** Reconstruction: UCI 8, OULAD early 9, FINAL 11 (median inner). Screening Phase8 max 16 patience 5. Outer freeze gốc: không có checkpoint byte-identical.  
8. **Epoch inner → outer/refit?** Reconstruction **có**: median inner best, refit đúng số đó, last state. Unified OULAD cũ (4 epoch cố định) **không** còn là hành vi F3.  
9. **Calibration có làm mất F1?** OULAD F3 ECE thấp; không phải nguyên nhân chính. UCI ECE Hybrid 0.08–0.21 > RF 0.03–0.06. Ranking (PR-AUC) vẫn thua — không phải chỉ threshold.  
10. **UCI S0/S1 đủ temporal?** **Không.** S0 = 0; S1 = 1 bước.  
11. **Leakage MAT/POR identity?** Phase8 group-safe trên quasi-id. Thesis split cũ: potential proxy overlap.  
12. **OULAD split group-safe?** **Có.**  
13. **Feature generator cutoff-safe?** Phase7/FINAL-100 builder: **có** (event < boundary). Active `src/prediction/data/oulad.py` không tự sinh array.  
14. **Tuning budget công bằng?** Cả hai phía outer freeze đều fixed-config. Hybrid **chưa** tune topology hiện tại; baseline không bị cắt cố ý. XGB phải giữ.  
15. **Underfit hay overfit?** UCI overfit/variance. OULAD F3: không overfit rõ; undertrain là hypothesis.  
16. **Branch collapse?** Mapping collapse BiLSTM khi hết aggregate: confirmed. Gate mass collapse tabular trên F3: **UNVERIFIABLE**. Norm Phase7: tabular ≫ temporal.  
17. **Temporal order đóng góp metric?** Lịch sử nhỏ; F3 **UNVERIFIABLE**.  
18. **Điểm yếu sửa đầu?** Availability + fusion/residual tabular vs temporal complementarity — không phải depth.  
19. **Chỉ tăng CNN/LSTM depth có giúp?** Evidence lịch sử: **không**. Reject-first bằng H5.  
20. **Unified architecture cần tính chất gì?** Một topology; mask/gate **tắt temporal khi T≈0**; không fork theo dataset; tabular residual không bị CNN/LSTM dìm; RF/multi-scale chỉ khi chuỗi dài; HPO/threshold inner-only; fit 6 GB.

---

## Provenance (không dùng README làm bằng chứng duy nhất)

| Hạng | Path |
|---|---|
| Active model | `src/prediction/model/hybrid.py`, `components.py` |
| Phase8 authority | `origin/codex/backup-hybrid-phase8-2026-08-17:src/hybrid/phase8/*` |
| Feature gen | `C:\hufit\kltn\src\hybrid\phase7\data.py`, `phase8\final100.py`, `data\oulad.py` |
| Trainer F3 | `phase8/execution.py`; reconstruct `scripts/reconstruction/reconstruct_phase8_hybrid.py` |
| Splits | `src/hybrid/data/splits.py`; hashes trong `model_selection.json` |
| Outer freeze | `artifacts/prediction/final/outer_test_final/*`, `reports/prediction/final/*` |
| Reconstruction | `artifacts/prediction/reconstructed/**` |
| Probe | `artifacts/audit/hybrid_vnext_forensic_probe.json` |
| Historical (họ khác) | `reports/audit/PHASE1_*`, `reports/hybrid/PHASE6_*`, `PHASE7*` |

**Không train. Không sửa production. Không chọn architecture bằng outer.**
