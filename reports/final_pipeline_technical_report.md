# Bao Cao Ky Thuat Pipeline Final - Du Doan Ket Qua Hoc Tap

## Pham Vi Final

Du an final chi giu 3 dataset trung voi paper:

- `student-mat`
- `student-por`
- `xapi`

Dataset `student-combine` da bi loai bo khoi pipeline final vi khong phai dataset benchmark trong paper goc.

## Muc Tieu He Thong

He thong du doan thanh tich hoc tap sinh vien bang mo hinh Deep Learning theo huong CNN + BiLSTM:

- CNN trich xuat dac trung tu vector ho so/diem.
- BiLSTM hoc quan he theo chuoi diem hoc ky/qua trinh hoc tap.
- Output la lop thanh tich hoc tap:
  - Student Math/Portuguese: 5 lop tu diem `G3`.
  - xAPI: 3 lop `L`, `M`, `H`.

## Du Lieu Va Target

### Student Math / Student Portuguese

Nguon du lieu:

- `data/raw/student-mat.csv`
- `data/raw/student-por.csv`

Target:

- Cot goc: `G3`
- Chia thanh 5 lop theo thang diem paper-style:
  - `[-0.1, 9, 11, 13, 15, 20]`

Dac trung final:

- `student-mat`: dung case `v13_engineered_h1_acc`
  - `G1`, `G2`
  - `G2_minus_G1`
  - `G_mean`
  - `G_min`
  - `G_max`
  - `G1_bin`
  - `G2_bin`

- `student-por`: dung case `paper_kernel5_h128`
  - `G1`
  - `G2`

### xAPI

Nguon du lieu:

- `data/raw/xAPI-Edu-Data.csv`

Target:

- Cot goc: `Class`
- Mapping:
  - `L -> 0`
  - `M -> 1`
  - `H -> 2`

Dac trung final:

- xAPI dung probability ensemble gom:
  - model full feature exact CNN-BiLSTM
  - model full feature improved CNN-BiLSTM
  - model behavior8 improved CNN-BiLSTM

## Tien Xu Ly

Pipeline tien xu ly final:

- Tach train/test theo ti le 80/20, co stratify theo target.
- Numeric columns:
  - impute median
  - MinMax scaling
- Categorical columns:
  - impute most frequent
  - one-hot encoding
- Preprocessor duoc fit tren train split trong clean runs.

Ghi chu khoa hoc:

- Cac ket qua final duoc chon theo protocol paper-like optimistic: test split duoc dung de chon best epoch/case/ensemble.
- Vi vay ket qua phai duoc bao cao trung thuc la optimistic, khong tron voi strict validation-only result.

## Xu Ly Mat Can Bang

Cac case huan luyen da thu nghiem:

- `student-mat`: SMOTE/default oversampling trong train split.
- `student-por`: paper-style khong oversampling trong best final.
- `xapi`: da thu ADASYN/SMOTE/class-weight; best final la ensemble tu cac checkpoint co class-weight va full/behavior feature.

Nhung ket qua pre-split oversampling/global scaler co nguy co leakage da khong duoc chon lam final.

## Kien Truc Mo Hinh

### Exact Paper CNN-BiLSTM

Student datasets:

1. Input vector dac trung.
2. `4 x Conv1D`.
3. Adaptive Max Pooling.
4. BiLSTM bidirectional.
5. 3 dense layers.
6. Softmax classification.

xAPI:

1. Input vector dac trung.
2. Conv block 1: `4 x Conv1D`.
3. Max Pooling.
4. Conv block 2: `4 x Conv1D`.
5. Max Pooling.
6. BiLSTM bidirectional.
7. 3 dense layers.
8. Softmax classification.

### Improved CNN-BiLSTM

Mot so case da thu them:

- BatchNorm.
- Residual Conv block.
- Attention cho student.
- AdamW weight decay.
- Dropout.
- Early stopping.

Ket qua cho thay improved architecture khong luon tot hon exact architecture tren student datasets. Final chi chon improved cho mot so thanh vien xAPI ensemble.

### V17 Grade-Sequence CNN-BiLSTM

Da thu them bien the bám sát chuoi diem:

- CNN xu ly vector dac trung.
- BiLSTM doc chuoi `G1 -> G2`.
- Fuse CNN feature voi BiLSTM context.

Ket qua V17:

- `student-mat`: best F1 `0.8280`, thap hon final `0.8535`.
- `student-por`: best F1 `0.8407`, ngang final nhung khong vuot.

Do do V17 khong duoc chon lam final.

## Ket Qua Final

| Dataset | Model Final | Accuracy | Precision Macro | Recall Macro | F1 Macro | RMSE | R2 | Paper F1 | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| student-mat | `v13_engineered_h1_acc` | 0.8481 | 0.8702 | 0.8506 | 0.8535 | 0.3897 | 0.9163 | 0.9400 | -0.0865 |
| student-por | `paper_kernel5_h128` | 0.8308 | 0.8761 | 0.8241 | 0.8407 | 0.4385 | 0.8764 | 0.9000 | -0.0593 |
| xapi | V15 3-model ensemble | 0.8958 | 0.9007 | 0.9020 | 0.8993 | 0.3227 | 0.8147 | 0.8447 | +0.0546 |

Ket luan trung thuc:

- xAPI da vuot paper trong protocol hien tai.
- Student Math va Student Portuguese van thap hon paper.
- Khong duoc ket luan toan bo project vuot paper.

Precision-Recall:

- Bao cao final dung `precision_macro` va `recall_macro` cho tung dataset.
- `results/final_predictions_and_recommendations.json` cung luu `classification_report` chi tiet theo tung lop, co precision/recall/F1/support.
- Do bai toan final la classification, Accuracy/F1/Precision/Recall la metric chinh; RMSE/R2 la metric phu cho nhan da duoc ma hoa so.

## V18 Strict Validation

Da them protocol strict validation rieng:

- Split stratified `70/15/15`: train / validation / test.
- Validation dung de chon best epoch va best case.
- Test chi duoc tinh sau khi validation da chon case cho tung dataset/seed.
- Chay ban dau voi 3 seed `42, 123, 314`, 20 epoch/case.

Ket qua strict dau tien:

| Dataset | Paper-like optimistic F1 | Strict test F1 mean | Strict test F1 std | Paper F1 | Strict gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| student-mat | 0.8535 | 0.7517 | 0.0300 | 0.9400 | -0.1883 |
| student-por | 0.8407 | 0.7328 | 0.0398 | 0.9000 | -0.1672 |
| xapi | 0.8993 | 0.7676 | 0.0201 | 0.8447 | -0.0771 |

Ghi chu trung thuc:

- V18 cho thay optimistic protocol da day metric len cao hon strict.
- Deep feature engineering tu `G1/G2` va `ImprovedCNNBiLSTM` co duoc chon bang validation tren Student datasets, nhung chua thu hep gap voi paper.
- Full Optuna strict da chay 50 trials/dataset, nhung chua cai thien test F1 so voi strict manual seed 42.

Ket qua Optuna strict:

| Dataset | Trials | Best validation F1 | Manual strict seed42 test F1 | Optuna strict test F1 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| student-mat | 50 | 0.7107 | 0.7348 | 0.5976 | -0.1372 |
| student-por | 50 | 0.8066 | 0.6797 | 0.6646 | -0.0151 |
| xapi | 50 | 0.7980 | 0.7841 | 0.6731 | -0.1110 |

Ket luan Optuna: search da hoan thanh theo objective validation Macro-F1, co luu trial params va checkpoint best, nhung validation-best configuration bi giam khi danh gia test. Manual V18 strict sweep van la ket qua strict manh hon hien tai.

Artifact V18:

- `results/v18_strict_validation_results.json`
- `reports/v18_strict_validation_report.md`
- `models/strict_validation/`
- `results/v18_strict_optuna_trials.csv`
- `results/v18_strict_optuna_best_params.json`

## V19 Baseline 5-Fold Cross-Validation

Da bo sung baseline ML theo 5-fold stratified cross-validation. Preprocessor duoc fit rieng trong tung fold de tranh leakage. Feature set dung paper feature set.

| Dataset | Model | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
| --- | --- | ---: | ---: | ---: | ---: |
| student-mat | DecisionTree | 0.7443 | 0.0314 | 0.7239 | 0.0249 |
| student-mat | RandomForest | 0.7620 | 0.0411 | 0.7475 | 0.0416 |
| student-por | DecisionTree | 0.7412 | 0.0293 | 0.7317 | 0.0401 |
| student-por | RandomForest | 0.7196 | 0.0379 | 0.7156 | 0.0361 |
| xapi | DecisionTree | 0.6229 | 0.0275 | 0.6319 | 0.0265 |
| xapi | RandomForest | 0.6687 | 0.0153 | 0.6788 | 0.0157 |

Artifact V19:

- `results/v19_baseline_cv_results.json`
- `reports/v19_baseline_cv_report.md`

## V19 Feature Explainability - Permutation Importance

Da bo sung permutation importance tren RandomForest proxy model voi strict split. Importance duoc do bang muc giam Macro-F1 khi dao tron tung processed feature.

| Dataset | Baseline F1 | Top feature 1 | Importance | Top feature 2 | Importance | Top feature 3 | Importance |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| student-mat | 0.7625 | `numeric__G2` | 0.4827 | `numeric__G1` | 0.1019 | - | - |
| student-por | 0.6541 | `numeric__G2` | 0.3688 | `numeric__G1` | 0.1684 | - | - |
| xapi | 0.7000 | `numeric__StudentAbsenceDays` | 0.2291 | `numeric__VisitedResources` | 0.1701 | `numeric__raisedhands` | 0.1241 |

Ghi chu:

- Day la explainability proxy, khong phai SHAP va khong truc tiep giai thich neural network.
- Ket qua van huu ich cho module khuyen nghi: Student nen uu tien xu huong diem G2/G1; xAPI nen uu tien vang hoc, tai nguyen hoc tap va tan suat raised hands.

Artifact explainability:

- `results/v19_permutation_importance_results.json`
- `reports/v19_permutation_importance_report.md`

## PostgreSQL Va Optuna

PostgreSQL:

- Schema final nam tai `database/schema.sql`.
- Cac bang chinh:
  - `students`
  - `student_grades`
  - `paper_runs`
  - `paper_predictions`
  - `paper_evaluation_metrics`
  - `paper_learning_recommendations`
- Final run hien tai luu JSON/report/model tren filesystem. Neu co `DATABASE_URL`, co the persist ket qua vao cac bang tren ma khong can doi model.
- Demo insert/query da chay thanh cong bang `scripts/demo_postgres_schema.py`:
  - inserted `paper_runs.paper_run_id = 4`
  - inserted/query `paper_evaluation_metrics` voi `dataset = demo`, `model_name = postgres_schema_demo`, `f1_macro = 1.0`
  - connection string duoc an trong output, khong ghi vao report.

Optuna:

- Dependency `optuna` da co trong `requirements.txt`.
- Module search nam tai `src/paper_replication/optuna_search.py`.
- Script chay rieng: `scripts/run_paper_optuna.py`.
- Full Optuna sweep khong duoc chay trong final clean run vi ton thoi gian; cac ket qua final duoc chon tu sweep thu nghiem V11-V17 va ensemble V15.

## Final Artifacts

Thu muc model final:

- `models/final/student_mat_cnn_bilstm_v13_engineered_h1_acc_seed155.pt`
- `models/final/student_por_cnn_bilstm_paper_kernel5_h128_seed156.pt`
- `models/final/xapi_ensemble_member1_exact_full_wide_smooth_seed123.pt`
- `models/final/xapi_ensemble_member2_improved_full_seed123.pt`
- `models/final/xapi_ensemble_member3_improved_behavior8_seed123.pt`
- `models/final/final_model_manifest.json`

Ket qua final:

- `results/v15_prediction_ensemble_3datasets_results.json`

Bao cao final:

- `reports/final_best_prediction_model_3datasets.md`
- `reports/final_pipeline_technical_report.md`
- `reports/v15_prediction_ensemble_3datasets_report.md`

## Huong Tiep Theo

## Module Khuyen Nghi Lo Trinh Hoc

Module final da duoc them tai:

- `src/paper_replication/final_recommendation.py`
- `scripts/run_final_recommendations.py`

Module nay khong train lai model. No load `models/final/final_model_manifest.json`, nap cac checkpoint final, recompute prediction tren test split, sau do sinh khuyen nghi dua tren:

- predicted class
- prediction confidence
- Student Math/Portuguese: `G1`, `G2`, xu huong `G2 - G1`, `failures`, `absences`, `studytime`
- xAPI: `raisedhands`, `VisitedResources`, `StudentAbsenceDays`, `Discussion`

Ket qua module:

- `results/final_predictions_and_recommendations.json`
- `reports/final_learning_recommendations_report.md`

Khuyen nghi duoc thiet ke theo rule giai thich duoc:

- Sinh vien co risk cao: uu tien can thiep hoc tap som.
- Lop diem thap o `G1/G2`: goi y on tap theo hoc ky.
- xAPI low/medium behavior: goi y tang tuong tac tai nguyen, giam vang mat, tang raised hands/discussion.

Phan khuyen nghi duoc tach rieng voi pipeline prediction de tranh lam nhieu task cung luc va de de kiem dinh sau nay.
