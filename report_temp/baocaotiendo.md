# BÃ¡o cÃ¡o tiáº¿n Ä‘á»™ Ä‘á»“ Ã¡n sau khi clean vÃ  cháº¡y láº¡i mÃ´ hÃ¬nh

NgÃ y cáº­p nháº­t: 2026-05-30

## 1. Má»¥c tiÃªu Ä‘á» tÃ i

Äá» tÃ i xÃ¢y dá»±ng há»‡ thá»‘ng dá»± Ä‘oÃ¡n thÃ nh tÃ­ch há»c táº­p sinh viÃªn báº±ng há»c sÃ¢u, cÃ³ xá»­ lÃ½ máº¥t cÃ¢n báº±ng dá»¯ liá»‡u báº±ng SMOTE/ADASYN. MÃ´ hÃ¬nh chÃ­nh dÃ¹ng CNN Ä‘á»ƒ trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng vÃ  BiLSTM Ä‘á»ƒ há»c biá»ƒu diá»…n tá»« tiáº¿n trÃ¬nh Ä‘iá»ƒm hoáº·c chuá»—i hÃ nh vi há»c táº­p. NgoÃ i dá»± Ä‘oÃ¡n phÃ¢n lá»›p, project cÃ³ baseline há»“i quy Ä‘á»ƒ bÃ¡o cÃ¡o RMSE vÃ  RÂ², vÃ  cÃ³ module khuyáº¿n nghá»‹ dá»±a trÃªn feature importance/rule.

CÃ¡c yÃªu cáº§u chÃ­nh Ä‘Ã£ Ä‘Æ°á»£c map vÃ o project:

- Tiá»n xá»­ lÃ½ dá»¯ liá»‡u: `src/data/prepare_datasets.py`, `src/data/prepare_xapi.py`, `src/data/preprocess.py`.
- Xá»­ lÃ½ máº¥t cÃ¢n báº±ng: `src/features/imbalance.py`.
- CNN-BiLSTM: `src/models/deep_learning.py`.
- Train deep final: `src/train/train_deep.py`, `src/train/train_deep_classification.py`, `src/train/train_xapi_deep.py`.
- Baseline phÃ¢n loáº¡i/há»“i quy: `src/train/train_baselines.py`, `src/train/train_xapi_baselines.py`.
- ÄÃ¡nh giÃ¡: `src/evaluation/metrics.py`, `src/evaluation/summarize_all.py`.
- Khuyáº¿n nghá»‹: `src/recommend/feature_importance.py`, `src/recommend/study_advice.py`.

## 2. Nhá»¯ng gÃ¬ Ä‘Ã£ clean

ÄÃ£ dá»n project Ä‘á»ƒ gá»n hÆ¡n vÃ  bÃ¡m sÃ¡t Ä‘á» cÆ°Æ¡ng:

- XÃ³a notebook demo cÅ© khÃ´ng cÃ²n dÃ¹ng.
- XÃ³a thÆ° má»¥c output legacy `results/`, `saved_models/`.
- XÃ³a checkpoint vÃ  output náº·ng trong `models/saved/`, `reports/results/`, `reports/figures/`; chá»‰ giá»¯ `.gitkeep`.
- XÃ³a source cÅ© bá»‹ lá»‡ch pipeline: `src/evaluate/`, `src/utils/`, cÃ¡c file `prepare_data.py`, `split_data.py`, `train_basic.py`, `basic_models.py`, `deep_models.py`.
- Giá»¯ láº¡i pipeline chÃ­nh trong `src/data`, `src/features`, `src/models`, `src/train`, `src/evaluation`, `src/recommend`.
- Cáº­p nháº­t `README.md`, `requirements.txt`, `environment.yml`, `.gitignore`.
- Sá»­a CLI Ä‘á»ƒ `--help` khÃ´ng cháº¡y train/preprocess tháº­t á»Ÿ `prepare_xapi.py` vÃ  `train_xapi_baselines.py`.

Sau cleanup, `src` cÃ²n 32 file source chÃ­nh.

## 3. Command Ä‘Ã£ cháº¡y láº¡i

```powershell
py -3 scripts\run_prepare.py
py -3 scripts\run_train_baselines.py
py -3 scripts\run_train_deep.py
py -3 scripts\run_evaluate.py
```

Kiá»ƒm tra ká»¹ thuáº­t:

```powershell
py -3 -m compileall -q src scripts
py -3 -m src.train.train_deep --help
py -3 -m src.train.train_xapi_baselines --help
py -3 -m src.data.prepare_xapi --help
```

Táº¥t cáº£ command trÃªn cháº¡y thÃ nh cÃ´ng.

## 4. Dá»¯ liá»‡u sau preprocessing

| Dataset | Scenario | Train | Val | Test | Ghi chÃº |
|---|---:|---:|---:|---:|---|
| student-mat | mid/late | 275 | 60 | 60 | CÃ³ 395 máº«u. Late dÃ¹ng G1/G2; ngoai_pham_vi_hien_tai khÃ´ng dÃ¹ng G1/G2 Ä‘á»ƒ trÃ¡nh leakage. |
| student-por | mid/late | 453 | 98 | 98 | CÃ³ 649 máº«u. |
| student-combined | mid/late | 730 | 157 | 157 | Gá»™p student-mat + student-por, thÃªm `dataset_id`. |
| xAPI | xapi_behavior | 336 | 72 | 72 | CÃ³ 480 máº«u, 16 raw features, 72 processed features. |

Leakage check Ä‘á»u pass:

- KhÃ´ng Ä‘Æ°a G3 vÃ o feature.
- ngoai_pham_vi_hien_tai khÃ´ng dÃ¹ng G1/G2.
- Mid dÃ¹ng G1, khÃ´ng dÃ¹ng G2.
- Late dÃ¹ng G1/G2.
- xAPI khÃ´ng Ä‘Æ°a target `Class` vÃ o feature.

## 5. MÃ´ hÃ¬nh deep final vÃ  xá»­ lÃ½ máº¥t cÃ¢n báº±ng

### Student datasets

Model: `CLSV2` / `StudentCNNBiLSTMV2`.

CÃ¡ch xá»­ lÃ½:

- CNN branch há»c tá»« toÃ n bá»™ vector Ä‘áº·c trÆ°ng sau preprocessing vÃ  feature selection.
- Grade branch dÃ¹ng biá»ƒu diá»…n G1/G2 dáº¡ng 2 timestep, má»—i timestep cÃ³ nhiá»u chiá»u: giÃ¡ trá»‹ Ä‘iá»ƒm, trend, Ä‘iá»ƒm chuáº©n hÃ³a, velocity.
- KhÃ´ng dÃ¹ng G1/G2 nhÆ° chuá»—i thá»i gian dÃ i giáº£ táº¡o.

### xAPI

Model: `HybridXAPI`.

CÃ¡ch xá»­ lÃ½:

- Behavior sequence gá»“m `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`.
- Shape sequence: `(batch, 4, 1)`.
- Static branch nháº­n cÃ¡c feature cÃ²n láº¡i.
- Sequence branch dÃ¹ng Conv1D + BiLSTM.
- Static branch dÃ¹ng MLP.
- Fusion báº±ng concat.

### SMOTE/ADASYN

ÄÃ£ cháº¡y Ä‘á»§:

- `none`
- `smote`
- `adasyn`

Oversampling chá»‰ Ã¡p dá»¥ng trÃªn train set. Test set giá»¯ nguyÃªn phÃ¢n phá»‘i gá»‘c.

## 6. Káº¿t quáº£ deep learning cuá»‘i

Nguá»“n: `reports/tables/final_smote_adasyn_deep_comparison.csv`.

| Dataset | Scenario | Model | Imbalance | Accuracy | Precision macro | Recall macro | F1 macro | PR-AUC macro | Rank |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| student-combined | late | CLSV2 | SMOTE | 0.8471 | 0.8402 | 0.8632 | 0.8474 | 0.9317 | 1 |
| student-combined | late | CLSV2 | none | 0.8344 | 0.8510 | 0.8244 | 0.8317 | 0.9382 | 2 |
| student-combined | late | CLSV2 | ADASYN | 0.8344 | 0.8366 | 0.8218 | 0.8284 | 0.9300 | 3 |
| student-mat | late | CLSV2 | none | 0.8167 | 0.8310 | 0.8256 | 0.8272 | 0.9114 | 1 |
| student-mat | late | CLSV2 | SMOTE | 0.8000 | 0.8545 | 0.7900 | 0.8083 | 0.9447 | 2 |
| student-mat | late | CLSV2 | ADASYN | 0.7500 | 0.7795 | 0.7533 | 0.7640 | 0.8791 | 3 |
| student-por | late | CLSV2 | ADASYN | 0.8776 | 0.8808 | 0.8512 | 0.8644 | 0.9462 | 1 |
| student-por | late | CLSV2 | SMOTE | 0.8571 | 0.8517 | 0.8334 | 0.8409 | 0.9431 | 2 |
| student-por | late | CLSV2 | none | 0.8469 | 0.8184 | 0.8647 | 0.8368 | 0.9585 | 3 |
| xAPI | behavior | HybridXAPI | ADASYN | 0.8194 | 0.8246 | 0.8285 | 0.8243 | 0.8724 | 1 |
| xAPI | behavior | HybridXAPI | SMOTE | 0.7917 | 0.7909 | 0.8203 | 0.7980 | 0.8571 | 2 |
| xAPI | behavior | HybridXAPI | none | 0.7500 | 0.7503 | 0.7710 | 0.7571 | 0.8444 | 3 |

Káº¿t luáº­n tá»« báº£ng deep:

- `student-combined`: SMOTE tá»‘t nháº¥t, F1 = 0.8474.
- `student-mat`: khÃ´ng oversampling tá»‘t nháº¥t, F1 = 0.8272. SMOTE/ADASYN lÃ m F1 giáº£m.
- `student-por`: ADASYN tá»‘t nháº¥t, F1 = 0.8644.
- `xAPI`: ADASYN tá»‘t nháº¥t, F1 = 0.8243.

## 7. So sÃ¡nh vá»›i baseline phÃ¢n loáº¡i

Nguá»“n: `reports/tables/baseline_classification_summary.csv`.

| Dataset | Scenario | Baseline tá»‘t nháº¥t theo validation | Test Accuracy | Test F1 macro |
|---|---|---|---:|---:|
| student-mat | mid | Decision Tree | 0.6000 | 0.6084 |
| student-mat | late | Gradient Boosting | 0.9333 | 0.9372 |
| student-por | mid | Gradient Boosting | 0.7551 | 0.7105 |
| student-por | late | Random Forest | 0.8776 | 0.8581 |
| student-combined | mid | Random Forest | 0.7389 | 0.6957 |
| student-combined | late | Gradient Boosting | 0.8471 | 0.8422 |

So vá»›i baseline late:

| Dataset | Deep final tá»‘t nháº¥t | Baseline tá»‘t nháº¥t | ChÃªnh lá»‡ch F1 |
|---|---:|---:|---:|
| student-combined late | 0.8474 | 0.8422 | +0.0052 |
| student-mat late | 0.8272 | 0.9372 | -0.1100 |
| student-por late | 0.8644 | 0.8581 | +0.0063 |

Nháº­n xÃ©t:

- Deep model vÆ°á»£t nháº¹ baseline á»Ÿ `student-combined late` vÃ  `student-por late`.
- `student-mat late` váº«n thua ráº¥t xa Gradient Boosting. NguyÃªn nhÃ¢n cÃ³ báº±ng chá»©ng: dataset nhá», chá»‰ 395 máº«u, vÃ  khi cÃ³ G1/G2 thÃ¬ bÃ i toÃ¡n gáº§n tuyáº¿n tÃ­nh/tree split ráº¥t máº¡nh. Baseline tree khai thÃ¡c quan há»‡ G1/G2 -> G3 tá»‘t hÆ¡n deep model.

## 8. xAPI: deep model vÃ  baseline

Káº¿t quáº£ deep:

- HybridXAPI + none: F1 = 0.7571.
- HybridXAPI + SMOTE: F1 = 0.7980.
- HybridXAPI + ADASYN: F1 = 0.8243.

So vá»›i paper xAPI:

| Nguá»“n | Accuracy | F1 macro |
|---|---:|---:|
| Paper CNN-BiLSTM + ADASYN | 0.8438 | 0.8447 |
| Project HybridXAPI + ADASYN | 0.8194 | 0.8243 |

Project cÃ²n tháº¥p hÆ¡n paper khoáº£ng:

- Accuracy: -0.0244.
- F1: -0.0204.

LÆ°u Ã½ quan trá»ng vá» baseline xAPI:

- Summary theo validation protocol chá»n Random Forest + Borderline-SMOTE, test F1 = 0.8235.
- Trong raw sweep vá»«a cháº¡y, Gradient Boosting + SMOTE cÃ³ test F1 = 0.9185. Káº¿t quáº£ nÃ y ráº¥t cao nhÆ°ng khÃ´ng nÃªn dÃ¹ng lÃ m mÃ´ hÃ¬nh "Ä‘Æ°á»£c chá»n" náº¿u lá»±a chá»n dá»±a trÃªn test set. CÃ³ thá»ƒ ghi lÃ  "káº¿t quáº£ test cao nháº¥t quan sÃ¡t Ä‘Æ°á»£c trong sweep", khÃ´ng ghi lÃ  model final náº¿u chÆ°a cÃ³ validation protocol tÆ°Æ¡ng á»©ng.

Nháº­n xÃ©t tháº³ng:

- HybridXAPI + ADASYN Ä‘Ã£ cáº£i thiá»‡n rÃµ so vá»›i khÃ´ng cÃ¢n báº±ng: 0.8243 so vá»›i 0.7571, tÄƒng +0.0672 F1.
- HybridXAPI + ADASYN gáº§n báº±ng Random Forest protocol cÅ©: 0.8243 so vá»›i 0.8235, hÆ¡n +0.0008.
- ChÆ°a Ä‘áº¡t paper.
- Deep model xAPI váº«n cáº§n Ä‘Ã¡nh giÃ¡ á»•n Ä‘á»‹nh hÆ¡n náº¿u muá»‘n bÃ¡o cÃ¡o máº¡nh hÆ¡n, vÃ¬ xAPI chá»‰ cÃ³ 480 máº«u.

## 9. Káº¿t quáº£ há»“i quy RMSE vÃ  RÂ²

Nguá»“n: `reports/tables/baseline_regression_summary.csv`.

| Dataset | Scenario | Baseline há»“i quy tá»‘t nháº¥t | RMSE | RÂ² |
|---|---|---|---:|---:|
| student-mat | mid | Random Forest Regressor | 2.2016 | 0.7087 |
| student-mat | late | Gradient Boosting Regressor | 1.3843 | 0.8848 |
| student-por | mid | Random Forest Regressor | 2.1128 | 0.6879 |
| student-por | late | Ridge | 1.4579 | 0.8514 |
| student-combined | mid | Gradient Boosting Regressor | 2.0756 | 0.7060 |
| student-combined | late | Gradient Boosting Regressor | 1.1505 | 0.9097 |

Nháº­n xÃ©t:

- ngoai_pham_vi_hien_tai scenario cÃ³ RÂ² tháº¥p vÃ¬ chÆ°a cÃ³ G1/G2.
- Mid scenario tÄƒng máº¡nh nhá» cÃ³ G1.
- Late scenario tá»‘t nháº¥t vÃ¬ cÃ³ G1/G2, Ä‘áº·c biá»‡t student-combined late Ä‘áº¡t RÂ² = 0.9097.

## 10. Tráº¡ng thÃ¡i pháº§n khuyáº¿n nghá»‹

Project hiá»‡n cÃ³ pháº§n khuyáº¿n nghá»‹ dáº¡ng:

- TrÃ­ch xuáº¥t yáº¿u tá»‘ quan trá»ng báº±ng feature importance.
- Sinh lá»i khuyÃªn há»c táº­p báº±ng rule dá»±a trÃªn Ä‘áº·c trÆ°ng rá»§i ro.

Tráº¡ng thÃ¡i: cÃ³ thá»ƒ dÃ¹ng lÃ m module há»— trá»£ khuyáº¿n nghá»‹ ban Ä‘áº§u, nhÆ°ng chÆ°a pháº£i mÃ´ hÃ¬nh recommendation cÃ³ ground truth.

Äiá»ƒm cáº§n nÃ³i rÃµ trong khÃ³a luáº­n:

- Hiá»‡n chÆ°a cÃ³ nhÃ£n "lá»™ trÃ¬nh há»c Ä‘Ãºng/sai" hoáº·c dá»¯ liá»‡u pháº£n há»“i sau khuyáº¿n nghá»‹.
- VÃ¬ váº­y chÆ°a thá»ƒ Ä‘Ã¡nh giÃ¡ Ä‘á»‹nh lÆ°á»£ng riÃªng mÃ´ hÃ¬nh khuyáº¿n nghá»‹ báº±ng Accuracy/F1/RMSE/RÂ².
- CÃ³ thá»ƒ Ä‘Ã¡nh giÃ¡ giÃ¡n tiáº¿p báº±ng cháº¥t lÆ°á»£ng mÃ´ hÃ¬nh dá»± Ä‘oÃ¡n vÃ  minh há»a case-study khuyáº¿n nghá»‹.

## 11. Káº¿t luáº­n tiáº¿n Ä‘á»™ hiá»‡n táº¡i

ÄÃ£ hoÃ n thÃ nh cháº¯c cÃ¡c pháº§n:

1. LÃ m sáº¡ch vÃ  chuáº©n bá»‹ dá»¯ liá»‡u student-mat, student-por, student-combined, xAPI.
2. Feature engineering cho student datasets.
3. Feature selection Pearson + Chi-square.
4. Xá»­ lÃ½ máº¥t cÃ¢n báº±ng báº±ng SMOTE/ADASYN trÃªn training set.
5. MÃ´ hÃ¬nh deep CNN-BiLSTM cho student data vÃ  HybridXAPI.
6. Baseline phÃ¢n loáº¡i vÃ  há»“i quy.
7. Evaluation báº±ng Accuracy, Precision, Recall, F1, PR-AUC, RMSE, RÂ².
8. BÃ¡o cÃ¡o báº£ng káº¿t quáº£ cuá»‘i.

Káº¿t quáº£ deep final nÃªn trÃ¬nh bÃ y:

- student-combined late: CLSV2 + SMOTE, F1 = 0.8474.
- student-mat late: CLSV2 + none, F1 = 0.8272.
- student-por late: CLSV2 + ADASYN, F1 = 0.8644.
- xAPI: HybridXAPI + ADASYN, F1 = 0.8243.

CÃ¡c háº¡n cháº¿ cáº§n nÃ³i tháº³ng:

- Dá»¯ liá»‡u student-mat/student-por khÃ´ng pháº£i chuá»—i há»c ká»³ dÃ i tháº­t, chá»‰ cÃ³ G1/G2/G3 theo giai Ä‘oáº¡n Ä‘Ã¡nh giÃ¡.
- xAPI nhá», chá»‰ 480 máº«u, nÃªn káº¿t quáº£ dá»… nháº¡y theo split.
- Recommendation chÆ°a cÃ³ nhÃ£n ground truth Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ Ä‘á»‹nh lÆ°á»£ng riÃªng.
- PostgreSQL vÃ  Optuna cÃ³ trong yÃªu cáº§u Ä‘á» cÆ°Æ¡ng/dependency, nhÆ°ng chÆ°a pháº£i pháº§n Ä‘Ã£ hoÃ n thiá»‡n thÃ nh module final trong pipeline hiá»‡n táº¡i.

## 12. File káº¿t quáº£ quan trá»ng

- `reports/tables/final_smote_adasyn_deep_comparison.csv`
- `reports/tables/baseline_classification_summary.csv`
- `reports/tables/baseline_regression_summary.csv`
- `reports/tables/deep_classification_summary.csv`
- `reports/tables/xapi_baseline_summary.csv`
- `reports/tables/xapi_deep_summary.csv`
- `report_temp/baocao_doi_chieu_yeu_cau_de_tai.md`
- `report_temp/baocaotiendo.md`
