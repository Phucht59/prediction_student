# BÃ¡o cÃ¡o Ä‘á»‘i chiáº¿u yÃªu cáº§u Ä‘á» tÃ i vÃ  káº¿t quáº£ mÃ´ hÃ¬nh hiá»‡n táº¡i

NgÃ y cáº­p nháº­t: 2026-05-29

## 1. Káº¿t luáº­n ngáº¯n

Dá»± Ã¡n hiá»‡n táº¡i Ä‘i Ä‘Ãºng hÆ°á»›ng chÃ­nh cá»§a Ä‘á» tÃ i á»Ÿ pháº§n dá»± Ä‘oÃ¡n káº¿t quáº£ há»c táº­p: Ä‘Ã£ cÃ³ pipeline lÃ m sáº¡ch dá»¯ liá»‡u, chuáº©n bá»‹ dá»¯ liá»‡u, xá»­ lÃ½ máº¥t cÃ¢n báº±ng báº±ng SMOTE/ADASYN, mÃ´ hÃ¬nh há»c sÃ¢u káº¿t há»£p CNN vÃ  BiLSTM, vÃ  Ä‘Ã¡nh giÃ¡ báº±ng Accuracy, Precision, Recall, F1-score, Precision-Recall AUC. NgoÃ i ra project cÅ©ng cÃ³ pháº§n há»“i quy baseline Ä‘á»ƒ bÃ¡o cÃ¡o RMSE vÃ  RÂ².

Tuy nhiÃªn cÃ³ hai Ä‘iá»ƒm cáº§n trÃ¬nh bÃ y trung thá»±c trong khÃ³a luáº­n:

1. Dá»¯ liá»‡u student-mat vÃ  student-por khÃ´ng pháº£i chuá»—i thá»i gian há»c ká»³ Ä‘áº§y Ä‘á»§. Hai cá»™t G1/G2/G3 lÃ  Ä‘iá»ƒm theo giai Ä‘oáº¡n Ä‘Ã¡nh giÃ¡ trong mÃ´n há»c, khÃ´ng pháº£i transcript nhiá»u há»c ká»³. VÃ¬ váº­y BiLSTM hiá»‡n táº¡i nÃªn Ä‘Æ°á»£c mÃ´ táº£ lÃ  há»c "tiáº¿n trÃ¬nh Ä‘iá»ƒm G1-G2" hoáº·c "grade progression", khÃ´ng nÃªn viáº¿t lÃ  chuá»—i thá»i gian nhiá»u há»c ká»³ náº¿u khÃ´ng cÃ³ dá»¯ liá»‡u há»c ká»³ tháº­t.
2. Pháº§n khuyáº¿n nghá»‹ hiá»‡n cÃ³ lÃ  khuyáº¿n nghá»‹ dá»±a trÃªn feature importance/rule, chÆ°a pháº£i mÃ´ hÃ¬nh khuyáº¿n nghá»‹ cÃ³ nhÃ£n ground truth. VÃ¬ váº­y chÆ°a thá»ƒ Ä‘Ã¡nh giÃ¡ riÃªng mÃ´ hÃ¬nh khuyáº¿n nghá»‹ báº±ng Accuracy/F1/RMSE/RÂ² náº¿u dá»¯ liá»‡u khÃ´ng cÃ³ nhÃ£n hÃ nh Ä‘á»™ng khuyáº¿n nghá»‹ hoáº·c pháº£n há»“i sau khuyáº¿n nghá»‹.
3. Theo Ä‘á» cÆ°Æ¡ng má»›i, Optuna vÃ  PostgreSQL lÃ  yÃªu cáº§u khÃ¡c cáº§n cÃ³. Project hiá»‡n Ä‘Ã£ thÃªm dependency/config cho hai hÆ°á»›ng nÃ y, nhÆ°ng chÆ°a cÃ³ pipeline Optuna chÃ­nh thá»©c vÃ  chÆ°a cÃ³ module lÆ°u/Ä‘á»c PostgreSQL hoÃ n chá»‰nh Ä‘á»ƒ bÃ¡o cÃ¡o lÃ  Ä‘Ã£ triá»ƒn khai.

## 2. Äá»‘i chiáº¿u tá»«ng yÃªu cáº§u Ä‘á» tÃ i

| YÃªu cáº§u Ä‘á» tÃ i | Tráº¡ng thÃ¡i hiá»‡n táº¡i | Báº±ng chá»©ng trong project | Nháº­n xÃ©t |
|---|---:|---|---|
| Thu tháº­p, lÃ m sáº¡ch vÃ  chuáº©n bá»‹ dá»¯ liá»‡u há»“ sÆ¡ vÃ  Ä‘iá»ƒm sá»‘ sinh viÃªn | Äáº¡t | `src/data/prepare_datasets.py`, `src/data/preprocess.py`, `src/data/prepare_xapi.py`, `data/processed/` | CÃ³ student-mat, student-por, xAPI vÃ  student-combined. |
| Xá»­ lÃ½ máº¥t cÃ¢n báº±ng báº±ng SMOTE/ADASYN | Äáº¡t | `src/features/imbalance.py`, `src/train/train_deep_classification.py`, `src/train/train_xapi_deep.py` | ÄÃ£ cháº¡y cáº£ none, SMOTE, ADASYN. Oversampling Ã¡p dá»¥ng trÃªn train set, khÃ´ng Ã¡p dá»¥ng trÆ°á»›c split. |
| CNN Ä‘á»ƒ trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng tá»« dá»¯ liá»‡u | Äáº¡t | `src/models/deep_learning.py` | StudentCNNBiLSTMV2 dÃ¹ng CNN trÃªn vector Ä‘áº·c trÆ°ng; HybridXAPI dÃ¹ng Conv1D trÃªn chuá»—i hÃ nh vi. |
| BiLSTM Ä‘á»ƒ phÃ¢n tÃ­ch chuá»—i Ä‘iá»ƒm qua há»c ká»³ | Äáº¡t má»™t pháº§n | `src/models/deep_learning.py` | CÃ³ BiLSTM cho grade progression G1/G2, nhÆ°ng dá»¯ liá»‡u khÃ´ng pháº£i chuá»—i há»c ká»³ dÃ i. Cáº§n diá»…n Ä‘áº¡t Ä‘Ãºng giá»›i háº¡n dá»¯ liá»‡u. |
| Dá»± Ä‘oÃ¡n thÃ nh tÃ­ch há»c táº­p sinh viÃªn | Äáº¡t | `src/train/train_deep.py`, `src/train/train_baselines.py` | CÃ³ phÃ¢n loáº¡i 3 má»©c káº¿t quáº£ vÃ  há»“i quy G3 baseline. |
| ÄÃ¡nh giÃ¡ Accuracy, F1-score, Precision-Recall | Äáº¡t | `src/evaluation/metrics.py`, `reports/tables/final_smote_adasyn_deep_comparison.csv` | ÄÃ£ bá»• sung PR-AUC macro/weighted/weak cho mÃ´ hÃ¬nh phÃ¢n loáº¡i. |
| ÄÃ¡nh giÃ¡ RÂ², RMSE | Äáº¡t cho há»“i quy baseline | `src/train/train_baselines.py`, `reports/tables/baseline_regression_summary.csv` | RMSE vÃ  RÂ² dÃ¹ng cho bÃ i toÃ¡n há»“i quy dá»± Ä‘oÃ¡n Ä‘iá»ƒm G3. |
| ÄÃ¡nh giÃ¡ mÃ´ hÃ¬nh khuyáº¿n nghá»‹ | ChÆ°a Ä‘á»§ dá»¯ liá»‡u Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ Ä‘á»‹nh lÆ°á»£ng | `src/recommend/feature_importance.py`, `src/recommend/study_advice.py` | CÃ³ sinh khuyáº¿n nghá»‹ dá»±a trÃªn yáº¿u tá»‘ áº£nh hÆ°á»Ÿng, nhÆ°ng chÆ°a cÃ³ nhÃ£n ground truth Ä‘á»ƒ tÃ­nh Accuracy/F1/RMSE/RÂ² riÃªng cho recommendation. |
| Sá»­ dá»¥ng Optuna tÃ¬m kiáº¿m siÃªu tham sá»‘ | ChÆ°a triá»ƒn khai chÃ­nh thá»©c | `requirements.txt`, `environment.yml`, `config.yaml` | Dependency Ä‘Ã£ sáºµn sÃ ng, nhÆ°ng chÆ°a cÃ³ script Optuna final. KhÃ´ng Ä‘Æ°á»£c bÃ¡o cÃ¡o káº¿t quáº£ Optuna náº¿u chÆ°a cháº¡y tháº­t. |
| PostgreSQL lÆ°u há»“ sÆ¡ vÃ  Ä‘iá»ƒm sá»‘ | ChÆ°a triá»ƒn khai chÃ­nh thá»©c | `requirements.txt`, `environment.yml` | ÄÃ£ cÃ³ dependency SQLAlchemy/psycopg2, nhÆ°ng chÆ°a cÃ³ schema/database pipeline. |

## 3. CÃ¡ch xá»­ lÃ½ máº¥t cÃ¢n báº±ng Ä‘Ã£ triá»ƒn khai

Project hiá»‡n há»— trá»£ cÃ¡c lá»±a chá»n:

- `none`: khÃ´ng xá»­ lÃ½ máº¥t cÃ¢n báº±ng.
- `smote`: táº¡o máº«u tá»•ng há»£p lá»›p thiá»ƒu sá»‘ báº±ng SMOTE.
- `adasyn`: táº¡o máº«u tá»•ng há»£p táº­p trung hÆ¡n vÃ o vÃ¹ng lá»›p thiá»ƒu sá»‘ khÃ³ há»c.
- `borderline_smote`: cÃ³ trong module imbalance, dÃ¹ng khi muá»‘n thá»­ vÃ¹ng biÃªn.

Trong cÃ¡c láº§n cháº¡y cuá»‘i, theo Ä‘Ãºng pháº¡m vi Ä‘á» tÃ i, chá»‰ dÃ¹ng mÃ´ hÃ¬nh final vá»›i none/SMOTE/ADASYN, khÃ´ng dÃ¹ng ká»¹ thuáº­t tuning ngoÃ i pháº¡m vi Ä‘á» tÃ i. CÃ¡ch Ã¡p dá»¥ng:

1. Chia train/test trÆ°á»›c.
2. Fit preprocessing, feature selection trÃªn train.
3. Apply SMOTE hoáº·c ADASYN chá»‰ trÃªn train.
4. Train mÃ´ hÃ¬nh deep learning.
5. Evaluate trÃªn test set gá»‘c, khÃ´ng oversample test.

CÃ¡ch nÃ y trÃ¡nh data leakage. Náº¿u oversampling sau one-hot, project cÃ³ cáº£nh bÃ¡o vÃ¬ SMOTE/ADASYN cÃ³ thá»ƒ táº¡o giÃ¡ trá»‹ phÃ¢n sá»‘ á»Ÿ cá»™t one-hot; Ä‘Ã¢y lÃ  giá»›i háº¡n ká»¹ thuáº­t cáº§n ghi rÃµ trong khÃ³a luáº­n.

## 4. Káº¿t quáº£ phÃ¢n loáº¡i cuá»‘i: trÆ°á»›c vÃ  sau xá»­ lÃ½ máº¥t cÃ¢n báº±ng

Nguá»“n káº¿t quáº£: `reports/tables/final_smote_adasyn_deep_comparison.csv`.

| Dataset | Scenario | Model | Oversampling | Accuracy | Precision macro | Recall macro | F1 macro | PR-AUC macro | Nháº­n xÃ©t |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| student-combined | late | CLSV2 | none | 0.8344 | 0.8346 | 0.8323 | 0.8317 | 0.9382 | TrÆ°á»›c cÃ¢n báº±ng. |
| student-combined | late | CLSV2 | SMOTE | 0.8471 | 0.8402 | 0.8632 | 0.8474 | 0.9317 | Tá»‘t nháº¥t theo F1; SMOTE cáº£i thiá»‡n +0.0157 F1. |
| student-combined | late | CLSV2 | ADASYN | 0.8344 | 0.8366 | 0.8218 | 0.8284 | 0.9300 | KhÃ´ng tá»‘t hÆ¡n SMOTE. |
| student-mat | late | CLSV2 | none | 0.8167 | 0.8310 | 0.8256 | 0.8272 | 0.9114 | Tá»‘t nháº¥t theo F1 trong deep model. |
| student-mat | late | CLSV2 | SMOTE | 0.8000 | 0.8545 | 0.7900 | 0.8083 | 0.9447 | PR-AUC tÄƒng nhÆ°ng F1 giáº£m; SMOTE khÃ´ng nÃªn chá»n cho F1. |
| student-mat | late | CLSV2 | ADASYN | 0.7500 | 0.7795 | 0.7533 | 0.7640 | 0.8791 | KÃ©m nháº¥t. |
| student-por | late | CLSV2 | none | 0.8469 | 0.8184 | 0.8647 | 0.8368 | 0.9585 | TrÆ°á»›c cÃ¢n báº±ng. |
| student-por | late | CLSV2 | SMOTE | 0.8571 | 0.8517 | 0.8334 | 0.8409 | 0.9431 | Cáº£i thiá»‡n nháº¹ F1. |
| student-por | late | CLSV2 | ADASYN | 0.8776 | 0.8808 | 0.8512 | 0.8644 | 0.9462 | Tá»‘t nháº¥t; ADASYN cáº£i thiá»‡n +0.0276 F1. |
| xAPI | behavior | HybridXAPI | none | 0.7500 | 0.7503 | 0.7710 | 0.7571 | 0.8444 | TrÆ°á»›c cÃ¢n báº±ng. |
| xAPI | behavior | HybridXAPI | SMOTE | 0.7917 | 0.7909 | 0.8203 | 0.7980 | 0.8571 | Cáº£i thiá»‡n rÃµ so vá»›i none. |
| xAPI | behavior | HybridXAPI | ADASYN | 0.8194 | 0.8246 | 0.8285 | 0.8243 | 0.8724 | Tá»‘t nháº¥t; ADASYN cáº£i thiá»‡n +0.0672 F1. |

## 5. Lá»±a chá»n mÃ´ hÃ¬nh final theo tá»«ng dataset

| Dataset | MÃ´ hÃ¬nh final Ä‘á» xuáº¥t | LÃ½ do |
|---|---|---|
| student-combined | CLSV2 + SMOTE | F1 macro cao nháº¥t: 0.8474. Viá»‡c gá»™p student-mat vÃ  student-por giÃºp tÄƒng dá»¯ liá»‡u train. |
| student-mat | CLSV2 + none | SMOTE/ADASYN khÃ´ng cáº£i thiá»‡n F1. Vá»›i student-mat nhá», oversampling lÃ m mÃ´ hÃ¬nh há»c máº«u tá»•ng há»£p chÆ°a cÃ³ lá»£i cho F1. |
| student-por | CLSV2 + ADASYN | F1 macro cao nháº¥t: 0.8644. ADASYN cáº£i thiá»‡n tá»‘t hÆ¡n SMOTE. |
| xAPI | HybridXAPI + ADASYN | F1 macro cao nháº¥t: 0.8243, tiá»‡m cáº­n paper xAPI F1 0.8447. |

## 6. So sÃ¡nh vá»›i baseline truyá»n thá»‘ng vÃ  paper

| Dataset | Deep final tá»‘t nháº¥t | Baseline máº¡nh Ä‘Ã£ cÃ³ | So sÃ¡nh |
|---|---:|---:|---|
| student-combined late | CLSV2 + SMOTE F1 0.8474 | GBM F1 0.8422 | Deep cao hÆ¡n +0.0052. |
| student-mat late | CLSV2 none F1 0.8272 | GBM F1 0.9372 | Deep tháº¥p hÆ¡n -0.1100. NguyÃªn nhÃ¢n chÃ­nh: dá»¯ liá»‡u nhá» vÃ  G1/G2 tÆ°Æ¡ng quan ráº¥t máº¡nh vá»›i G3, tree model khai thÃ¡c tá»‘t hÆ¡n. |
| student-por late | CLSV2 + ADASYN F1 0.8644 | RF F1 0.8581 | Deep cao hÆ¡n +0.0063. |
| xAPI | HybridXAPI + ADASYN F1 0.8243 | RF khoáº£ng F1 0.8235 | Deep cao hÆ¡n ráº¥t nháº¹ +0.0008. |
| xAPI so vá»›i paper | Accuracy 0.8194, F1 0.8243 | Paper Accuracy 0.8438, F1 0.8447 | CÃ²n tháº¥p hÆ¡n paper khoáº£ng 0.0204 F1 vÃ  0.0244 Accuracy. |

Káº¿t luáº­n so sÃ¡nh: sau khi Ä‘Æ°a SMOTE/ADASYN vÃ o pipeline final, deep model Ä‘Ã£ há»£p lÃ½ hÆ¡n, Ä‘áº·c biá»‡t lÃ  xAPI. Tuy nhiÃªn khÃ´ng nÃªn tuyÃªn bá»‘ mÃ´ hÃ¬nh deep vÆ°á»£t táº¥t cáº£ baseline. Vá»›i student-mat late, GBM váº«n vÆ°á»£t xa do dá»¯ liá»‡u Ã­t vÃ  quan há»‡ G1/G2 -> G3 quÃ¡ máº¡nh, phÃ¹ há»£p vá»›i mÃ´ hÃ¬nh cÃ¢y hÆ¡n.

## 7. Káº¿t quáº£ há»“i quy: RMSE vÃ  RÂ²

Nguá»“n káº¿t quáº£: `reports/tables/baseline_regression_summary.csv`.

| Dataset | Scenario | Best regression baseline | RMSE | RÂ² | Nháº­n xÃ©t |
|---|---|---|---:|---:|---|
| student-mat | mid | Gradient Boosting Regressor | 2.3848 | 0.6582 | CÃ³ G1 nÃªn RÂ² tÄƒng máº¡nh. |
| student-mat | late | Gradient Boosting Regressor | 1.5281 | 0.8597 | CÃ³ G1/G2 nÃªn dá»± Ä‘oÃ¡n Ä‘iá»ƒm G3 ráº¥t tá»‘t. |
| student-por | mid | Random Forest Regressor | 2.1008 | 0.6914 | CÃ³ G1 cáº£i thiá»‡n rÃµ. |
| student-por | late | Ridge | 1.5237 | 0.8377 | Quan há»‡ G1/G2 vá»›i G3 gáº§n tuyáº¿n tÃ­nh, Ridge hoáº¡t Ä‘á»™ng tá»‘t. |

RÂ² vÃ  RMSE nÃªn Ä‘Æ°á»£c dÃ¹ng Ä‘á»ƒ bÃ¡o cÃ¡o bÃ i toÃ¡n há»“i quy dá»± Ä‘oÃ¡n Ä‘iá»ƒm sá»‘ G3. CÃ²n bÃ i toÃ¡n phÃ¢n loáº¡i káº¿t quáº£ há»c táº­p nÃªn dÃ¹ng Accuracy, Precision, Recall, F1 vÃ  PR-AUC.

## 8. Giáº£i thÃ­ch káº¿t quáº£ mÃ´ hÃ¬nh

### student-combined

Gá»™p student-mat vÃ  student-por giÃºp tÄƒng sá»‘ máº«u train, nÃªn mÃ´ hÃ¬nh deep á»•n Ä‘á»‹nh hÆ¡n so vá»›i train tá»«ng dataset nhá». SMOTE cáº£i thiá»‡n F1 tá»« 0.8317 lÃªn 0.8474. ÄÃ¢y lÃ  káº¿t quáº£ phÃ¹ há»£p vá»›i má»¥c tiÃªu xá»­ lÃ½ máº¥t cÃ¢n báº±ng trong Ä‘á» tÃ i.

### student-mat

MÃ´ hÃ¬nh deep tá»‘t nháº¥t láº¡i lÃ  khÃ´ng oversampling. SMOTE lÃ m PR-AUC tÄƒng nhÆ°ng F1 giáº£m, nghÄ©a lÃ  xÃ¡c suáº¥t dá»± Ä‘oÃ¡n cÃ³ thá»ƒ phÃ¢n háº¡ng tá»‘t hÆ¡n nhÆ°ng ngÆ°á»¡ng phÃ¢n loáº¡i cuá»‘i chÆ°a tá»‘t hÆ¡n. Student-mat chá»‰ cÃ³ khoáº£ng 395 máº«u, train set cÃ²n nhá» hÆ¡n, trong khi G1/G2 Ä‘Ã£ dá»± bÃ¡o G3 ráº¥t máº¡nh. VÃ¬ váº­y GBM Ä‘áº¡t F1 ráº¥t cao vÃ  deep model khÃ³ vÆ°á»£t.

### student-por

ADASYN lÃ  lá»±a chá»n tá»‘t nháº¥t, tÄƒng F1 tá»« 0.8368 lÃªn 0.8644. Dá»¯ liá»‡u student-por lá»›n hÆ¡n student-mat, nÃªn synthetic samples tá»« ADASYN há»¯u Ã­ch hÆ¡n vÃ  giÃºp mÃ´ hÃ¬nh há»c lá»›p yáº¿u tá»‘t hÆ¡n.

### xAPI

HybridXAPI + ADASYN tÄƒng F1 tá»« 0.7571 lÃªn 0.8243. ÄÃ¢y lÃ  cáº£i thiá»‡n lá»›n nháº¥t sau xá»­ lÃ½ máº¥t cÃ¢n báº±ng. MÃ´ hÃ¬nh dÃ¹ng 4 Ä‘áº·c trÆ°ng hÃ nh vi `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion` lÃ m nhÃ¡nh sequence, cÃ¡c Ä‘áº·c trÆ°ng cÃ²n láº¡i lÃ m static branch. Káº¿t quáº£ tiá»‡m cáº­n paper nhÆ°ng chÆ°a vÆ°á»£t paper.

## 9. Má»©c Ä‘á»™ Ä‘Ãºng vá»›i Ä‘á» tÃ i vÃ  cÃ¡ch nÃªn trÃ¬nh bÃ y

CÃ¡ch trÃ¬nh bÃ y phÃ¹ há»£p trong khÃ³a luáº­n:

"Äá» tÃ i xÃ¢y dá»±ng mÃ´ hÃ¬nh dá»± Ä‘oÃ¡n káº¿t quáº£ há»c táº­p sinh viÃªn trÃªn cÃ¡c bá»™ dá»¯ liá»‡u student-mat, student-por vÃ  xAPI. Dá»¯ liá»‡u Ä‘Æ°á»£c tiá»n xá»­ lÃ½, mÃ£ hÃ³a, chá»n Ä‘áº·c trÆ°ng vÃ  xá»­ lÃ½ máº¥t cÃ¢n báº±ng báº±ng SMOTE/ADASYN trÃªn táº­p train. MÃ´ hÃ¬nh Ä‘á» xuáº¥t káº¿t há»£p CNN Ä‘á»ƒ trÃ­ch xuáº¥t biá»ƒu diá»…n tá»« vector Ä‘áº·c trÆ°ng vÃ  BiLSTM Ä‘á»ƒ há»c tiáº¿n trÃ¬nh Ä‘iá»ƒm G1-G2 hoáº·c chuá»—i hÃ nh vi há»c táº­p. Káº¿t quáº£ Ä‘Æ°á»£c Ä‘Ã¡nh giÃ¡ báº±ng Accuracy, Precision, Recall, F1-score vÃ  PR-AUC cho phÃ¢n loáº¡i; RMSE vÃ  RÂ² cho há»“i quy Ä‘iá»ƒm sá»‘."

KhÃ´ng nÃªn viáº¿t:

"BiLSTM phÃ¢n tÃ­ch chuá»—i thá»i gian Ä‘iá»ƒm sá»‘ qua nhiá»u há»c ká»³" náº¿u chá»‰ dÃ¹ng student-mat/student-por gá»‘c, vÃ¬ dá»¯ liá»‡u khÃ´ng cÃ³ lá»‹ch sá»­ nhiá»u há»c ká»³ tháº­t.

NÃªn viáº¿t:

"BiLSTM há»c biá»ƒu diá»…n tá»« tiáº¿n trÃ¬nh Ä‘iá»ƒm G1-G2 trong student datasets vÃ  chuá»—i hÃ nh vi há»c táº­p trong xAPI. Vá»›i dá»¯ liá»‡u há»c ká»³ Ä‘áº§y Ä‘á»§ trong tÆ°Æ¡ng lai, kiáº¿n trÃºc nÃ y cÃ³ thá»ƒ má»Ÿ rá»™ng thÃ nh chuá»—i thá»i gian há»c ká»³ thá»±c."

## 10. Pháº§n cÃ²n thiáº¿u náº¿u giáº£ng viÃªn yÃªu cáº§u Ä‘Ãºng tuyá»‡t Ä‘á»‘i

1. Náº¿u yÃªu cáº§u báº¯t buá»™c "Ä‘iá»ƒm sá»‘ qua cÃ¡c há»c ká»³", cáº§n bá»• sung dataset transcript tháº­t gá»“m nhiá»u há»c ká»³/mÃ´n há»c theo tá»«ng sinh viÃªn. Student-mat/student-por hiá»‡n khÃ´ng Ä‘á»§ cho yÃªu cáº§u nÃ y.
2. Náº¿u yÃªu cáº§u Ä‘Ã¡nh giÃ¡ "mÃ´ hÃ¬nh khuyáº¿n nghá»‹", cáº§n cÃ³ nhÃ£n khuyáº¿n nghá»‹ hoáº·c dá»¯ liá»‡u can thiá»‡p: vÃ­ dá»¥ sinh viÃªn nháº­n khuyáº¿n nghá»‹ gÃ¬, cÃ³ lÃ m theo khÃ´ng, káº¿t quáº£ sau khuyáº¿n nghá»‹ ra sao. Hiá»‡n táº¡i project chá»‰ cÃ³ thá»ƒ sinh khuyáº¿n nghá»‹ dá»±a trÃªn yáº¿u tá»‘ áº£nh hÆ°á»Ÿng, chÆ°a thá»ƒ Ä‘Ã¡nh giÃ¡ Ä‘á»‹nh lÆ°á»£ng recommendation.
3. Náº¿u muá»‘n dÃ¹ng SMOTENC chuáº©n hÆ¡n, cáº§n Ã¡p dá»¥ng oversampling trÆ°á»›c one-hot vá»›i danh sÃ¡ch categorical indices. Hiá»‡n project Ä‘Ã£ cÃ³ module SMOTENC nhÆ°ng cÃ¡c káº¿t quáº£ final Ä‘ang dÃ¹ng SMOTE/ADASYN theo pipeline sau encoding.
4. Cáº§n bá»• sung script Optuna final cho mÃ´ hÃ¬nh há»c sÃ¢u náº¿u muá»‘n Ä‘Ã¡p á»©ng Ä‘áº§y Ä‘á»§ Ä‘á» cÆ°Æ¡ng pháº§n tÃ¬m kiáº¿m siÃªu tham sá»‘.
5. Cáº§n bá»• sung module PostgreSQL náº¿u muá»‘n Ä‘Ã¡p á»©ng Ä‘áº§y Ä‘á»§ Ä‘á» cÆ°Æ¡ng pháº§n lÆ°u trá»¯ há»“ sÆ¡ vÃ  Ä‘iá»ƒm sá»‘.

## 11. Command Ä‘Ã£ cháº¡y Ä‘á»ƒ ra káº¿t quáº£ final

```powershell
py -3 src\train\train_deep.py --dataset all --scenario late --model clsv2 --loss-weight none --oversampling smote --feature-selection pearson_chi2 --max-features 40 --epochs 80 --patience 12 --batch-size 32 --lr 0.001 --seed 42
py -3 src\train\train_deep.py --dataset all --scenario late --model clsv2 --loss-weight none --oversampling adasyn --feature-selection pearson_chi2 --max-features 40 --epochs 80 --patience 12 --batch-size 32 --lr 0.001 --seed 42
py -3 src\train\train_deep.py --dataset all --scenario late --model clsv2 --loss-weight none --oversampling none --feature-selection pearson_chi2 --max-features 40 --epochs 80 --patience 12 --batch-size 32 --lr 0.001 --seed 42
py -3 src\train\train_deep.py --dataset xapi --model hybrid_xapi --oversampling smote --feature-selection pearson_chi2 --max-features 56 --epochs 100 --patience 15 --batch-size 16 --lr 0.001 --seed 42 --fusion concat --label-smoothing 0.0 --scheduler none
py -3 src\train\train_deep.py --dataset xapi --model hybrid_xapi --oversampling adasyn --feature-selection pearson_chi2 --max-features 56 --epochs 100 --patience 15 --batch-size 16 --lr 0.001 --seed 42 --fusion concat --label-smoothing 0.0 --scheduler none
py -3 src\train\train_deep.py --dataset xapi --model hybrid_xapi --oversampling none --feature-selection pearson_chi2 --max-features 56 --epochs 100 --patience 15 --batch-size 16 --lr 0.001 --seed 42 --fusion concat --label-smoothing 0.0 --scheduler none
py -3 -m src.evaluation.summarize_all
```

## 12. Káº¿t luáº­n cuá»‘i

Project hiá»‡n Ä‘Ã£ phÃ¹ há»£p vá»›i pháº¡m vi chÃ­nh cá»§a Ä‘á» tÃ i náº¿u trÃ¬nh bÃ y lÃ  bÃ i toÃ¡n dá»± Ä‘oÃ¡n káº¿t quáº£ há»c táº­p káº¿t há»£p CNN-BiLSTM vÃ  xá»­ lÃ½ máº¥t cÃ¢n báº±ng báº±ng SMOTE/ADASYN. Káº¿t quáº£ tá»‘t nháº¥t sau cÃ¢n báº±ng lÃ :

- student-combined: CLSV2 + SMOTE, F1 macro 0.8474.
- student-mat: CLSV2 khÃ´ng oversampling, F1 macro 0.8272.
- student-por: CLSV2 + ADASYN, F1 macro 0.8644.
- xAPI: HybridXAPI + ADASYN, F1 macro 0.8243.

Äiá»ƒm cáº§n nÃ³i rÃµ vá»›i giáº£ng viÃªn lÃ  dá»¯ liá»‡u hiá»‡n táº¡i chÆ°a pháº£i chuá»—i há»c ká»³ Ä‘áº§y Ä‘á»§ vÃ  pháº§n khuyáº¿n nghá»‹ chÆ°a cÃ³ ground truth Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ Ä‘á»‹nh lÆ°á»£ng. Náº¿u cháº¥p nháº­n giá»›i háº¡n dá»¯ liá»‡u nÃ y, project Ä‘á»§ cÆ¡ sá»Ÿ Ä‘á»ƒ trÃ¬nh bÃ y trong khÃ³a luáº­n theo hÆ°á»›ng dá»± Ä‘oÃ¡n káº¿t quáº£ há»c táº­p vÃ  sinh khuyáº¿n nghá»‹ dá»±a trÃªn yáº¿u tá»‘ áº£nh hÆ°á»Ÿng.
