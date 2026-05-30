# BAO CAO TIEN DO DU AN

## 1. Thong tin chung

- Ten de tai: Du doan ket qua hoc tap cua sinh vien bang hoc may va hoc sau
- Cong nghe chinh: Python, pandas, scikit-learn, imbalanced-learn, PyTorch, SQLAlchemy
- Du lieu dang su dung:
  - UCI Student Performance: `student-mat`, `student-por`, `student-combined`
  - xAPI Student Academic Performance: `xAPI-Edu-Data`
- Ngay cap nhat: 30/05/2026

## 2. Muc tieu giai doan hien tai

Trong giai doan nay, du an tap trung vao viec hoan thien pipeline thuc nghiem va uu tien mo hinh hoc sau. Cac muc tieu chinh gom:

- Chuan hoa tien xu ly du lieu va chia tap train/validation/test.
- Chuan hoa pham vi thuc nghiem Student Performance, chi giu lai hai kich ban `mid` va `late`.
- Giu nguyen kiem tra ro ri du lieu:
  - `mid`: duoc dung G1, khong duoc dung G2 va G3.
  - `late`: duoc dung G1 va G2, khong duoc dung G3.
  - xAPI: khong dua cot target `Class` vao feature.
- Chay lai mo hinh deep learning cho cac bo du lieu chinh.
- Thu nghiem them class weight de so sanh voi SMOTE va ADASYN.
- Bo sung huong tich hop PostgreSQL de luu du lieu sinh vien va ket qua du doan sau nay.

## 3. Cong viec da thuc hien

### 3.1. Tien xu ly va chia du lieu

Pipeline tien xu ly da duoc cap nhat de chi sinh du lieu cho hai kich ban `mid` va `late`. Cac output cu khong thuoc pham vi hien tai da duoc don dep khoi `data/processed`.

Ket qua chia du lieu da duoc kiem tra lai:

| Dataset | Scenario | Train | Validation | Test | Leakage check |
| --- | --- | ---: | ---: | ---: | --- |
| student-mat | mid | 275 | 60 | 60 | Pass |
| student-mat | late | 275 | 60 | 60 | Pass |
| student-por | mid | 453 | 98 | 98 | Pass |
| student-por | late | 453 | 98 | 98 | Pass |
| student-combined | mid | 730 | 157 | 157 | Pass |
| student-combined | late | 730 | 157 | 157 | Pass |
| xAPI | xapi_behavior | 336 | 72 | 72 | Pass |

Ket qua tren cho thay ty le chia du lieu van giu on dinh, dong thoi khong co dau hieu ro ri target vao feature.

### 3.2. Cap nhat pipeline huan luyen

Da cap nhat cac module huan luyen sau:

- `src.train.train_baselines`
- `src.train.train_imbalance_baselines`
- `src.train.train_deep_classification`
- `src.train.train_xapi_deep`
- `src.train.train_deep`
- Cac script tong hop trong `scripts/`

Pipeline hien tai ho tro:

- Baseline truyen thong.
- Xu ly mat can bang lop bang SMOTE, ADASYN, random over-sampling va class weight.
- Deep learning voi CNN-BiLSTM/CLSV2 cho Student Performance.
- Hybrid xAPI deep model cho du lieu xAPI.
- Tong hop ket qua ra CSV va report trong `reports/tables`.

### 3.3. Mo hinh deep learning da chay

Mo hinh deep learning da duoc chay lai voi cau hinh chinh:

- Student datasets:
  - Mo hinh: `clsv2`
  - Scenario: `mid`, `late`
  - Feature selection: `pearson_chi2`
  - So feature toi da: 40
  - Imbalance strategies: `none`, `smote`, `adasyn`, `class_weight_balanced`
- xAPI:
  - Mo hinh: `hybrid_xapi`
  - Scenario: `xapi_behavior`
  - Feature selection: `pearson_chi2`
  - So feature toi da: 56
  - Imbalance strategies: `none`, `smote`, `adasyn`, `class_weight_balanced`

File ket qua chinh:

- `reports/results/deep_classification_results.csv`
- `reports/results/xapi_deep_results.csv`
- `reports/tables/deep_classification_summary.csv`
- `reports/tables/final_imbalance_deep_comparison.csv`

## 4. Ket qua hien tai

### 4.1. Ket qua deep learning tren Student Performance

Bang duoi day tom tat mo hinh deep tot nhat theo validation Macro-F1 cho tung dataset/scenario:

| Dataset | Scenario | Best model | Loss weight | Test Macro-F1 | Test Accuracy | Test Recall weak |
| --- | --- | --- | --- | ---: | ---: | ---: |
| student-combined | mid | clsv2 | none | 0.7324 | 0.7389 | 0.7143 |
| student-combined | late | clsv2 | none | 0.8317 | 0.8344 | 0.8286 |
| student-mat | mid | clsv2 | balanced | 0.7247 | 0.7167 | 0.9000 |
| student-mat | late | clsv2 | none | 0.8272 | 0.8167 | 0.8500 |
| student-por | mid | clsv2 | none | 0.7625 | 0.7755 | 0.7333 |
| student-por | late | clsv2 | none | 0.8368 | 0.8469 | 0.8667 |

Nhan xet:

- Scenario `late` cho ket qua tot hon `mid`, vi co them thong tin G2 va cac dac trung tien trinh diem.
- `student-por late` va `student-combined late` dang cho ket qua on dinh nhat.
- Class weight co tac dung ro trong mot so truong hop, dac biet voi recall lop yeu.

### 4.2. So sanh imbalance trong deep learning

Bang xep hang theo Test Macro-F1 cho cac cau hinh deep learning quan trong:

| Dataset | Scenario | Strategy | Test Macro-F1 | Accuracy | Recall weak |
| --- | --- | --- | ---: | ---: | ---: |
| student-combined | late | smote | 0.8474 | 0.8471 | 0.8857 |
| student-combined | late | class_weight_balanced | 0.8472 | 0.8471 | 0.8857 |
| student-mat | late | class_weight_balanced | 0.8448 | 0.8333 | 0.8500 |
| student-por | late | class_weight_balanced | 0.8829 | 0.8878 | 0.8667 |
| xAPI | xapi_behavior | adasyn | 0.8243 | 0.8194 | 0.8947 |
| xAPI | xapi_behavior | smote | 0.7980 | 0.7917 | 0.9474 |
| xAPI | xapi_behavior | class_weight_balanced | 0.7976 | 0.7917 | 0.8947 |

Nhan xet:

- Doi voi Student Performance, class weight cho ket qua rat canh tranh va dat ket qua tot nhat tren `student-mat late` va `student-por late`.
- Tren `student-combined late`, SMOTE va class weight gan nhu ngang nhau.
- Doi voi xAPI, ADASYN dat Macro-F1 cao nhat, trong khi SMOTE dat recall lop yeu cao nhat.
- Ket qua tong hop cho thay balanced loss cai thien recall lop yeu trong 2/6 cap chay deep, voi delta trung binh +0.0333.

### 4.3. Baseline va imbalance baseline

Da chay baseline truyen thong va class weight baseline de doi chieu. Ket qua imbalance baseline cho thay:

- `class_weight_balanced` thang baseline theo Test Macro-F1 o 2/6 cap dataset-scenario.
- Weak recall chi cai thien o 1/6 cap khi chon theo validation Macro-F1.
- Mot so cau hinh RandomForest/DecisionTree co dau hieu overfit do train Macro-F1 dat 1.0 nhung validation/test thap hon.

Ket qua nay cung co y nghia doi chieu: class weight huu ich nhat khi ket hop voi deep learning, con voi baseline truyen thong thi can can than vi co the lam giam Macro-F1 hoac recall tren mot so bo du lieu.

## 5. Tich hop PostgreSQL

Da bo sung tang luu tru PostgreSQL dang tuy chon:

- `src/storage/postgres.py`
- `scripts/init_postgres.py`
- `scripts/import_processed_to_postgres.py`
- `.env.example`

Schema hien co gom:

- `student_records`: luu du lieu sinh vien da xu ly va ban ghi raw.
- `prediction_results`: luu ket qua du doan sau khi co inference script/API.

Trang thai hien tai:

- Da co ma tao bang va import du lieu da preprocess.
- Chua bat buoc dung PostgreSQL trong pipeline train, de tranh lam phu thuoc vao database khi chay thuc nghiem.
- Bang `prediction_results` da san sang cho giai doan xay dung module du doan/inference.

## 6. Cac thay doi quan trong trong ma nguon

- Chuan hoa config, preprocessing, training va evaluation theo hai scenario `mid`, `late`.
- Cap nhat cac summary script de chi tong hop cac row thuoc pham vi hien tai.
- Them `class_weight_balanced` vao nhom thuc nghiem imbalance.
- Cap nhat `run_train_deep.py` de chay day du:
  - `none`
  - `smote`
  - `adasyn`
  - `class_weight_balanced`
- Cap nhat `run_evaluate.py` va `summarize_all.py` de tong hop ket qua moi.
- Sua loi summary khi co nhieu imbalance strategy cung luc trong phan phan tich loss weight.

## 7. Danh gia tien do

Muc do hoan thanh hien tai:

| Hang muc | Trang thai |
| --- | --- |
| Thu thap va chuan bi du lieu | Da hoan thanh |
| Tien xu ly va chia train/val/test | Da hoan thanh |
| Kiem tra leakage | Da hoan thanh va pass |
| Baseline ML | Da hoan thanh |
| Xu ly imbalance bang SMOTE/ADASYN/class weight | Da hoan thanh thuc nghiem chinh |
| Deep learning cho Student Performance | Da chay xong |
| Deep learning cho xAPI | Da chay xong |
| Tong hop ket qua va bao cao bang CSV/TXT | Da hoan thanh |
| PostgreSQL optional storage | Da co khung tich hop |
| Inference/API du doan | Chua thuc hien |
| Giao dien ung dung | Chua thuc hien |
| Danh gia voi multi-seed/Optuna | Chua thuc hien day du |

## 8. Han che hien tai

- Du lieu Student Performance chi co G1, G2, G3, khong phai chuoi hoc ky dai han. Vi vay BiLSTM chi nen duoc mo ta la hoc bieu dien tien trinh diem ngan, khong phai mo hinh chuoi thoi gian hoc ky day du.
- xAPI khong co G1/G2/G3, nen mo hinh hybrid xAPI su dung chuoi hanh vi ngan gom raised hands, visited resources, announcement views va discussion.
- Ket qua hien tai moi chay voi seed 42, chua co multi-seed de danh gia do on dinh thong ke.
- PostgreSQL moi dung o muc persistence optional, chua ket noi voi module inference thuc te.
- Chua co API/web app de nhap thong tin sinh vien va tra ve ket qua du doan.

## 9. Ke hoach tiep theo

1. Hoan thien module inference cho mo hinh deep learning tot nhat.
2. Ket noi inference voi PostgreSQL de luu input sinh vien va ket qua du doan.
3. Xay dung API hoac giao dien don gian de nhap du lieu sinh vien.
4. Chay them multi-seed cho cac cau hinh tot nhat de danh gia do on dinh.
5. Neu con thoi gian, them Optuna cho viec tinh chinh hyperparameter.
6. Hoan thien noi dung bao cao khoa luan:
   - Mo ta du lieu.
   - Quy trinh tien xu ly.
   - Mo hinh baseline va deep learning.
   - Xu ly mat can bang lop.
   - Ket qua thuc nghiem va phan tich.
   - Han che va huong phat trien.

## 10. Ket luan tam thoi

Du an da hoan thanh phan pipeline thuc nghiem cot loi. Cac buoc tien xu ly, kiem tra leakage, huan luyen baseline, huan luyen deep learning va tong hop ket qua da chay thanh cong. Ket qua hien tai cho thay mo hinh deep learning `clsv2` dat ket qua tot tren cac bo Student Performance, dac biet o scenario `late`. Doi voi xAPI, cau hinh `hybrid_xapi` voi ADASYN dang cho Macro-F1 tot nhat.

Trong giai doan tiep theo, trong tam nen chuyen tu thuc nghiem sang ung dung: xay dung module du doan, ket noi PostgreSQL va chuan bi giao dien/API de phuc vu demo khoa luan.
