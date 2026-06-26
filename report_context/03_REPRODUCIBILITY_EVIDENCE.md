# 03. Reproducibility Evidence

Tài liệu này ghi lại môi trường, lệnh kiểm tra đã chạy, artifact hiện có và các điều kiện cần để tái lập kết quả. Không có training nặng được chạy trong quá trình forensic audit.

## 1. Python Version

| Runtime | Kết quả | Trạng thái | Evidence |
|---|---|---|---|
| Default `python` | Python 3.14, thiếu `pytest` | Không phù hợp để chạy test hiện tại | Command `python -m pytest -q` trả lỗi `No module named pytest` |
| `py -3.10` | Python 3.10.8 | Phù hợp với `environment.yml` | Command `py -3.10 --version` |

## 2. Package Và Dependency

`requirements.txt` khai báo:

```text
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
torch
optuna
sqlalchemy
psycopg2-binary
python-dotenv
joblib
xgboost
pytest
```

`environment.yml` khai báo môi trường `student-performance-paper`, `python=3.10`, kênh `pytorch`, `conda-forge`, `defaults`, và các thư viện chính: pandas, numpy, scikit-learn, imbalanced-learn, pytorch, matplotlib, seaborn, pyyaml, joblib, sqlalchemy, psycopg2, python-dotenv, optuna, xgboost, pytest.

Package versions đã kiểm tra với `py -3.10 -m pip show`:

| Package | Version đã thấy | Evidence |
|---|---:|---|
| pandas | 2.3.3 | `py -3.10 -m pip show pandas` |
| numpy | 2.2.6 | `py -3.10 -m pip show numpy` |
| scikit-learn | 1.7.2 | `py -3.10 -m pip show scikit-learn` |
| imbalanced-learn | 0.14.1 | `py -3.10 -m pip show imbalanced-learn` |
| matplotlib | 3.10.9 | `py -3.10 -m pip show matplotlib` |
| seaborn | 0.13.2 | `py -3.10 -m pip show seaborn` |
| torch | 2.12.0 | `py -3.10 -m pip show torch` |
| optuna | 4.8.0 | `py -3.10 -m pip show optuna` |
| xgboost | 3.2.0 | `py -3.10 -m pip show xgboost` |
| pytest | 9.0.3 | `py -3.10 -m pip show pytest` |

## 3. Dataset Paths

| Dataset | Expected raw path | Current workspace status | Processing output path | Current processed status | Evidence |
|---|---|---|---|---|---|
| student-mat | `data/raw/student-mat.csv` | Không có; `data/raw` chỉ có `.gitkeep` | `data/processed/final/` | Trống | `src/config.py`, filesystem check |
| student-por | `data/raw/student-por.csv` | Không có; `data/raw` chỉ có `.gitkeep` | `data/processed/final/` | Trống | `src/config.py`, filesystem check |
| xAPI | `data/raw/xAPI-Edu-Data.csv` | Không có; `data/raw` chỉ có `.gitkeep` | `data/processed/final/` | Trống | `src/config.py`, filesystem check |

`.gitignore` loại trừ dữ liệu thô và dữ liệu processed. Vì vậy repository hiện tại không đủ raw/processed data để tái lập toàn bộ split/training nếu không bổ sung dataset ngoài repo.

## 4. Model Checkpoints

| Location | Nội dung | Trạng thái tái lập | Evidence |
|---|---|---|---|
| `models/saved/final/` | Trống | Script final pipeline và recommender pipeline hiện tại kỳ vọng checkpoint/metadata ở đây, nên không rerun được full final | Filesystem check, `scripts/run_pipeline.py`, `scripts/run_recommender_pipeline.py` |
| `models/final/` | Có nhiều checkpoint `.pt` và `final_model_manifest.json` | Artifact tồn tại nhưng không khớp hoàn toàn với metrics final trong `reports/final`; Student checkpoints có dấu hiệu 5-class/non-final | Filesystem check, `models/final/final_model_manifest.json`, checkpoint metadata inspection |
| `models/recommendation/` | `xapi_mlp.pt`, `student-por_mlp.pt`, `student-mat_mlp.pt` | Risk diagnosis head checkpoints có metadata; xAPI/student-por có final recommender output | Checkpoint metadata, `outputs/recommender/*/recommender_metrics.json` |

## 5. Lệnh Chạy Test

| Command | Kết quả | Trạng thái |
|---|---|---|
| `python -m pytest -q` | Fail: default Python 3.14 không có pytest | Environment mismatch, không phải test failure của repo |
| `py -3.10 -m pytest -q` | `31 passed in 20.57s` | `PASS` |

## 6. Lệnh Chạy Training Nếu Muốn Tái Lập

Các lệnh dưới đây được xác định từ source/README nhưng không được chạy trong forensic audit vì có thể là training nặng và hiện thiếu raw/processed data/checkpoint metadata.

```powershell
py -3.10 scripts\run_pipeline.py --dataset xapi
py -3.10 scripts\run_pipeline.py --dataset student-por
py -3.10 scripts\run_pipeline.py --dataset student-mat
```

Điều kiện bắt buộc trước khi chạy:

- Khôi phục `data/raw/student-mat.csv`, `data/raw/student-por.csv`, `data/raw/xAPI-Edu-Data.csv`.
- Tạo lại hoặc cung cấp `data/processed/final/*`.
- Cung cấp đúng `models/saved/final/*_best_params.json`, `*_seed*.pt`, `*_ensemble_features.json` nếu dùng recommender pipeline hiện tại.
- Xác minh script nào tạo đúng `sequence_cnn_bilstm_only` Student final rows và `gated_fusion_v28` xAPI final row.

## 7. Lệnh Chạy Evaluation

Evaluation final được đọc từ artifact hiện có, không rerun từ checkpoint.

Artifact final chính:

| Artifact | Nội dung | Trạng thái |
|---|---|---|
| `reports/final/final_model_manifest.json` | Manifest xAPI final gated fusion v28 | `VERIFIED` |
| `reports/final/final_deep_results_table.csv` | Deep result xAPI final | `VERIFIED` |
| `reports/final/final_baseline_comparison.csv` | Deep vs RandomForest baseline xAPI | `VERIFIED` |
| `reports/final/FINAL_PROJECT_STATUS.md` | 4-row project status gồm Student và xAPI | xAPI `VERIFIED`; Student rows `PARTIALLY VERIFIED` |
| `reports/final/final_prediction_model_report.md` | Guardrails và summary prediction | `VERIFIED` cho xAPI/guardrails |

## 8. Lệnh Chạy Recommender

Script chính:

```powershell
py -3.10 scripts\run_recommender_pipeline.py --dataset xapi
py -3.10 scripts\run_recommender_pipeline.py --dataset student-por
py -3.10 scripts\run_recommender_pipeline.py --dataset student-mat
```

Không chạy trong audit vì `models/saved/final/` trống và processed splits không có. Source cho thấy script cần:

- `models/saved/final/{dataset}_3class_best_params.json`
- `models/saved/final/{dataset}_3class_cnn_bilstm_mlp_seed{seed}.pt`
- Optional `models/saved/final/{dataset}_3class_ensemble_features.json`
- Processed split từ `data/processed/final`

Recommender outputs hiện có và đã đọc:

| Dataset | Metrics file | Output rows đã xác minh | Trạng thái |
|---|---|---:|---|
| xAPI | `outputs/recommender/xapi/recommender_metrics.json` | 96 risk predictions, 96 learning paths, 480 recommendation rows | `VERIFIED offline output` |
| student-por | `outputs/recommender/student-por/recommender_metrics.json` | 130 risk predictions, 130 learning paths, 563 recommendation rows | `VERIFIED offline output` |
| student-mat | Stale archive only | 79 stale risk predictions trong archive | `PENDING final refreshed output` |

## 9. Lệnh Chạy Dashboard Nếu Có

Không tìm thấy dashboard app/runtime final đủ rõ để ghi lệnh chạy dashboard như một thành phần đã xác minh. Có thể có report/docx/figures trong `reports/final`, nhưng không có evidence đủ mạnh để claim dashboard vận hành.

Trạng thái: `MISSING`.

## 10. Lệnh Tạo Visual Pack

Visual pack trong `report_context/figures/` được tạo bằng:

```powershell
py -3.10 report_context\figures\create_verified_figures.py
```

Kết quả đã tạo:

- `report_context/figures/fig_01_prediction_metrics.png`
- `report_context/figures/fig_02_xapi_baseline_comparison.png`
- `report_context/figures/fig_03_low_class_focus.png`
- `report_context/figures/fig_04_macro_f1_ranking.png`
- `report_context/figures/fig_05_recommender_offline_metrics.png`
- `report_context/figures/fig_06_risk_diagnosis_metrics.png`
- `report_context/figures/fig_07_ranking_metrics.png`
- `report_context/figures/fig_08_path_quality_metrics.png`
- `report_context/figures/fig_09_pipeline_overview.png`
- `report_context/figures/fig_10_ra_hlpr_flow.png`
- `report_context/figures/figure_manifest.csv`
- `report_context/figures/README.md`

## 11. Test Pass/Fail

| Test group | Command | Result | Ghi chú |
|---|---|---|---|
| Full test suite | `py -3.10 -m pytest -q` | `31 passed in 20.57s` | Test source hiện tại pass trên Python 3.10 |
| Default Python test | `python -m pytest -q` | Fail trước khi collect tests | Thiếu pytest trong Python 3.14 default |
| Visual generation | `py -3.10 report_context\figures\create_verified_figures.py` | Success | Tạo 10 figure từ final artifacts/output JSON |

## 12. Lỗi Hiện Hữu Và Rủi Ro Tái Lập

| Vấn đề | Tác động | Evidence |
|---|---|---|
| Raw datasets không có trong repo | Không kiểm trực tiếp schema/missing/duplicate và không rerun preprocessing từ đầu | `data/raw/.gitkeep`, `.gitignore` |
| Processed splits không có | Không đối chiếu exact locked test indices/row counts bằng file split | `data/processed/final/` trống |
| `models/saved/final` trống | Recommender script và final pipeline không rerun được theo đường dẫn mặc định | Filesystem check, `scripts/run_recommender_pipeline.py` |
| Student final metrics thiếu per-run artifacts | Student rows chỉ partially verified | `reports/final/FINAL_PROJECT_STATUS.md` có số; thiếu matching manifest/CSV/checkpoint |
| `models/final/final_model_manifest.json` có metrics thấp hơn và model strict validation | Không được dùng làm final result hiện tại | `models/final/final_model_manifest.json` vs `reports/final/*` |
| Exact numeric threshold cho `low_f1_tuned` không có trong final manifest cho Student rows | Không ghi threshold cụ thể trong khóa luận | Source threshold tuning có logic, artifact threshold thiếu |
| Baseline final Student thiếu | Không claim deep model vượt baseline trên Student datasets | `final_baseline_comparison.csv` chỉ chứa xAPI |

## 13. Điều Kiện Tái Lập Kết Quả

Để tái lập đúng kết quả final thay vì chỉ đọc artifact:

1. Khôi phục raw CSV đúng version cho `student-mat`, `student-por`, `xAPI`.
2. Khôi phục hoặc tạo lại processed train pool/locked test với cùng seed `42`, test size `0.2`, stratification theo target.
3. Xác định script và commit tạo ra bốn dòng final prediction trong `FINAL_PROJECT_STATUS.md`.
4. Khôi phục checkpoint/metadata exact cho:
   - `student-mat late sequence_cnn_bilstm_only low_f1_tuned`
   - `student-por late sequence_cnn_bilstm_only low_f1_tuned`
   - `student-por midterm sequence_cnn_bilstm_only argmax`
   - `xAPI default gated_fusion_v28 low_f1_tuned`
5. Khôi phục OOF probability hoặc threshold metadata đã dùng để chọn `low_f1_tuned`.
6. Chạy evaluation một lần trên locked test và ghi lại confusion matrix/predictions.
7. Chạy recommender pipeline sau khi đã có prediction probability cho từng dataset.

## 14. Artifact Không Thể Tái Lập Ngay Nếu Thiếu Metadata

| Artifact/Kết quả | Vì sao chưa tái lập ngay | Cách khắc phục |
|---|---|---|
| Student final prediction metrics | Thiếu checkpoint/manifest/per-run CSV exact | Tìm lại artifact từ archive/remote hoặc rerun pipeline xác định đúng script |
| xAPI `gated_fusion_v28` exact checkpoint | Final manifest có metric nhưng workspace không có class/checkpoint tên v28 rõ ràng | Tìm commit/artifact trước cleanup hoặc bổ sung checkpoint metadata |
| Threshold numeric cho `low_f1_tuned` | Source có logic OOF nhưng output threshold final không nằm trong final manifest | Lưu threshold vào manifest hoặc rerun từ OOF predictions |
| Confusion matrix final | Không thấy artifact final tương ứng | Rerun evaluation hoặc tìm predictions locked test |
| User-facing dashboard evidence | Không có lệnh/source/screenshot dashboard verified | Bổ sung screenshot thật và source chạy dashboard |
| Statistical significance | Không có test/CSV final | Thực hiện McNemar/bootstrap/paired test trên predictions nếu có |
