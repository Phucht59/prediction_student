# Final project cleanup

## Status

`FINAL_UNIFIED_RELEASE_CANDIDATE`

Cleanup này chỉ tổ chức lại code/evidence. Không model nào được train lại,
không threshold/split/label nào thay đổi và không truy cập Future OULAD.

## Public model authority

| Model ID | Display name | Dataset | Macro-F1 |
|---|---|---|---:|
| `cnn_bilstm_mat` | CNN-BiLSTM MAT | Student-Mat | 0.9014601961315334 |
| `cnn_bilstm_por` | CNN-BiLSTM POR | Student-Por | 0.8622587167738002 |
| `cnn_bilstm_oulad` | CNN-BiLSTM OULAD | OULAD | 0.8280835945631038 |

Stage-aware UCI và OULAD dùng một estimator/checkpoint cho mọi stage thuộc
cùng dataset/fold/seed. Không tạo model identity hybrid song song theo stage.

## Refactor

| Trước cleanup | Sau cleanup | Lý do |
|---|---|---|
| `src/studies/unified_stage.py` | `src/pipelines/uci.py` | Pipeline UCI là production pipeline, không phải study |
| `src/studies/oulad_multistage.py` | `src/pipelines/oulad.py` | Pipeline OULAD là production pipeline |
| teacher-feedback monolith | `src/pipelines/uci_support.py` + local `test_lab` | Chỉ giữ data/preprocessing code thực sự cần |
| versioned final config names | `configs/final/uci_prediction.yaml`, `configs/final/oulad_prediction.yaml` | Tên public không mang research version |
| `artifacts/refactor/` | `artifacts/final/database/` hoặc `protocol_snapshots/` | Evidence cuối có một authority rõ |
| `reports/refactor/` | report cần thiết trong `reports/final/` | Không để refactor namespace ở public root |
| legacy UCI fold cache | local `test_lab` | Đã được supersede bởi unified checkpoint evidence |

Hai module rất nhỏ còn lại trong `src/studies/` chỉ là compatibility shim cho
import path được nhúng trong checkpoint joblib bất biến. Xóa chúng sẽ làm hỏng
replay; chúng không chứa implementation song song.

## Local archive

265 file (264,487,521 bytes) lịch sử/code/runtime đã được lưu tại:

`test_lab/archived_experiments/final_cleanup_20260730/`

`test_lab/` bị Git ignore. Manifest provenance và checksum cần cho bảo vệ kết
quả vẫn được giữ tại `artifacts/final/provenance/`; public validation không phụ
thuộc local archive.

## Kept evidence

- official and stage-aware checkpoints;
- record-aligned predictions and seed predictions;
- split/training-run/checkpoint-stage manifests;
- per-class, calibration, bootstrap, top-k and ablation evidence;
- ML comparator and MLP evidence;
- recommendation technical validation;
- PostgreSQL reconciliation, backup and cutover evidence;
- protocol snapshots and provenance checksums.

## CLI decision

`project.py` được giữ và thu gọn. Đây là entry point cho:

- `final status/report/validate`;
- explicit UCI/OULAD pipeline validation;
- database final operations.

Xóa file này sẽ làm hỏng reproducibility và test contract. Các command study cũ
đã bị loại.

## Scientific freeze

- Canonical metrics changed: **NO**
- Official model retrained: **NO**
- New Optuna search: **NO**
- Outer data used for tuning: **NO**
- Best seed selected: **NO**
- Future OULAD accessed: **NO**
- Recommendation counts changed: **NO**
- DOCX/PDF modified: **NO**
