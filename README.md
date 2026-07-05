# KLTN - Du doan thanh tich hoc tap sinh vien bang CNN-BiLSTM

Do an xay dung pipeline du doan thanh tich hoc tap sinh vien theo 3 lop
`Low`, `Medium`, `High` tu bo du lieu Student Performance `student-mat`.
Pipeline final doc du lieu theo huong database-first tu PostgreSQL, huan luyen
mo hinh CNN-BiLSTM va sinh khuyen nghi hoc tap rule-based co giai thich.

## Final scope

- Dataset final: `student-mat`
- Dataset version: `1`
- Source records: `395`
- Split final: `train = 316`, `test = 79`
- Target: 3 lop suy ra tu `G3`
- Input model: chi `G1`, `G2`
- Final model: `cnn_bilstm_classifier`
- Khong co Context MLP trong kien truc final
- Recommendation policy: `student_mat_rule_policy_v2`
- PostgreSQL database: `student_predict`
- Runtime application role: `student_predict_app`

## Final model

Kien truc final:

```text
[G1, G2]
-> Conv1D + BatchNorm + ReLU
-> Bi-LSTM
-> Dropout
-> Linear classification head
-> 3-class probabilities
```

`G1` va `G2` la hai moc danh gia truoc cua sinh vien. `G3` chi dung de tao
nhan muc tieu va khong duoc dua vao input model. Cac metadata lineage nhu
`__source_row_number`, `record_id`, `dataset_version_id`, `run_id` cung khong
duoc dua vao feature matrix.

## Final run verified

- Run ID: `4ac7c8a9-8f20-4abd-ac80-e414f8dd3eaf`
- Git commit recorded by run:
  `99aa80cf157ce709c9aafd4e44702b0c8ab79dd4`
- Status: `completed`
- Test predictions: `79`
- Recommendations v2: `79`

Final holdout confirmation metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.9113924050632911 |
| F1-Macro | 0.9256122872561229 |
| Precision-Macro | 0.9234811165845649 |
| Recall-Macro | 0.9304993252361674 |
| RMSE | 0.2976702788937936 |
| R2 | 0.8226427196921103 |

`RMSE` va `R2` chi la diagnostic tren nhan thu bac `0/1/2`, khong phai
regression output chinh thuc.

## PostgreSQL source/ML lineage

Schema lineage final gom:

```text
source_dataset_versions
source_records
ml_experiment_runs
ml_run_record_splits
ml_predictions
ml_run_metrics
ml_recommendations
```

CSV chi dung de seed/import dataset lan dau. Normal training path doc
`source_records` tu PostgreSQL, khong fallback am tham ve CSV.

Lenh seed dataset khi moi khoi tao DB:

```powershell
python scripts/ingest_dataset_to_postgres.py --dataset student-mat
```

Lenh chay pipeline DB-first:

```powershell
python scripts/run_pipeline.py --dataset student-mat --target-mode 3class --dataset-version-id 1 --n-trials 50
```

## Recommendation

Recommendation final la policy rule-based version
`student_mat_rule_policy_v2`. Policy nay dung predicted label,
confidence/probability va cac feature quan sat duoc de sinh learning path.

Khong duoc claim recommender da chung minh lam sinh vien hoc tot hon ngoai
thuc te. Hieu qua thuc te can human review, feedback nguoi dung hoac du lieu
intervention theo thoi gian.

## Bao cao final

Chi giu ban bao cao hien tai:

```text
reports/final/KLTN_CNN_BiLSTM_Du_doan_Thanh_tich_Hoc_tap_Sinh_vien_FINAL_REVIEWED.docx
```

Cac bao cao cu, diagnostics va generated report artifacts cu da duoc don de
tranh nham lan voi final result.

## Tests

Chay test:

```powershell
python -m pytest -q
```

Trong moi truong Codex da xac minh bang bundled Python:

```text
79 passed, 5 skipped
```

## Guardrails

- Khong goi final model la `cnn_bilstm_mlp`.
- Khong mo ta final architecture la CNN-BiLSTM + Context MLP.
- Khong dung historical run `748dfafb-acac-4565-b96c-3093b2abb37a` lam final
  result.
- Khong dua `student-por` hoac `xAPI` vao ket qua final chinh.
- Khong noi `G1/G2` la nhieu hoc ky; chi mo ta la hai moc danh gia truoc.
- Khong dung `G3`, true label hoac metadata DB de sinh operational
  recommendation.
- Khong claim causal improvement cho recommender khi chua co du lieu can thiep.
- Khong sua/xoa legacy tables `paper_*`, `students`, `student_grades`.
