# 00 — Source and scope audit

Cập nhật lúc khởi động chương trình `hybrid_superiority_v2`.

## Git

| Hạng mục | Giá trị |
|---|---|
| Repo | `C:\hufit\student` (GitHub `Phucht59/prediction_student`) |
| Branch đầu vào | `hybrid` @ `0cb02479154a734240b55bf5525a96e11a72e863` |
| Branch làm việc | `research/hybrid-superiority-v2` (không reset, không xóa lịch sử) |
| Working tree lúc fork | sạch |
| Diff vs commit đã kiểm toán | không (HEAD trùng) |

## Claim → code → artifact → status

| Claim công khai | Code | Artifact | Status |
|---|---|---|---|
| Hybrid CNN–BiLSTM là authority | `src/prediction/model/hybrid.py` C0 parallel softmax | `reports/CURRENT_REPORTS.md` | **Không promote** cho đến confirmation |
| Robust inner 3×3 AP (repo gọi PR-AUC) | Phase 4 trainer | `FINAL_PREDICTION_MODEL_REPORT.md` | Provenance; **không** dùng để chọn model mới |
| UCI S1/S2 Hybrid > SVM/RF | bảng report | `uci_final.csv` | Point estimate; S2 sát RF (+0.0029) |
| OULAD Hybrid > LR/RF | bảng report | `oulad_final.csv` | Lợi thế 0.003–0.008; XGB đã bị loại roster |
| XGBoost không active | `configs/prediction/registry.json` | historical Phase 8 tables | **Đưa XGB/CatBoost trở lại** trong protocol mới |
| Phase 4 gate | `NOT_READY_FOR_FINAL_EVAL` | FINAL report §K | Owner chọn authority; **không** phải gate pass |
| Recommendation V3 NDCG@3 0.88785 | weak Gemini consensus | V3 report | **Không** phải expert validation; exact-best Top-1 ~0.407 |
| Pipeline raw-to-feature | `src/prediction/data/oulad_features.py` | `data/raw/*` có trong workspace | Tái sử dụng; `experiments/hybrid_vnext` vẫn hardcode `C:\hufit\kltn` — **không dùng** cho v2 |
| Optuna/Postgres serving | `student_db` schemas raw/catalog/prediction/recommendation | PostgreSQL 18.4 :5432 | Schema `research` / `optuna_hs_v2` **mới**, không drop serving |

## Tài liệu khóa luận

| Tệp | Tìm thấy | Ghi chú |
|---|---|---|
| `07 - DE-CUONG-KLTN-PTKQHT.pdf` | `C:\hufit\07 - DE-CUONG-KLTN-PTKQHT.pdf` (basename không có `(1)`) | CNN+BiLSTM, SMOTE/ADASYN, PostgreSQL, Optuna, Dropout/ES, khuyến nghị; metric có R²/RMSE — xem SCOPE_CLARIFICATION |
| Bài báo CNN–BiLSTM | `C:\hufit\A_hybrid_framework_for_improving_students__performance_prediction_based_on_CNN_with_Bi_LSTM___Format_Final.pdf` | Pearson toàn cục, G1/G2, Accuracy ~100% Math / 92.31% POR / 84.38% xAPI. **Không** phải benchmark trực tiếp |
| `Đã dán markdown (1).md` | **MISSING** | Không có trong workspace/Downloads |

## Bài báo tham chiếu — giá trị và giới hạn

Động cơ kiến trúc: CNN 1D + BiLSTM. Không sao chép Pearson trên toàn data, không lấy Accuracy 100% làm mục tiêu, không coi xAPI là miền hiện tại.

UCI G1/G2 tương quan mạnh G3 (Cortez & Silva) — đó là lý do S0 yếu và S2 mạnh; không được nhồi G3.

OULAD (Kuzilek et al., Scientific Data 2017, DOI 10.1038/sdata.2017.171): enrollment + VLE; 100% không phải early warning.

Tabular trees mạnh trên dữ liệu bảng cỡ vừa (Grinsztajn et al., NeurIPS 2022). Selection overfit (Cawley & Talbot, JMLR 2010). AP trên imbalance (Saito & Rehmsmeier 2015).

## Root cause đã biết (không dùng outer cũ để chọn model)

1. UCI T=0/1/2 — CNN/BiLSTM ít inductive advantage.
2. ~513k params vs ~440 FIT records/fold.
3. Topology song song; CNN yếu OULAD.
4. 13 aggregate mạnh + tree phù hợp tabular.
5. Objective macro không ép worst warm-stage.
6. Risk = Fail|Withdrawn trộn hai cơ chế; length ≈ Withdrawn ở 100%.
7. Cohort OULAD đổi theo cutoff (risk-set), không phải cùng sinh viên thêm dữ liệu.
8. SMOTE/ADASYN trên tensor hỗn hợp làm AP giảm.
9. vNext đã có C0–C3, AMP, availability sửa — tái sử dụng ý, **không** phụ thuộc kltn.

## Hardcoded path

`experiments/hybrid_vnext/protocol.py` `KLTN = Path(r"C:\hufit\kltn")`. Namespace mới dùng `DATA_ROOT` / `ARTIFACT_ROOT`. `kltn` tồn tại trên máy nhưng **không** là dependency của pipeline active.

## Raw data

Có đủ CSV trong `data/raw`, gồm `studentVle.csv` 453 836 331 bytes. Checksum SHA-256 khóa trong `protocol.py` `RAW_SHA256`.

## Serving

Không sửa `src/prediction/model/hybrid.py` public C0 cho đến khi confirmation pass. Nghiên cứu nằm trong `experiments/hybrid_superiority_v2`.
