## 2026-06-15T10:12:21+07:00
You are a Worker agent. Your working directory is `c:\Huflit\kltn\.agents\teamwork_preview_worker_implementation_1\`.
The project working directory is `c:\Huflit\kltn`.

Your task is to implement the RA-HLPR Refactoring (Phase 1 & Phase 2) as follows:

PHASE 1: SỬA MÔ HÌNH KHUYẾN NGHỊ TRONG CODE
1. Tạo `src/recommender/risk_rules.py`:
   - Định nghĩa 6 risk:
     - `R1_LOW_PRIOR_PERFORMANCE`
     - `R2_DECLINING_TREND`
     - `R3_ATTENDANCE_RISK`
     - `R4_LOW_ENGAGEMENT`
     - `R5_INSUFFICIENT_STUDY_TIME`
     - `R6_HIGH_FAILURE_PROBABILITY`
   - Cung cấp hàm `generate_weak_labels(df: pd.DataFrame, dataset_kind: str) -> np.ndarray`.
   - Quy tắc "Không dùng risk không có feature":
     - Đối với dữ liệu `student` (mat/por), tất cả 6 risk đều có đặc trưng (failures, G1/G2, absences, freetime/goout/activities, studytime, failures/predicted_class). Hãy viết luật gán nhãn yếu rõ ràng.
     - Đối với dữ liệu `xapi`, các risk R1, R2, R5 KHÔNG có đặc trưng (không có failures, không có grade lịch sử, không có studytime). Do đó chỉ sinh nhãn cho 3 risk có đặc trưng: R3_ATTENDANCE_RISK (StudentAbsenceDays == 'Above-7'), R4_LOW_ENGAGEMENT (VisITedResources/raisedhands/Discussion/AnnouncementsView), và R6_HIGH_FAILURE_PROBABILITY (Class == 'L' hoặc tương đương). Trả về mảng shape (N, 3) cho xapi và (N, 6) cho student.
2. Sửa `src/recommender/risk_head.py`:
   - Cập nhật hàm `train_risk_head` để khởi tạo `RiskDiagnosisHead` với `output_dim=targets.shape[1]` thay vị mặc định 6. Điều này cho phép tự động điều chỉnh số lượng đầu ra rủi ro dự đoán theo dataset (6 cho student, 3 cho xapi).
3. Tạo thư mục `data/recommender/` và file `data/recommender/intervention_catalog.csv` chứa ít nhất 10 interventions. Các cột gồm: `item_id`, `intervention_name`, `description`, `target_risks`, `difficulty_level`, `estimated_hours_per_week`, `recommended_phase`, `expected_effect`, `prerequisite_level`. Đảm bảo các `target_risks` khớp với tên 6 risk mới.
4. Cập nhật `src/recommender/hybrid_scorer.py`:
   - Điểm số tính theo công thức: score = 0.3*risk_match + 0.2*performance_need + 0.15*difficulty_fit + 0.15*time_fit + 0.1*prerequisite_fit + 0.1*expected_effect.
   - Trích xuất `target_risks` từ catalog khớp với các risk codes tương ứng của dataset hiện tại.
5. Tạo `src/recommender/candidate_generator.py`:
   - Lọc các catalog items phù hợp dựa trên rủi ro được chẩn đoán (ví dụ rủi ro >= 0.3) và lớp học lực dự đoán trước khi chuyển qua Scorer.
6. Sửa/Tạo `src/recommender/path_planner.py`:
   - Lập lộ trình 4 tuần (Week 1: Stabilize, Week 2: Practice, Week 3: Reinforce, Week 4: Evaluate & Adjust) phân bổ các can thiệp tốt nhất.
7. Tạo `src/recommender/explanation.py`:
   - Sinh diễn giải (explanation) thân thiện giải thích lý do lựa chọn can thiệp đó cho học sinh.
8. Tạo `src/evaluation/recommender_metrics.py` và `src/evaluation/path_quality.py`:
   - Tách các hàm tính toán metric đánh giá Risk Diagnosis & Ranking sang `recommender_metrics.py`.
   - Tách hàm đánh giá chất lượng lộ trình sang `path_quality.py`.
   - Đảm bảo `src/evaluation/recommender_eval.py` vẫn tương thích (ví dụ import từ hai file mới).
9. Cập nhật `scripts/run_recommender_pipeline.py`:
   - Chạy end-to-end, lưu tất cả kết quả đầu ra vào `outputs/recommender/{dataset}/`.
   - Các file kết quả: `risk_predictions.csv`, `recommendation_results.csv`, `learning_paths.json`, `recommender_metrics.json`, `recommender_report.md`.
   - Đảm bảo script hỗ trợ cả 3 dataset (`student-mat`, `student-por`, `xapi`) chạy thành công.

PHASE 2: CHỈNH BÁO CÁO SAU KHI MODEL CHẠY XONG
1. Chỉnh sửa `generate_doc.py`:
   - Cập nhật mục 3.5 thành các mục chi tiết từ 3.5.1 đến 3.5.5 tương ứng với thiết kế hệ thống RA-HLPR mới (Đầu chẩn đoán, Cơ sở tri thức, Gán nhãn yếu, Bộ chấm điểm hỗn hợp, Bộ lập lộ trình).
   - Thêm phần giải thích về weak labels và các hạn chế của phương pháp đánh giá (thiếu kiểm chứng thực nghiệm thực tế/longitudinal validation).
   - Cập nhật đường dẫn load dữ liệu bảng 4.1 và 4.2 từ `outputs/recommender/{dataset}/recommender_metrics.json`.
   - Xuất ra file `Bao_cao_cuoi_cung.docx` và đảm bảo script chạy thành công tạo ra file này.
2. Tạo file `outputs/recommender/final_recommender_section.md` chứa phần nội dung tiếng Việt của mục 3.5 và các bảng kết quả để tham khảo.

QUY TẮC BẮT BUỘC:
1. Không phá mô hình dự đoán CNN-BiLSTM + Context MLP hiện có.
2. Không train lại hoặc sửa classifier chính nếu không cần.
3. RA-HLPR phải là downstream module, nhận input từ output dự đoán hiện có.
4. Không được bịa metric. Không được ghi bảng đánh giá cho dataset chưa chạy.
5. Không được gọi là collaborative filtering nếu không có dữ liệu user-item interaction.
6. Không được gọi là knowledge graph nếu chưa xây graph thật.
7. Không dùng các risk không có feature trong dataset.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Sau khi hoàn thành, hãy chạy toàn bộ pipeline cho cả 3 dataset, chạy generate_doc.py để sinh báo cáo, chạy bộ kiểm thử pytest (nếu có), xác nhận mọi thứ hoạt động hoàn hảo và ghi kết quả chạy vào handoff.md.
