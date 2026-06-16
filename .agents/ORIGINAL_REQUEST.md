# Original User Request

## Initial Request — 2026-06-14T05:15:18Z

# Teamwork Project Prompt

Khôi phục và xây dựng lại Mô hình Khuyến nghị bằng Học máy (ML/DL) cùng hệ thống Đánh giá khoa học, nhưng **TUYỆT ĐỐI GIỮ NGUYÊN** logic tiền xử lý dữ liệu và thuật toán Resampling ban đầu của hệ thống.

Working directory: c:\Huflit\kltn
Integrity mode: development

## Requirements

### R1. Xây dựng Mô hình Khuyến nghị bằng ML/DL (PyTorch MLP)
Thay thế `RuleBasedLearningPathEngine` hiện tại bằng một mô hình mạng nơ-ron (Multi-Layer Perceptron) để dự đoán và sinh lộ trình học tập cá nhân hóa. Đội ngũ AI tự do thiết kế kiến trúc mạng phù hợp với dữ liệu (`student-mat`, `student-por`, `xapi`).

### R2. Xây dựng Hệ thống Đánh giá Khoa học (Evaluation Pipeline)
Xây dựng một script riêng biệt (như `src/eval_recommendation.py`) để đánh giá lộ trình khuyến nghị:
1. **Định lượng:** Tính toán các chỉ số xếp hạng (Precision@K, Recall@K, NDCG@K).
2. **Định tính:** Tích hợp bộ chấm điểm tự động LLM-Judge theo tiêu chí sư phạm.
Lưu kết quả đầu ra thành các file JSON vào thư mục `reports/final/recommendations/`.

### R3. Ràng buộc Kỹ thuật Nghiêm ngặt (Constraint)
**TUYỆT ĐỐI KHÔNG ĐƯỢC CHẠM VÀO** hoặc thay đổi bất kỳ dòng code nào liên quan đến kỹ thuật Resampling (ADASYN/SMOTENC), ép kiểu dữ liệu (casting), hoặc Data Preprocessing trong các file `src/data_pipeline.py` hay `src/train_pipeline.py`. Giữ nguyên kết quả gốc vì nó đang mang lại hiệu năng cao nhất.

## Acceptance Criteria

### Tính chính xác và Logic
- [ ] Codebase sử dụng mạng PyTorch MLP cho phần Learning Path thay vì rule-based if-else.
- [ ] Logic của thuật toán ADASYN/SMOTE vẫn giữ nguyên bản như cũ, không có thêm các bước ép kiểu hay cắt xén dữ liệu.

### Đánh giá và Kiểm định
- [ ] Có script `src/eval_recommendation.py` chạy độc lập sinh ra file JSON chứa NDCG, Precision, và điểm LLM-Judge.
- [ ] Toàn bộ hệ thống chạy end-to-end không phát sinh lỗi.

## Follow-up — 2026-06-14T08:42:17Z

# Teamwork Project Prompt — Report Update

Cập nhật mã nguồn sinh Báo cáo Khóa luận (`generate_doc.py`) để tạo ra bản báo cáo mới nhất, phản ánh chính xác 100% mô hình Khuyến nghị ML/DL mới vừa được xây dựng, đồng thời xóa bỏ hoàn toàn các nhắc nhớ về tập luật cũ. 

**Lưu ý cực kỳ quan trọng:** TUYỆT ĐỐI Không nhắc đến việc sửa lỗi thuật toán Resampling vì người dùng đã yêu cầu giữ nguyên phương pháp Resampling gốc (ADASYN/SMOTE bị lỗi ép kiểu) do nó đem lại F1 tốt hơn.

Working directory: c:\Huflit\kltn
Integrity mode: development

## Requirements

### R1. Cập nhật Nội dung Báo cáo (Chương 3)
Cập nhật nội dung trong file Python sinh báo cáo để phản ánh kiến trúc hệ thống hiện tại:
- **Mô hình Khuyến nghị:** Xóa bỏ hoàn toàn mô tả về tập luật if-else (Rule-based). Viết lại chi tiết kiến trúc mạng nơ-ron đa tầng (PyTorch MLP) hiện đại đã được dùng để dự đoán và sinh lộ trình học tập.

### R2. Cập nhật Kết quả Đánh giá (Chương 4 & 5)
- Bổ sung bảng biểu/kết quả của hệ thống Đánh giá Mô hình Khuyến nghị dựa trên các file báo cáo JSON (bao gồm các chỉ số khoa học: Precision@K, Recall@K, NDCG@K và đánh giá LLM-Judge) để giải quyết triệt để nhận xét của giảng viên. Thư mục chứa JSON là `reports/final/recommendations`.
- Cập nhật mục "Hạn chế của đề tài" thành "Những vấn đề đã khắc phục" (Giải thích việc chuyển từ Rule-based sang ML/DL để tăng tính cá nhân hóa) và định hướng "Hướng phát triển tương lai".

### R3. Hoàn thiện Format Báo cáo
Đảm bảo script sinh ra file Word (.docx) chuẩn chỉnh theo format hiện tại, văn phong khoa học, logic chặt chẽ, đáp ứng toàn bộ các góp ý trong file PDF nhận xét. Hệ thống cần chạy thử file script để đảm bảo ra được thành phẩm `Bao_cao_cuoi_cung.docx` (bạn hãy sửa code để output ra file này thay vì Bao_cao_tien_do.docx).

## Acceptance Criteria

### Tính chính xác của nội dung
- [ ] File Word được sinh ra KHÔNG còn bất kỳ chữ "Rule-based" nào cho phần Learning Path.
- [ ] File Word CÓ mục trình bày lý thuyết và kiến trúc mạng PyTorch MLP cho hệ thống khuyến nghị.
- [ ] File Word CÓ mục kết quả đánh giá bằng chỉ số Ranking (NDCG, Precision) và LLM-Judge rõ ràng, đầy đủ cơ sở khoa học.

### Khả năng thực thi
- [ ] Script `generate_doc.py` chạy thành công hoàn toàn, tự động load các thông số từ file JSON kết quả để điền vào Word.
- [ ] Thành phẩm cuối cùng là một bản Word `Bao_cao_cuoi_cung.docx` hoàn chỉnh, đọc liền mạch và mang đậm chất báo cáo học thuật.

## Follow-up — 2026-06-14T16:59:56Z

Xây dựng hệ thống RA-HLPR (Risk-Aware Hybrid Learning Path Recommender) lai dựa trên kết quả của mô hình CNN+BiLSTM dự đoán học tập. Hệ thống sử dụng MLP hiện tại như một Risk Diagnosis Head và kết hợp với Intervention Knowledge Base để sinh lộ trình học tập 4 tuần cá nhân hóa cho sinh viên. Mô hình khuyến nghị phải hoạt động như một module downstream độc lập.

Working directory: c:\Huflit\kltn
Integrity mode: development

## Requirements

### R1. Tích hợp Downstream & Risk Diagnosis Head
- **Downstream Predictor:** KHÔNG bắt buộc sửa hoặc fine-tune mô hình CNN-BiLSTM hiện tại. Module khuyến nghị chỉ nhận đầu vào từ file prediction CSV hoặc output inference có sẵn (gồm `predicted_label`, `class_probabilities`, `confidence` và các đặc trưng gốc). Chỉ trích xuất `student_embedding` nếu việc lấy embedding KHÔNG làm thay đổi checkpoint, KHÔNG làm thay đổi metric locked test và KHÔNG phá vỡ pipeline hiện có.
- **Risk Diagnosis:** Refactor mô hình MLP multi-label hiện tại thành `RiskDiagnosisHead`. Đầu vào nhận `student_features` và `class_probabilities`. Đầu ra dự đoán 6 rủi ro học tập (R1->R6). Sử dụng `BCEWithLogitsLoss + pos_weight` và sinh weak labels bằng rule minh bạch. Viết toàn bộ rule tạo weak label vào file riêng có comment giải thích rõ ràng.

### R2. Xây dựng Intervention Knowledge Base & Hybrid Scorer
- **Knowledge Base:** Tạo thư mục và dữ liệu catalog (`intervention_catalog.csv` và `risk_intervention_mapping.csv`) với tối thiểu các cột: item_id, intervention_name, description, target_risks, difficulty_level, estimated_hours_per_week, recommended_phase, expected_effect, prerequisite_level.
- **Hybrid Scorer:** Tính điểm can thiệp dựa trên công thức trọng số: risk_match (0.3), performance_need (0.2), difficulty_fit (0.15), time_fit (0.15), prerequisite_fit (0.1), expected_effect (0.1). Trả về danh sách top K can thiệp kèm score và explanation. (Tùy chọn: Xây dựng thêm `LearningPathRanker` bằng Neural Network nếu khả thi).

### R3. Learning Path Planner & Evaluation
- **Path Planner:** Module `path_planner.py` phân bổ các top can thiệp thành lộ trình 4 tuần: Week 1 (Stabilize), Week 2 (Practice), Week 3 (Reinforce), Week 4 (Evaluate & Adjust). Yêu cầu mỗi tuần phải có: objective, recommended_actions, expected_outcome, và explanation.
- **Evaluation:** Đánh giá Risk Diagnosis (Micro/Macro F1, Precision, Recall, Hamming Loss), Ranking (Precision@K, Recall@K, NDCG@K, Coverage), và Path Quality (Risk Coverage Rate, Workload Balance, Difficulty Progression, Prerequisite Violation). Không ghi LLM-Judge nếu chưa chạy.

### R4. Scripts & Output Files
- Tái cấu trúc thư mục logic (`src/models`, `src/recommender`, `src/evaluation`, `scripts`).
- Tạo file chạy `scripts/run_recommender_pipeline.py --dataset <dataset>` thực thi end-to-end từ load data dự đoán, weak labeling, train Risk Head, sinh lộ trình, đến đánh giá.
- Sinh các file kết quả vào `outputs/recommender/`: `risk_predictions.csv`, `recommendation_results.csv`, `learning_paths.json`, `recommender_metrics.json`, và `recommender_report.md` (chứa báo cáo phân tích và ví dụ 3 sinh viên). 

## Acceptance Criteria

### Tính toàn vẹn kiến trúc
- [ ] Tuyệt đối KHÔNG thay đổi phá vỡ pipeline CNN + BiLSTM có sẵn, cũng như không làm sai lệch metrics test.
- [ ] Khuyến nghị hoạt động hoàn toàn như một hệ thống Downstream.
- [ ] MLP hiện tại không còn được gọi là hệ thống recommendation chính, mà chỉ đóng vai trò Risk Diagnosis Head.
- [ ] Catalog can thiệp phải là dữ liệu học tập/hành động hợp lý, không được "bịa" quá xa bối cảnh giáo dục.

### Tính thực thi và Đánh giá
- [ ] Lệnh `python scripts/run_recommender_pipeline.py --dataset student-mat` chạy thành công hoàn toàn không gặp lỗi.
- [ ] File `learning_paths.json` sinh ra lộ trình 4 tuần rõ ràng, có diễn giải (explanation) tại sao tài nguyên đó được chọn.
- [ ] Báo cáo `recommender_report.md` chứa bảng metrics hoàn toàn dựa trên số liệu đánh giá thực tế của script, không dùng thông số bịa đặt.
- [ ] Cung cấp đoạn văn giải thích hệ thống "Risk-Aware Hybrid Learning Path Recommender" để chuẩn bị đưa vào báo cáo khóa luận.

## Follow-up — 2026-06-15T10:09:15+07:00

# Teamwork Project Prompt: RA-HLPR Refactoring
Nhiệm vụ: Sửa PHẦN MÔ HÌNH KHUYẾN NGHỊ, triển khai RA-HLPR (Risk-Aware Hybrid Learning Path Recommender) như một downstream module.

QUY TẮC BẮT BUỘC:
1. Không phá mô hình dự đoán CNN-BiLSTM + Context MLP hiện có.
2. Không train lại hoặc sửa classifier chính nếu không cần.
3. RA-HLPR phải là downstream module, nhận input từ output dự đoán hiện có.
4. Không được bịa metric. Không được ghi bảng đánh giá cho dataset chưa chạy.
5. Không được gọi là collaborative filtering nếu không có dữ liệu user-item interaction.
6. Không được gọi là knowledge graph nếu chưa xây graph thật.
7. Không dùng các risk không có feature trong dataset.

PHASE 1: SỬA MÔ HÌNH KHUYẾN NGHỊ TRONG CODE
- Tạo `src/recommender/risk_rules.py`: Định nghĩa 6 risk (R1_LOW_PRIOR_PERFORMANCE, R2_DECLINING_TREND, R3_ATTENDANCE_RISK, R4_LOW_ENGAGEMENT, R5_INSUFFICIENT_STUDY_TIME, R6_HIGH_FAILURE_PROBABILITY). Không dùng risk không có feature.
- Tạo `data/recommender/intervention_catalog.csv`: Ít nhất 10 items có các cột (item_id, intervention_name, description, target_risks, difficulty_level, estimated_hours_per_week, recommended_phase, expected_effect, prerequisite_level).
- Tạo `src/recommender/hybrid_scorer.py`: Hàm score = 0.3*risk_match + 0.2*performance_need + 0.15*difficulty_fit + 0.15*time_fit + 0.1*prerequisite_fit + 0.1*expected_effect.
- Tạo `src/recommender/candidate_generator.py`.
- Tạo `src/recommender/path_planner.py`: Lộ trình 4 tuần (Stabilize, Practice, Reinforce, Evaluate & Adjust).
- Tạo `src/recommender/explanation.py`.
- Tạo `src/evaluation/recommender_metrics.py` & `src/evaluation/path_quality.py`.
- Cập nhật `scripts/run_recommender_pipeline.py`: Chạy end-to-end, lưu output vào `outputs/recommender/{dataset}/`.

PHASE 2: CHỈNH BÁO CÁO SAU KHI MODEL CHẠY XONG
- Chỉnh sửa `generate_doc.py` để phản ánh kiến trúc mới (3.5.1 đến 3.5.5) và bảng kết quả (4.4). Bắt buộc thêm các câu giải thích về weak labels và các hạn chế.
- Tạo file `outputs/recommender/final_recommender_section.md` chứa nội dung báo cáo để người dùng có thể tham khảo.

## Follow-up — 2026-06-15T08:04:55Z

# Teamwork Project Prompt — Draft

> Status: Ready for launch — awaiting user approval.
> Goal: Craft prompt → get user approval → delegate to teamwork_preview

Cải tiến mô hình dự đoán kết quả học tập CNN-BiLSTM + Context MLP trên 3 dataset (student-mat, student-por, xapi) để tối ưu hóa Macro-F1 và Recall nhóm Low, tuân thủ nghiêm ngặt đề cương khóa luận và kiểm soát rủi ro data leakage.

Working directory: c:\Huflit\kltn
Integrity mode: development

## Requirements

### R1. Phase 1-2: Audit & Fix Resampling
- Kiểm tra toàn bộ pipeline hiện tại để đảm bảo không có rò rỉ dữ liệu (locked test phải được cách ly hoàn toàn).
- Sửa lỗi Resampling cho dữ liệu hỗn hợp: Ưu tiên SMOTENC, nếu dùng ADASYN chỉ áp dụng cho phần numeric-safe. Không bao giờ áp dụng trên validation hoặc locked test. Tạo bảng so sánh các phương pháp (outputs/experiments/resampling_comparison.csv).

### R2. Phase 3-5: Kiến trúc V27 & Tối ưu hóa
- Triển khai mô hình `StudentHybridV27` (src/models_v27.py) với Sequence Branch (Conv1D + BiLSTM) và Context Branch (Categorical Embeddings + Context MLP), kết hợp bằng Gated Fusion.
- Thêm Auxiliary Heads: Ordinal cho thứ tự lớp học, Regression (chỉ với dataset có numeric grade).
- Thử nghiệm các Loss function (src/losses_v27.py) như Weighted CE, Focal Loss, CB-Focal, Ordinal loss. (outputs/experiments/loss_comparison.csv)
- Tối ưu hóa đặc thù theo từng dataset (Student-mat cần regularization, Student-por cần cân bằng lớp Medium, xAPI cần tương tác feature học tập).

### R3. Phase 6-8: Optuna & Ensemble
- Chạy Optuna tuning (scripts/run_v27_optuna.py) cho từng dataset (50-150 trials tùy tài nguyên). Tuyệt đối không tune trên locked test.
- Tune decision thresholds trên validation set sau khi train (outputs/experiments/thresholds_{dataset}.json).
- Tạo Seed Ensemble (42-46) để trung bình xác suất (outputs/v27/{dataset}/ensemble_metrics.json).

### R4. Phase 9-10: Ablation & Evaluation
- Thực hiện Ablation study bắt buộc với 10 biến thể (outputs/v27/ablation_results.csv).
- Đánh giá chuẩn các metrics: Accuracy, Precision/Recall macro, F1-Macro, F1 từng class, Recall Low, RMSE, R² (outputs/v27/{dataset}/metrics.json).

### R5. Phase 12: Báo cáo
- CHỈ tạo báo cáo (outputs/v27/final_prediction_section.md) SAU KHI có kết quả thật từ Artifacts. Không ghi đè báo cáo Word ngay. Trả về bảng đối chiếu đầy đủ.

## Acceptance Criteria

- [ ] Giữ nguyên mô hình chính là CNN-BiLSTM + Context MLP (không chuyển qua ML truyền thống).
- [ ] Không có data leakage (kiểm tra locked test isolation và feature target).
- [ ] Pipeline chạy thành công xuất ra đầy đủ metrics, ablation results và predictions.
- [ ] **Mục tiêu hiệu năng:** 
      - Macro-F1 trên locked test tăng so với baseline cũ (Mat: 0.8690, Por: 0.8156, xAPI: 0.7850) ít nhất `0.01`, HOẶC
      - Recall Low tăng đáng kể mà Macro-F1 không giảm quá `0.01`, HOẶC
      - Dataset xAPI cải thiện rõ ràng so với baseline `0.7850`.
- [ ] Nếu mô hình V27 không vượt qua baseline, giữ nguyên baseline cũ và báo cáo V27 là thí nghiệm mở rộng. Tuyệt đối không bịa số liệu.
