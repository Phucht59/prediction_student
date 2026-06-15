# Original User Request

## 2026-06-15T03:09:48Z

You are the Project Orchestrator (type: teamwork_preview_orchestrator).
Your working directory is: c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_ra_hlpr_refactor\
The project working directory is: c:\Huflit\kltn

Your task is to orchestrate and execute the RA-HLPR Refactoring task described in c:\Huflit\kltn\.agents\ORIGINAL_REQUEST.md.

Specifically:
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

QUY TẮC BẮT BUỘC:
1. Không phá mô hình dự đoán CNN-BiLSTM + Context MLP hiện có.
2. Không train lại hoặc sửa classifier chính nếu không cần.
3. RA-HLPR phải là downstream module, nhận input từ output dự đoán hiện có.
4. Không được bịa metric. Không được ghi bảng đánh giá cho dataset chưa chạy.
5. Không được gọi là collaborative filtering nếu không có dữ liệu user-item interaction.
6. Không được gọi là knowledge graph nếu chưa xây graph thật.
7. Không dùng các risk không có feature trong dataset.

Please initialize your plan.md, progress.md, and context.md in your working directory. Delegate tasks (e.g., exploration, worker implementation, review, etc.) using your subagents. Keep a close eye on the requirements and constraints. Keep progress.md updated. When done, write handoff.md and send me a completion message.
