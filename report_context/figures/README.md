# Verified Figure Pack

Các hình trong thư mục này được tạo từ artifact có sẵn trong repository, không dùng dữ liệu giả hoặc ảnh AI.

- `fig_01_prediction_metrics.png`: So sánh Macro F1, Recall Low và F1 Low của các mô hình dự đoán final theo dataset/scenario. Source: `reports/final/FINAL_PROJECT_STATUS.md`.
- `fig_02_xapi_baseline_comparison.png`: So sánh Macro F1 giữa mô hình deep xAPI final và RandomForestClassifier baseline. Source: `reports/final/final_baseline_comparison.csv`.
- `fig_03_low_class_focus.png`: Recall Low và F1 Low cho thấy năng lực phát hiện nhóm sinh viên có nguy cơ kết quả thấp. Source: `reports/final/FINAL_PROJECT_STATUS.md`.
- `fig_04_macro_f1_ranking.png`: Xếp hạng các cấu hình final theo Macro F1. Source: `reports/final/FINAL_PROJECT_STATUS.md`.
- `fig_05_recommender_offline_metrics.png`: Tổng hợp các chỉ số đánh giá offline của RA-HLPR trên xAPI và student-por. Source: `outputs/recommender/xapi/recommender_metrics.json; outputs/recommender/student-por/recommender_metrics.json`.
- `fig_06_risk_diagnosis_metrics.png`: Risk Macro F1 và Risk Micro F1 của đầu chẩn đoán rủi ro. Source: `outputs/recommender/xapi/recommender_metrics.json; outputs/recommender/student-por/recommender_metrics.json`.
- `fig_07_ranking_metrics.png`: Precision@3, Recall@3, NDCG@3 và Coverage@3 của bước xếp hạng can thiệp. Source: `outputs/recommender/xapi/recommender_metrics.json; outputs/recommender/student-por/recommender_metrics.json`.
- `fig_08_path_quality_metrics.png`: Các chỉ số bao phủ rủi ro, tiến triển độ khó và vi phạm prerequisite của lộ trình 4 tuần. Source: `outputs/recommender/xapi/recommender_metrics.json; outputs/recommender/student-por/recommender_metrics.json`.
- `fig_09_pipeline_overview.png`: Pipeline hệ thống từ dữ liệu thô đến dự đoán, chẩn đoán rủi ro, xếp hạng can thiệp và báo cáo. Source: `src/data_pipeline.py; src/models/models.py; src/models_v27.py; scripts/run_recommender_pipeline.py`.
- `fig_10_ra_hlpr_flow.png`: Luồng RA-HLPR dùng xác suất dự đoán và rủi ro quan sát được để tạo lộ trình học tập 4 tuần. Source: `scripts/run_recommender_pipeline.py; src/recommender/risk_rules.py; src/recommender/risk_head.py; src/recommender/candidate_generator.py; src/recommender/hybrid_scorer.py; src/recommender/path_planner.py`.
