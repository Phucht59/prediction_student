# 04. Unknown Or Missing Information

Các mục dưới đây là thông tin không được phép đoán khi viết khóa luận. Nếu cần đưa vào báo cáo, phải bổ sung evidence trước hoặc viết rõ là hạn chế.

| Thiếu gì | Ảnh hưởng chương nào | Vì sao chưa đủ dữ kiện | Cách lấy nhanh nhất |
|---|---|---|---|
| Raw dataset files `student-mat.csv`, `student-por.csv`, `xAPI-Edu-Data.csv` | Chương 2, Chương 3, Chương 4 | `data/raw/` chỉ có `.gitkeep`; không kiểm trực tiếp schema, missing values, duplicate, exact class distribution | Khôi phục raw CSV đúng version vào `data/raw/`, chạy script thống kê nhẹ và lưu output |
| Processed train/validation/locked-test splits | Chương 3, Chương 4 | `data/processed/final/` trống; không đối chiếu exact split rows/indices | Chạy preprocessing từ raw hoặc khôi phục `data/processed/final/*` đã dùng lúc đánh giá |
| Exact final checkpoints cho 4 dòng prediction final | Chương 3, Chương 4, Phụ lục | `models/saved/final/` trống; `models/final/` có checkpoint nhưng manifest/shape không khớp hoàn toàn final report | Tìm artifact trước cleanup hoặc rerun đúng script tạo `FINAL_PROJECT_STATUS.md` |
| Student final per-run metrics CSV/log | Chương 4 | Student rows chỉ có trong README/final status/CLEANUP; không có manifest/deep CSV final riêng | Tìm log run hoặc xuất lại metrics từ locked predictions/checkpoints |
| Exact source/checkpoint của `gated_fusion_v28` | Chương 3, Chương 4 | Final manifest ghi `gated_fusion_v28`, nhưng source class trực tiếp tên v28 không tìm thấy; source gần nhất là `src/models_v27.py` | Tìm commit/artifact v28 hoặc thêm mapping rõ từ `gated_fusion_v28` sang class implementation |
| Numeric threshold của `low_f1_tuned` cho từng final row | Chương 3, Chương 4 | Source có hàm chọn threshold trên OOF, nhưng final manifest không lưu threshold cụ thể cho Student rows | Lấy lại OOF probability/threshold log hoặc rerun threshold tuning và lưu vào manifest |
| OOF probabilities dùng để tuning threshold | Chương 4 | Artifact OOF không có trong final folders | Khôi phục predictions từ run cũ hoặc rerun CV/OOF |
| Confusion matrix final | Chương 4 | Không thấy confusion matrix final tương ứng với bốn dòng final | Rerun evaluation từ locked predictions hoặc tạo từ saved prediction CSV nếu tìm được |
| Full baseline final cho student-mat và student-por | Chương 4 | `final_baseline_comparison.csv` chỉ có xAPI | Chạy/khôi phục baseline locked-test output cho Student datasets |
| Low-class metrics của baseline xAPI | Chương 4 | Final baseline CSV ghi `not_available` cho Recall Low/F1 Low | Tính từ baseline predictions nếu còn file prediction, hoặc rerun baseline evaluation |
| Statistical significance test | Chương 4, Chương 5 | Không có artifact test ý nghĩa thống kê | Dùng locked predictions để chạy McNemar/bootstrap/paired test |
| Ablation study final | Chương 4 | Có source/variant thử nghiệm trong archive nhưng chưa có bảng ablation final đủ evidence | Khôi phục hoặc chạy ablation nhỏ theo same split và lưu CSV |
| Latency/inference-time benchmark | Chương 4, Chương 5 | Không thấy đo thời gian inference hoặc throughput | Chạy benchmark nhẹ trên CPU/GPU với checkpoint final |
| Calibration metrics | Chương 4, Chương 5 | Hệ thống dùng probability cho RA-HLPR nhưng không thấy ECE/Brier/calibration curve final | Tính calibration trên locked test nếu có predictions/probabilities |
| Dashboard source/lệnh chạy/screenshot | Chương 3, Chương 4 | Không có evidence đủ mạnh cho dashboard vận hành | Bổ sung app source hoặc screenshot thật kèm lệnh chạy |
| User feedback sau recommendation | Chương 4, Chương 5 | RA-HLPR chỉ có offline evaluation; không có khảo sát/A-B/user study | Thu thập phản hồi người dùng hoặc ghi rõ không có đánh giá thực tế |
| Bằng chứng recommendation cải thiện thành tích | Chương 4, Chương 5 | Không có can thiệp triển khai theo thời gian hay outcome sau can thiệp | Cần study longitudinal hoặc thử nghiệm với người dùng thật |
| Dataset source/version citation | Chương 2 | Repository chỉ cấu hình tên file; không có tài liệu nguồn/version raw data trong artifact final | Bổ sung citation chính thức khi viết báo cáo, nhưng không bịa từ repo |
| Exact missing values/duplicates | Chương 2 | Raw CSV không có; không thể kiểm trực tiếp | Khôi phục raw và chạy `df.isna().sum()`, `df.duplicated().sum()` |
| Exact feature list sau preprocessing của final run | Chương 3 | Feature selector source có logic, nhưng final feature metadata trong `models/saved/final` thiếu | Khôi phục `*_ensemble_features.json` hoặc chạy preprocessing và lưu danh sách feature |
| Random seeds của tất cả final Student runs | Chương 3, Chương 4 | Source có `DEFAULT_SEED=42`, archive có nhiều seed, nhưng final Student rows không có manifest | Tìm log/manifest final hoặc bổ sung seed table |
| Batch size/epoch/lr exact của final Student rows | Chương 3 | Source có defaults và checkpoint non-final có metadata, nhưng final Student exact config thiếu | Khôi phục best params JSON/log hoặc rerun Optuna/training |
| `student-combine` status nếu có trong archive | Chương 2, Chương 5 | Final reports nói không dùng dataset chính; không cần viết kết quả chính nếu không có final artifact | Chỉ nêu guardrail: không dùng làm dataset chính, dựa trên final manifest/report |
| ADASYN usage final | Chương 3, Chương 5 | Source có cảnh báo/fallback; final report nói ADASYN trực tiếp trên categorical label encoding bị loại trừ | Không claim dùng ADASYN final; nếu cần, tìm run/config cụ thể |
| Exact training hardware | Phụ lục, Chương 4 | Không thấy GPU/CPU log final | Bổ sung máy chạy, GPU, driver, thời gian chạy nếu cần |
| Report Word/PDF cũ có số liệu bổ sung | Tất cả chương | File Word/PDF là tài liệu tham khảo, không mạnh hơn source/artifact; chưa render/đối chiếu từng hình/bảng | Chỉ dùng sau khi đối chiếu với artifact source hoặc ghi là reference draft |
| Ethical/privacy handling chi tiết | Chương 5 | Có thể suy luận dataset giáo dục cần bảo mật, nhưng repo không có policy triển khai | Viết ở mức nguyên tắc, không claim đã triển khai privacy audit nếu không có code/policy |
