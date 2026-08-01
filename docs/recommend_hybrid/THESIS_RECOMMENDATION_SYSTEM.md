# Hybrid CNN-BiLSTM Learning Support Recommender

## 1. Mục tiêu hệ thống

Hệ thống tạo kế hoạch hỗ trợ học tập có thể truy vết từ kết quả dự đoán của CNN-BiLSTM và bằng chứng học tập quan sát được trước cutoff. Mục tiêu là tính đúng kỹ thuật, nhất quán và an toàn theo thời gian; hệ thống không tuyên bố hành động tối ưu hay cải thiện điểm số.

## 2. Kiến trúc nghiệp vụ

Luồng xử lý gồm mô hình dự đoán hybrid CNN-BiLSTM đã đóng băng, trạng thái học tập trước cutoff, policy bằng chứng theo dataset, bộ chọn priority xác định, Constraint Solver, Learning Plan Builder, explanation/lineage và persistence có thể replay. Recommendation là deterministic evidence-based policy, không phải neural ranker được huấn luyện.

## 3. Hai nhánh UCI và OULAD

`RecommendHybridUCI` tách biệt `student_mat` và `student_por`, dùng cùng contract nhưng policy threshold riêng. `RecommendHybridOULAD` xử lý dữ liệu hoạt động VLE và assessment theo tiến trình khóa học. Action catalog của hai nhánh được cô lập; sai mapping dataset/model bị từ chối.

## 4. Cơ chế stage và cutoff

UCI dùng S0 khi chưa có G1, S1 khi chỉ có G1 và S2 khi có G1/G2; G3 bị cấm. OULAD có anchor EARLY_20, EARLY_35, MIDDLE_50, LATE_75 và FINAL_EVALUATION. Yêu cầu trung gian 25, 36, 63 và 76 chỉ dùng anchor hợp lệ gần nhất trong quá khứ; chúng kiểm tra routing/recommendation behavior, không phải prediction performance tại cutoff mới.

## 5. Cách tạo khuyến nghị

Policy chỉ đưa action vào diện xem xét khi có bằng chứng trực tiếp, stage hợp lệ và không vi phạm missing-evidence rule. Risk class/probability chỉ điều chỉnh context sau bằng chứng; uncertainty chỉ giảm priority hoặc automation. Priority là CRITICAL/HIGH/MEDIUM/LOW, không phải xác suất hiệu quả.

## 6. Constraint và abstention

Kế hoạch tối đa bốn action và 180 phút mỗi period. Solver kiểm tra duplicate, prerequisite, contraindication, conflict, dataset/stage, human contact và thời gian còn lại. UCI dùng CURRENT_PERIOD/NEXT_ASSESSMENT/FOLLOW_UP; OULAD dùng IMMEDIATE/SHORT_TERM/FOLLOW_UP. ABSTAIN và EVALUATION_ONLY luôn có 0 action.

## 7. Phương pháp đánh giá

Đánh giá cuối dùng 260 record pseudonymized từ prediction OOF/seed hybrid canonical và feature trước cutoff: 120 UCI, 100 OULAD canonical-anchor và 40 OULAD inter-stage. Chọn mẫu theo khóa ổn định, không dùng target/outcome. Các nhóm đánh giá gồm safety, constraint, coverage/abstention, evidence/explanation, diversity, scenario/metamorphic, reproducibility, ablation, robustness và bootstrap student-level 1.000 lần.

## 8. Kết quả theo dataset

MAT và POR đều có coverage 95%, abstention 5%, evidence support 100%; mean action lần lượt 2,37 và 2,63. OULAD có coverage 89,17% trên 120 record intervention, abstention 10,83%, mean action 2,67 và prediction age trung bình 1,43 điểm phần trăm. FINAL_EVALUATION gồm 20 record, tất cả 0 intervention. Mọi safety/constraint violation đều bằng 0.

## 9. Ablation

A (risk class) và B (risk probability) đạt coverage 100% nhưng evidence support 0%, unsupported action 100% và constraint violation 7,69% do không có stage/final safeguard. C (risk + evidence, uncertainty neutralized) có evidence support 100% và coverage 100%. D là policy chính thức, coverage 92,08% và abstention 7,92% do uncertainty safety. Chỉ D được phát hành.

## 10. Robustness

Bảy kiểm thử có kiểm soát thay đổi assessment progress, inactivity, absences, study time, uncertainty, requested cutoff và remaining course time đều PASS. Vấn đề nghiêm trọng hơn không làm priority liên quan giảm; vấn đề đã giải quyết làm action giảm/biến mất; uncertainty tăng không làm automation tăng; cutoff thay đổi không tạo future anchor.

## 11. Bootstrap

Bootstrap percentile theo pseudonymous student, 1.000 lần với seed 20260801, cho coverage 95% CI [87,81%; 95,97%], abstention [4,03%; 12,19%], mean action [2,39; 2,77], mean workload [118,91; 151,48] và top-action share [23,81%; 32,27%]. Khoảng này mô tả tập kiểm định kỹ thuật, không suy ra hiệu quả giáo dục.

## 12. Hạn chế

Dự án không có chuyên gia thật, khảo sát người dùng, dữ liệu can thiệp, action relevance ground truth hoặc outcome sau khuyến nghị. `PROGRESS_MONITORING` xuất hiện trong 78,28% plan actionable; audit cho thấy action có evidence và không do selector thêm mặc định, nhưng eligibility LOW-severity của policy khá rộng. Phase 5 chỉ báo cáo, không tối ưu policy sau khi xem kết quả.

## 13. Claim boundary

Được hỗ trợ: prediction baseline không đổi, an toàn cutoff, evidence linkage, UCI/OULAD routing, constraint correctness và deterministic replay. Chỉ được hỗ trợ một phần: tính phù hợp giáo dục, vì chưa có đánh giá con người. Không được hỗ trợ: optimality, grade improvement, expert validation, user acceptance, causal effectiveness và các metric ranking như Precision@K/NDCG.

## 14. Kết luận

Hệ thống hoàn thành một pipeline khuyến nghị hỗ trợ học tập xác định, truy vết được và an toàn theo protocol đã khóa. Kết quả chứng minh chất lượng kỹ thuật trong phạm vi test và evaluation sample. Bất kỳ tuyên bố nào về hiệu quả học tập cần một nghiên cứu mới có chuyên gia/người dùng hoặc can thiệp thực tế, thiết kế và preregister độc lập với release này.
