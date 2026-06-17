# Tóm tắt học thuật module khuyến nghị lộ trình học tập

## Mô tả module
Module khuyến nghị lộ trình học tập được xây dựng như một thành phần downstream sau mô hình dự đoán CNN-BiLSTM. Mục tiêu của module là chuyển xác suất dự đoán học lực Low/Medium/High thành các khuyến nghị can thiệp và lộ trình học tập 4 tuần có xét đến rủi ro học tập của từng sinh viên.

Module này không phải collaborative filtering, vì các bộ dữ liệu không có lịch sử tương tác user-item hoặc phản hồi thực tế sau khi sinh viên nhận khuyến nghị. Do đó, hệ thống được thiết kế theo hướng rule-aware và prediction-aware: sử dụng xác suất dự đoán, đặc trưng quan sát được và tri thức can thiệp giáo dục.

## Sơ đồ pipeline

```text
Xác suất CNN-BiLSTM (Low/Medium/High)
-> RiskDiagnosisHead
-> CandidateGenerator theo dataset/risk
-> HybridScorer
-> PathPlanner
-> Lộ trình học tập 4 tuần
```

## Công thức chấm điểm

```text
score =
w1 * risk_match
+ w2 * performance_need
+ w3 * difficulty_fit
+ w4 * time_fit
+ w5 * prerequisite_fit
+ w6 * expected_effect
+ rule_adjustment
```

Trong đó, các trọng số được điều chỉnh theo lớp dự đoán và mức rủi ro. Với sinh viên được dự đoán Low hoặc có rủi ro cao, hệ thống ưu tiên `risk_match` và `performance_need`. Với sinh viên Medium, hệ thống dùng trọng số cân bằng. Với sinh viên High hoặc ổn định, hệ thống ưu tiên hoạt động duy trì, mở rộng, độ phù hợp độ khó và điều kiện tiên quyết.

Thành phần `rule_adjustment` được dùng để đảm bảo logic sư phạm: Student R1/R2 ưu tiên luyện tập, tutoring, bootcamp và academic coaching; xAPI R4 ưu tiên LMS/resource/discussion; hỗ trợ phụ huynh chỉ được đẩy cao khi R6 hoặc support risk cao.

## Bảng yếu tố rủi ro

| Nhóm dữ liệu | Rủi ro | Tín hiệu sử dụng |
|---|---|---|
| Student | R1 - năng lực nền thấp | failures, G1 |
| Student | R2 - xu hướng giảm | G2 thấp hơn G1 |
| Student | R3 - rủi ro chuyên cần | absences |
| Student | R4 - mức độ tham gia thấp | goout, freetime, activities |
| Student | R5 - thời gian học chưa đủ | studytime |
| Student | R6 - nguy cơ thất bại cao | failures, G1/G2 và xu hướng; không dùng G3 |
| xAPI | R3 - rủi ro chuyên cần | StudentAbsenceDays |
| xAPI | R4 - mức độ tương tác thấp | VisITedResources, raisedhands, Discussion, AnnouncementsView |
| xAPI | R6 - nguy cơ thất bại cao | chuyên cần, tương tác, hỗ trợ phụ huynh/nhà trường; không dùng true Class |

## Nhóm can thiệp

| Nhóm can thiệp | Ví dụ | Phạm vi áp dụng |
|---|---|---|
| Chuyên cần | Daily Attendance Monitoring, Absence Recovery Pack | both/xAPI khi có R3 |
| Lập kế hoạch học tập | Time Management Workshop, Standard Practice Plan | both |
| Tương tác LMS | Resource Checklist, Maintain LMS Engagement, Interactive Quiz | xAPI |
| Hỗ trợ bạn học/nhóm | Peer Tutoring, Study Group | student/both |
| Luyện tập bù đắp | Targeted Practice, Remedial Bootcamp, Academic Coaching | student |
| Hỗ trợ phụ huynh/nhà trường | Parent Sync, Family Progress Contract | both, chỉ ưu tiên khi R6 cao |
| Duy trì/mở rộng | Weekly Progress Review, Advanced Seminar, Optional Discussion | both/xAPI |

## Lộ trình 4 tuần

| Tuần | Chủ đề | Nội dung |
|---|---|---|
| Tuần 1 | Stabilize | ổn định chuyên cần, hỗ trợ và lịch học |
| Tuần 2 | Practice | luyện tập và bù đắp lỗ hổng kiến thức |
| Tuần 3 | Reinforce | củng cố thông qua tương tác, LMS và học nhóm |
| Tuần 4 | Evaluate & Adjust | đánh giá tiến độ và điều chỉnh chu kỳ tiếp theo |

## Kết quả đánh giá offline

| Dataset | Risk Macro F1 | Risk Micro F1 | Precision@3 | Recall@3 | NDCG@3 | Coverage@3 | Risk Coverage | Workload Std | Difficulty Progression | Prereq Violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xapi | 0.9831 | 0.9813 | 0.6840 | 0.4720 | 0.8229 | 0.6500 | 0.8958 | 1.1210 | 0.7153 | 0.0000 |
| student-por | 0.9359 | 0.9094 | 0.6641 | 0.3185 | 0.7455 | 0.5500 | 0.9508 | 1.3137 | 0.6000 | 0.0449 |

Ghi chú Student-Mat: pending full run vì thiếu metadata checkpoint `models/saved/final/student-mat_3class_ensemble_features.json`. Checkpoint Student-Mat hiện có không khớp input shape khi tái tạo feature selection, nên pipeline recommender Student-Mat không được refresh trong lần chạy cuối.

## Kiểm tra logic sau khi sửa
- Case Student-Por có R1/R2 cao đã ưu tiên Peer-Led Study Tutoring, Targeted Practice Exercises và Biweekly Academic Coaching trong top 3.
- Case xAPI Medium không có rủi ro đã chuyển sang Standard Practice Plan, Weekly Progress Review và Maintain LMS Engagement.
- Case xAPI Low có engagement risk đã ưu tiên Daily LMS Resource Checklist, Guided Discussion Prompts và LMS Interactive Quizzing.

## Giới hạn
- Đánh giá recommender là đánh giá offline dựa trên weak-supervision/rule-based reference.
- Chưa có dữ liệu phản hồi thực tế của sinh viên sau khi nhận khuyến nghị.
- Vì vậy, kết quả không được diễn giải như bằng chứng cải thiện nhân quả. Module chỉ được claim là hỗ trợ ra quyết định và cá nhân hóa lộ trình học tập dựa trên dự đoán và rủi ro quan sát được.
