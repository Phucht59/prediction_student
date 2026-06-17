# T?m t?t h?c thu?t module khuy?n ngh? l? tr?nh h?c t?p

## M? t? module
Module khuy?n ngh? l? tr?nh h?c t?p ???c x?y d?ng nh? m?t th?nh ph?n downstream sau m? h?nh d? ?o?n CNN-BiLSTM. M?c ti?u l? chuy?n k?t qu? d? ?o?n h?c l?c th?nh c?c khuy?n ngh? can thi?p v? l? tr?nh h?c t?p 4 tu?n c? x?t ??n r?i ro h?c t?p c?a t?ng sinh vi?n.

Module n?y kh?ng ph?i collaborative filtering, v? c?c b? d? li?u kh?ng c? l?ch s? t??ng t?c user-item ho?c ph?n h?i sau khi sinh vi?n nh?n khuy?n ngh?.

## S? ?? pipeline

```text
X?c su?t CNN-BiLSTM (Low/Medium/High)
-> RiskDiagnosisHead
-> CandidateGenerator theo dataset/risk
-> HybridScorer
-> PathPlanner
-> L? tr?nh h?c t?p 4 tu?n
```

## C?ng th?c ch?m ?i?m

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

C?c tr?ng s? ???c ?i?u ch?nh theo l?p d? ?o?n v? m?c r?i ro. V?i sinh vi?n ???c d? ?o?n Low ho?c c? r?i ro cao, h? th?ng ?u ti?n `risk_match` v? `performance_need`. V?i sinh vi?n Medium, h? th?ng d?ng tr?ng s? c?n b?ng. V?i sinh vi?n High ho?c ?n ??nh, h? th?ng ?u ti?n ho?t ??ng n?ng cao, ?? ph? h?p ?? kh? v? ?i?u ki?n ti?n quy?t. `rule_adjustment` ???c d?ng ?? ??m b?o logic s? ph?m: Student R1/R2 ?u ti?n luy?n t?p, tutoring, bootcamp v? academic coaching; xAPI R4 ?u ti?n LMS/resource/discussion; h? tr? ph? huynh ch? ???c ??y cao khi R6/support risk cao.

## B?ng y?u t? r?i ro

| Nh?m d? li?u | R?i ro | T?n hi?u s? d?ng |
|---|---|---|
| Student | R1 - n?ng l?c n?n th?p | failures, G1 |
| Student | R2 - xu h??ng gi?m | G2 th?p h?n G1 |
| Student | R3 - r?i ro chuy?n c?n | absences |
| Student | R4 - m?c ?? tham gia th?p | goout, freetime, activities |
| Student | R5 - th?i gian h?c ch?a ?? | studytime |
| Student | R6 - nguy c? th?t b?i cao | failures, G1/G2 v? xu h??ng; kh?ng d?ng G3 |
| xAPI | R3 - r?i ro chuy?n c?n | StudentAbsenceDays |
| xAPI | R4 - m?c ?? t??ng t?c th?p | VisITedResources, raisedhands, Discussion, AnnouncementsView |
| xAPI | R6 - nguy c? th?t b?i cao | chuy?n c?n, t??ng t?c, h? tr? ph? huynh/nh? tr??ng; kh?ng d?ng true Class |

## Nh?m can thi?p

| Nh?m can thi?p | V? d? | Ph?m vi ?p d?ng |
|---|---|---|
| Chuy?n c?n | Daily Attendance Monitoring, Absence Recovery Pack | both/xAPI khi c? R3 |
| L?p k? ho?ch h?c t?p | Time Management Workshop, Standard Practice Plan | both |
| T??ng t?c LMS | Resource Checklist, Maintain LMS Engagement, Interactive Quiz | xAPI |
| H? tr? b?n h?c/nh?m | Peer Tutoring, Study Group | student/both |
| Luy?n t?p b? ??p | Targeted Practice, Remedial Bootcamp, Academic Coaching | student |
| H? tr? ph? huynh/nh? tr??ng | Parent Sync, Family Progress Contract | both, ch? ?u ti?n khi R6 cao |
| Duy tr?/m? r?ng | Weekly Progress Review, Advanced Seminar, Optional Discussion | both/xAPI |

## L? tr?nh 4 tu?n

| Tu?n | Ch? ?? | N?i dung |
|---|---|---|
| Tu?n 1 | Stabilize | ?n ??nh chuy?n c?n, h? tr? v? l?ch h?c |
| Tu?n 2 | Practice | luy?n t?p v? b? ??p l? h?ng ki?n th?c |
| Tu?n 3 | Reinforce | c?ng c? th?ng qua t??ng t?c, LMS v? h?c nh?m |
| Tu?n 4 | Evaluate & Adjust | ??nh gi? ti?n ?? v? ?i?u ch?nh chu k? ti?p theo |

## K?t qu? ??nh gi? offline

| Dataset | Risk Macro F1 | Risk Micro F1 | Precision@3 | Recall@3 | NDCG@3 | Coverage@3 | Risk Coverage | Workload Std | Difficulty Progression | Prereq Violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xapi | 0.9831 | 0.9813 | 0.6840 | 0.4720 | 0.8229 | 0.6500 | 0.8958 | 1.1210 | 0.7153 | 0.0000 |
| student-por | 0.9359 | 0.9094 | 0.6641 | 0.3185 | 0.7455 | 0.5500 | 0.9508 | 1.3137 | 0.6000 | 0.0449 |

Ghi ch? Student-Mat: pending full run because missing final prediction checkpoint metadata: models/saved/final/student-mat_3class_ensemble_features.json. The available Student-Mat checkpoint input shape does not match regenerated feature selection, so outputs/recommender/student-mat was not refreshed in this run.

## Ki?m tra logic sau khi s?a
- Case Student-Por c? R1/R2 cao ?? ?u ti?n Peer-Led Study Tutoring, Targeted Practice Exercises v? Academic Coaching trong top 3.
- Case xAPI Medium kh?ng c? r?i ro ?? chuy?n sang Standard Practice Plan, Weekly Progress Review v? Maintain LMS Engagement.
- Case xAPI Low c? engagement risk ?? ?u ti?n LMS Resource Checklist, Discussion Prompts v? Interactive Quizzing.

## Gi?i h?n
- ??nh gi? recommender l? ??nh gi? offline d?a tr?n weak-supervision/rule-based reference.
- Ch?a c? d? li?u ph?n h?i th?c t? c?a sinh vi?n sau khi nh?n khuy?n ngh?.
- V? v?y, k?t qu? kh?ng ???c di?n gi?i nh? b?ng ch?ng c?i thi?n nh?n qu?. Module ch? ???c claim l? h? tr? ra quy?t ??nh v? c? nh?n h?a l? tr?nh h?c t?p d?a tr?n d? ?o?n v? r?i ro quan s?t ???c.
