# LABEL_SPLIT_ANALYSIS

OOF serving 66 685 dòng (20/35/50/75, 3 inner fold, seed 42) join `studentInfo.final_result`.
Không có mốc 100% trong OOF này.

| stage | n | Fail | Withdrawn | Pass/Dist | AP gộp | AP Fail vs Pass | AP Withdrawn vs Pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20pct | 17787 | 4719 | 2849 | 10219 | 0.7557 | 0.6921 | 0.5511 |
| 35pct | 17066 | 4720 | 2125 | 10221 | 0.8073 | 0.7754 | 0.5851 |
| 50pct | 16394 | 4721 | 1452 | 10221 | 0.8468 | 0.8312 | 0.6060 |
| 75pct | 15438 | 4721 | 496 | 10221 | 0.8884 | 0.8847 | 0.6083 |

AP Fail vs Pass loại Withdrawn khỏi đánh giá; AP Withdrawn vs Pass loại Fail.
Serving AP 100% = 0.9204 (bảng khóa) **không** nằm file OOF này. 100% còn ~94 Withdrawn sau lọc cutoff — không dùng làm bằng chứng cảnh báo sớm.
CSV: `C:/hufit/student/reports/research/hybrid_superiority_v2/label_split.csv`.
