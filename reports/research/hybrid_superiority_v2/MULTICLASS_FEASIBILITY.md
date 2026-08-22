# Khả thi đa lớp OULAD (không đổi bài toán khóa)

Thử nghiệm **CPU**, Logistic Regression static-only, inner fold 0, FIT-only encode. **Không** train Hybrid, **không** mở outer.
Bài khóa vẫn nhị phân Fail|Withdrawn. Đây chỉ là feasibility.

## 1. Cỡ mẫu theo cutoff (sau lọc unregistration)

 stage     n  Pass  Pass_pct  Distinction  Distinction_pct  Fail  Fail_pct  Withdrawn  Withdrawn_pct  risk_binary  risk_binary_pct
 20pct 26697 12357  0.462861         3024         0.113271  7039  0.263663       4277       0.160205        11316         0.423868
 35pct 25606 12360  0.482699         3024         0.118097  7042  0.275014       3180       0.124190        10222         0.399203
 50pct 24599 12361  0.502500         3024         0.122932  7043  0.286312       2171       0.088256         9214         0.374568
 75pct 23159 12361  0.533745         3024         0.130576  7043  0.304115        731       0.031564         7774         0.335679
100pct 22522 12361  0.548841         3024         0.134269  7043  0.312716         94       0.004174         7137         0.316890

Withdrawn **bốc hơi** khi cutoff tăng: còn hạn nộp thì nhiều người đã rút trước đó bị loại khỏi risk-set.

## 2. Baseline đa lớp (static LR, fold 0 VALID)

stage          scheme  n_fit  n_valid  accuracy  macro_f1  weighted_f1  binary_ap_same_features  min_class_support_valid
35pct          4class   9122     5672  0.321756  0.314396     0.331966                 0.549806                      647
35pct 3class_passband   9122     5672  0.469323  0.417434     0.498044                 0.549806                      692
50pct          4class   8748     5461  0.321003  0.306530     0.336947                 0.533901                      481
50pct 3class_passband   8748     5461  0.470060  0.402463     0.508666                 0.533901                      481
75pct          4class   8246     5129  0.321505  0.285091     0.354071                 0.495007                      149
75pct 3class_passband   8246     5129  0.480016  0.373394     0.539788                 0.495007                      149

## 3. Kết luận khả thi

- 4 lớp (Distinction/Pass/Fail/Withdrawn): Distinction vốn ít; Withdrawn VALID ở 75% còn **731** trên toàn risk-set (toàn bộ cutoff, chưa split) — sau group-split VALID còn nhỏ hơn ~1/3.
- Withdrawn 35% còn **3180** (toàn risk-set) — 3 lớp PassBand/Fail/Withdrawn **khả thi hơn** 4 lớp.
- Macro-F1 4-class trên static LR (fold 0): 35pct=0.314, 50pct=0.307, 75pct=0.285.
- Macro-F1 3-class: 35pct=0.417, 50pct=0.402, 75pct=0.373.
- Static LR đa lớp **không** thay Hybrid; chỉ cho biết lớp có tách được trên ngữ cảnh tĩnh hay không.

**Khả thi?**

- **3 lớp PassBand / Fail / Withdrawn trên 35–50%:** có thể thử (Withdrawn còn đủ). 75% Withdrawn quá mỏng → F1 lớp đó không ổn định.
- **4 lớp (tách Distinction):** kém khả thi — Distinction ít, dễ bị nuốt vào Pass.
- **Không nên thay bài khóa** nếu mục tiêu vẫn là cảnh báo nguy cơ: nhị phân Fail|Withdrawn khớp Rec V (một ngưỡng t). Đa lớp đổi metric (macro-F1), đổi head, đổi Rec V.
- Nếu chỉ **bổ sung phân tích** (không khóa mô hình): 3-class ở 35/50% là mức đáng làm; 4-class và 75% Withdrawn thì không.
