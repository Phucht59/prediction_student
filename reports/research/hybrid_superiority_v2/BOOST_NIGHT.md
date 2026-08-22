# BOOST night — Hybrid dùng chung UCI+OULAD

Nhãn nhị phân **không đổi** (OULAD Fail|Withdrawn). Bỏ train/STOP **20%**. STOP 35/50/75 và UCI S1+S2.
Chỉ số STOP: AP + 0.3 F1 + 0.15 ROC-AUC + 0.1 Rec + 0.1 Prec − 0.25 ECE.
Kernel CNN 3 (cả hai miền), Δ tuần, last-step, FiLM progress, rank loss 0.05.
CSV: `C:/hufit/student/artifacts/research/kltn_science_fix/boost/boost_valid.csv`.

| domain | stage | Boost AP | locked AP | Boost F1 | n |
|---|---|---:|---:|---:|---:|
| oulad | 35pct | 0.8068 | 0.8058 | 0.7008 | 9 |
| oulad | 50pct | 0.8492 | 0.8483 | 0.7348 | 9 |
| oulad | 75pct | 0.8898 | 0.8885 | 0.7822 | 9 |
| uci | S0 | 0.4244 | — | 0.0861 | 9 |
| uci | S1 | 0.8133 | 0.8214 | 0.6916 | 9 |
| uci | S2 | 0.8946 | 0.9101 | 0.7660 | 9 |
