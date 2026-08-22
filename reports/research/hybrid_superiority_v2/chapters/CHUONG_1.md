# Chương 1. Tổng quan đề tài nghiên cứu

## 1.1. Lý do chọn đề tài

Cảnh báo sớm nguy cơ học tập (early warning) đặt ra trước khi có điểm cuối kỳ: nhà trường cần biết **ai đang có nguy cơ trượt hoặc rút**, theo thứ tự ưu tiên, để can thiệp khi vẫn còn thời gian. Hai bộ dữ liệu công khai thường dùng trong Educational Data Mining khác nhau về bản chất thời gian:

- **UCI Student Performance** (Cortez & Silva, 2008): bản ghi học kỳ, chuỗi điểm tối đa hai bước (G1, G2).
- **OULAD** (Kuzilek, Hlosta & Zdrahal, 2017): nhật ký VLE theo tuần, cutoff 20–100% chiều dài môn.

Một kiến trúc lai CNN–BiLSTM có thể dùng chung cho cả hai miền nếu tensor, mask và cổng fusion được thiết kế cutoff-safe. Recommendation đi sau dự đoán: xếp hành động khả thi, không ước lượng nhân quả lên kết quả cuối khóa.

## 1.2. Mục tiêu nghiên cứu

### 1.2.1. Mục tiêu tổng quát

Xây dựng và đánh giá mô hình **Hybrid CNN–BiLSTM** dự đoán nguy cơ học tập nhị phân trên UCI và OULAD, rồi gắn **Recommendation V** trên OULAD 20/35/50/75%.

### 1.2.2. Mục tiêu cụ thể (đo được)

1. Một class `Hybrid`, một checkpoint / miền, chấm mọi mốc thông tin.
2. Chỉ số chính: AP = `sklearn.metrics.average_precision_score` trên VALID inner, trung bình 9 run (3 fold × 3 seed).
3. So sánh cùng protocol với LR, DT, RF, SVM, MLP.
4. Recommendation V: NDCG@3 trên Panel C, invalid-action = 0, không ATE.
5. Kiểm định giả thuyết H1–H3 (mục 1.4) bằng ablation và Wilcoxon.

## 1.3. Đối tượng và phạm vi

- **Đối tượng:** Hybrid CNN–BiLSTM (mô hình khóa) và Recommendation V. Baseline chỉ để so sánh.
- **Phạm vi dữ liệu:** UCI gộp 1 044 dòng; OULAD enrollment + VLE, nhãn Fail|Withdrawn.
- **Phạm vi đánh giá:** inner FIT/STOP/VALID, group-split, **không mở outer test**.
- **Phạm vi khuyến nghị:** OULAD 20–75%; 100% không can thiệp.
- **Ngoài phạm vi:** dữ liệu sinh viên Việt Nam; thử nghiệm với giảng viên thật; ước lượng hiệu ứng can thiệp.

## 1.4. Câu hỏi và giả thuyết (kiểm định được)

- **H1:** Trên UCI S1 và OULAD 35%, AP Hybrid full > AP tabular-only (Wilcoxon hai phía, 9 run, α = 0.05).
- **H2:** Trên các mốc có chuỗi (UCI S1; OULAD 35%+), AP Hybrid > AP LR và AP RF (9 cặp fold×seed, α = 0.05).
- **H3:** Cổng softmax tăng khối lượng CNN+BiLSTM khi cutoff OULAD tăng.

Claim chính đặt ở S1/S2 và 35–75%. S0/20% là mốc thiếu chuỗi, không dùng để bác kiến trúc lai. Kết quả kiểm định: Chương 4.

## 1.5. Phương pháp

- Lý thuyết: EDM, early warning, CNN, BiLSTM, AP, ranking.
- Thực nghiệm: group-disjoint inner 3-fold, FIT-only scale và `pos_weight`, STOP early-stop AP, ngưỡng STOP (F1 → recall → `|t−0.5|`).
- Không dùng outer để chọn kiến trúc hay siêu tham số.

## 1.6. Ý nghĩa

- Học thuật: một protocol cutoff-safe, AP, cổng có mask, ablation và kiểm định cặp — không chỉ “một con số accuracy”.
- Thực tiễn: protocol cutoff-safe và Recommendation V trên OOF đã khóa. **Không** xây giao diện người dùng.

## 1.7. Bố cục

Chương 2 cơ sở lý thuyết; Chương 3 phân tích–thiết kế (không bảng hiệu suất); Chương 4 thực nghiệm; Chương 5 kết luận, hạn chế, hướng phát triển.
