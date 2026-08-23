# Chương 4. Kết quả thực nghiệm và đánh giá

Chương này trình bày kết quả thực nghiệm của mô hình Hybrid CNN–BiLSTM và của module khuyến nghị. Các mô hình học máy truyền thống gồm hồi quy logistic, cây quyết định, rừng ngẫu nhiên, SVM, mạng perceptron đa lớp và XGBoost được huấn luyện trên cùng quy trình để đối sánh. Chỉ số chính là Average Precision (AP). Phần chia ngoài không dùng khi công bố kết quả. Các hình được vẽ từ số liệu đã tổng hợp, không huấn luyện lại.

---

## 4.1. Môi trường thực nghiệm

Toàn bộ quá trình từ khám phá dữ liệu, huấn luyện mô hình đến lưu kết quả dự đoán được thực hiện trên một máy, nhằm bảo đảm tính ổn định và khả năng tái lập.

- Về phần cứng và hệ điều hành:
Các thí nghiệm được tiến hành trên hệ điều hành Windows 10 (64-bit). CPU gồm 12 luồng logic, bộ nhớ RAM khoảng 16 GB. GPU là NVIDIA GeForce RTX 2060, 6 GB VRAM. Huấn luyện học sâu dùng độ chính xác hỗn hợp trên GPU; nếu không có GPU thì chuyển về CPU.

- Về môi trường phát triển và thư viện:
Ngôn ngữ lập trình chính là Python 3.10. Việc phát triển được thực hiện trên Visual Studio Code. Notebook đánh giá tái tạo các hình từ số liệu đã lưu.

- Các thư viện và framework mã nguồn mở đóng vai trò cốt lõi bao gồm:
  - Framework học sâu: PyTorch 2.11, dùng để xây dựng và huấn luyện Hybrid CNN–BiLSTM.
  - Xử lý dữ liệu và học máy: scikit-learn, pandas, numpy, XGBoost, dùng cho tiền xử lý, các mô hình đối sánh và tính AP, F1, độ chính xác.
  - Trực quan hóa: Matplotlib.
  - Module khuyến nghị: Explainable Boosting Machine.
  - Lưu trữ kết quả: PostgreSQL.

Cấu hình Hybrid CNN–BiLSTM: không gian ẩn 128, CNN 64 kênh, BiLSTM ẩn 128. UCI dùng tốc độ học 8,61×10⁻⁵, Dropout 0,406, lô 32. OULAD dùng tốc độ học 1,18×10⁻⁴, Dropout 0,320, lô 128. Ba hạt giống 42, 1201 và 2026.

---

## 4.2. Các chỉ số đánh giá hiệu suất

### 4.2.1. Phương pháp đánh giá

Để hiệu suất được đánh giá khách quan, đề tài không dùng kiểm định chéo xáo trộn độc lập từng dòng. Phần không thuộc chia ngoài được chia ba phần theo nhóm: UCI theo nhóm học sinh (662 nhóm / 1.044 dòng), OULAD theo mã sinh viên. Trong mỗi phần, tập huấn luyện dùng để chuẩn hóa, tính trọng số lớp dương và cập nhật trọng số mạng; tập dừng dùng để dừng sớm theo AP và chọn ngưỡng; tập kiểm định dùng để báo cáo. Ba hạt giống nhân ba phần chia tạo thành chín số liệu cho mỗi mốc. Kết quả công bố là trung bình chín số, không lấy lần chạy cao nhất.

Tiêu chí chính là AP trên tập kiểm định. Độ chính xác, Precision, F1 và Recall được tính tại một ngưỡng đã chọn trên tập dừng (theo F1, rồi độ nhạy, rồi độ gần 0,5). Một ngưỡng không tối đa đồng thời cả ba chỉ số ngưỡng.

Module khuyến nghị được đánh giá trên tập độc lập 632 tình huống của 150 sinh viên, không hiệu chỉnh trên tập này. Các chỉ số gồm NDCG tại 3, Precision tại 1 và tỷ lệ hành động không hợp lệ.

### 4.2.2. Phân tích quá trình huấn luyện

Quá trình huấn luyện được theo dõi qua AP trên tập kiểm định theo từng phần chia, ngưỡng trên tập dừng, và khoảng cách giữa AP tập huấn luyện với AP tập kiểm định trên chín lần chạy. File trọng số đã lưu không chứa lịch sử theo epoch, nên không vẽ đường mất mát giả.

Hình 4.1 trình bày AP trên bốn mốc 20–75% theo ba phần chia (hạt giống 42), tương ứng 66.685 dòng dùng cho module khuyến nghị.

| Phần chia | AP dừng (trung bình bốn mốc) | AP 20% | AP 35% | AP 50% | AP 75% |
|---:|---:|---:|---:|---:|---:|
| 0 | 0,8412 | 0,7617 | 0,8145 | 0,8595 | 0,9009 |
| 1 | 0,8433 | 0,7617 | 0,8085 | 0,8402 | 0,8795 |
| 2 | 0,8502 | 0,7455 | 0,8008 | 0,8428 | 0,8867 |

Bảng 4.1. AP trên tập kiểm định theo phần chia, OULAD 20–75%, hạt giống 42.

![AP theo phần chia](figures/fig10_oulad_fold_stop_ap.png)

Hình 4.2. Ngưỡng quyết định trên tập dừng thay đổi theo phần chia và theo mốc, ví dụ tại 20% nhận các giá trị 0,18; 0,13; 0,13 và tại 75% nhận 0,49; 0,52; 0,27. Không có một ngưỡng dùng chung cho mọi mốc.

![Ngưỡng theo phần chia](figures/fig11_stop_threshold_by_fold.png)

| Mốc | AP kiểm định | Độ lệch chuẩn | AP huấn luyện | Khoảng cách | Mức |
|---|---:|---:|---:|---:|---|
| UCI S0 | 0,4547 | 0,043 | 0,5801 | 0,1254 | Cao |
| UCI S1 | 0,8214 | 0,034 | 0,8566 | 0,0352 | Trung bình |
| UCI S2 | 0,9101 | 0,022 | 0,9304 | 0,0203 | Trung bình |
| OULAD 20% | 0,7624 | 0,007 | 0,7963 | 0,0339 | Thấp |
| OULAD 35% | 0,8058 | 0,004 | 0,8371 | 0,0312 | Thấp |
| OULAD 50% | 0,8483 | 0,007 | 0,8722 | 0,0238 | Thấp |
| OULAD 75% | 0,8885 | 0,008 | 0,9088 | 0,0203 | Thấp |
| OULAD 100% | 0,9204 | 0,006 | 0,9359 | 0,0155 | Thấp |

Bảng 4.2. Khoảng cách AP giữa tập huấn luyện và tập kiểm định trên chín lần chạy. Mức cao khi khoảng cách không nhỏ hơn 0,10 hoặc độ lệch chuẩn AP không nhỏ hơn 0,05; mức trung bình khi khoảng cách không nhỏ hơn 0,04 hoặc độ lệch chuẩn không nhỏ hơn 0,02; còn lại là mức thấp.

![Khoảng cách huấn luyện và kiểm định](figures/fig04_overfit_fit_vs_valid.png)

![Trung bình và độ lệch chuẩn](figures/fig18_ap_mean_std_9run.png)

Nhận xét:

Các hình trên cho thấy mô hình học ổn định hơn khi chuỗi đủ dài. Trên OULAD, độ lệch chuẩn AP trên chín lần chạy không vượt 0,008; khoảng cách giữa tập huấn luyện và tập kiểm định giảm từ 0,034 tại 20% xuống 0,016 tại 100%. Trên UCI, S1 và S2 có khoảng cách 0,035 và 0,020. S0 có khoảng cách 0,125, phù hợp với thiết kế Chương 3: chưa có G1 và G2 nên CNN và BiLSTM tắt, tập huấn luyện chỉ khoảng 440 dòng. Đây là hạn chế của thiếu đầu vào chuỗi, không phải căn cứ để thay kiến trúc. Phần chia 2 có AP dừng trung bình 0,8502, cao nhất trong ba phần dùng cho khuyến nghị, nhưng không được chọn lại sau khi nhìn kết quả kiểm định tại 100%. Không kết luận hội tụ theo epoch vì trọng số đã lưu không chứa lịch sử vòng lặp.

### 4.2.3. Kết quả hiệu suất tổng thể

Sau chín lần chạy, hiệu suất Hybrid CNN–BiLSTM được tổng hợp trên từng mốc. Mỗi miền dùng một mô hình cho mọi mốc thông tin.

Hybrid CNN–BiLSTM trên UCI (tỷ lệ lớp dương 0,220, nhãn G3 nhỏ hơn 10):

| Mốc | Độ chính xác | AP | Precision | F1 | Recall | ECE |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0,5213 | 0,4547 | 0,2911 | 0,4291 | 0,8421 | 0,254 |
| S1 | 0,8553 | 0,8214 | 0,6604 | 0,6899 | 0,7587 | 0,129 |
| S2 | 0,9094 | 0,9101 | 0,7654 | 0,8010 | 0,8545 | 0,117 |

Bảng 4.3. Hybrid CNN–BiLSTM trên UCI, trung bình chín lần chạy. Từ S0 đến S1, AP tăng 0,3667; từ S1 đến S2 tăng 0,0887.

Hybrid CNN–BiLSTM trên OULAD (nhãn Fail hoặc Withdrawn):

| Mốc | Độ chính xác | AP | Precision | F1 | Recall | ECE | Số mẫu |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20% | 0,6862 | 0,7624 | 0,6033 | 0,6781 | 0,7769 | 0,069 | 26.697 |
| 35% | 0,7435 | 0,8058 | 0,6613 | 0,7001 | 0,7464 | 0,057 | 25.606 |
| 50% | 0,8001 | 0,8483 | 0,7445 | 0,7306 | 0,7207 | 0,030 | 24.599 |
| 75% | 0,8628 | 0,8885 | 0,8516 | 0,7807 | 0,7221 | 0,027 | 23.159 |
| 100% | 0,9034 | 0,9204 | 0,9048 | 0,8372 | 0,7807 | 0,020 | 22.522 |

Bảng 4.4. Hybrid CNN–BiLSTM trên OULAD, trung bình chín lần chạy. Từ 20% đến 100%, AP tăng 0,1580. Precision tăng từ 0,603 lên 0,905.

![AP trên UCI](figures/fig01_uci_ap_serving.png)

Hình 4.5. AP của Hybrid CNN–BiLSTM trên UCI theo mốc thông tin, kèm các mô hình đối sánh.

![AP trên OULAD](figures/fig02_oulad_ap_serving.png)

Hình 4.6. AP của Hybrid CNN–BiLSTM trên OULAD theo mốc cắt, một mô hình cho năm mốc.

![Năm chỉ số UCI](figures/fig05_uci_hybrid_five_metrics.png)

Hình 4.7. Độ chính xác, AP, Precision, F1 và Recall của Hybrid CNN–BiLSTM trên UCI tại ngưỡng đã chọn.

![Đường OULAD](figures/fig06_oulad_hybrid_curves.png)

Hình 4.8. Năm chỉ số của Hybrid CNN–BiLSTM trên OULAD theo mốc cắt.

![Tăng thông tin](figures/fig03_information_growth_ap.png)

Hình 4.9. Mức tăng AP khi thêm thông tin, cùng một mô hình Hybrid CNN–BiLSTM.

![ECE](figures/fig07_hybrid_ece.png)

Hình 4.10. Sai số hiệu chỉnh kỳ vọng. S0 bằng 0,254; OULAD 100% bằng 0,020.

Nhận xét:

Biểu đồ cột và đường cho thấy AP tăng khi lượng thông tin tăng: UCI có G1 rồi G2; OULAD có thêm tuần tương tác. Sai số hiệu chỉnh giảm từ 0,254 tại S0 xuống 0,020 tại OULAD 100%. S1, S2 và các mốc từ 35% trở đi là những thời điểm kiến trúc lai được thiết kế để phát huy. S0 chưa có chuỗi điểm nên CNN và BiLSTM tắt, không lấy S0 làm kết luận chính về kiến trúc.

Trên UCI, AP tại S1 đạt 0,8214, cao hơn hồi quy logistic 0,042, rừng ngẫu nhiên 0,032 và XGBoost 0,044 (kiểm định Wilcoxon so với hồi quy logistic và rừng ngẫu nhiên: p = 0,0039, thắng 9/9 lần chạy). Tại S2, AP đạt 0,9101, cao hơn hồi quy logistic 0,029 và XGBoost 0,014 (p so với hồi quy logistic = 0,0078). Trên OULAD, AP lần lượt 0,8058; 0,8483; 0,8885; 0,9204 từ 35% đến 100%, cao hơn hồi quy logistic và rừng ngẫu nhiên trên cùng quy trình (p = 0,0039, trừ 75% so với rừng ngẫu nhiên p = 0,055 nhưng điểm Hybrid vẫn cao hơn). So với XGBoost, chênh lệch AP trên 35–100% nằm trong khoảng ±0,002.

S0 và OULAD 20% là mốc thiếu chuỗi hoặc tín hiệu tuần còn mỏng.

| Mốc | Hybrid | LR | DT | RF | SVM | MLP | XGB |
|---|---:|---:|---:|---:|---:|---:|---:|
| UCI S0 | 0,4547 | 0,4754 | 0,4169 | 0,4995 | 0,4970 | 0,4486 | 0,4823 |
| UCI S1 | 0,8214 | 0,7794 | 0,7330 | 0,7895 | 0,7936 | 0,7595 | 0,7774 |
| UCI S2 | 0,9101 | 0,8812 | 0,8547 | 0,9072 | 0,8866 | 0,8778 | 0,8965 |
| OULAD 20% | 0,7624 | 0,7632 | 0,7084 | 0,7522 | 0,7534 | 0,6799 | 0,7663 |
| OULAD 35% | 0,8058 | 0,7986 | 0,7548 | 0,7940 | 0,7835 | 0,7388 | 0,8065 |
| OULAD 50% | 0,8483 | 0,8399 | 0,7954 | 0,8402 | 0,8257 | 0,7998 | 0,8460 |
| OULAD 75% | 0,8885 | 0,8828 | 0,8530 | 0,8847 | 0,8723 | 0,8556 | 0,8902 |
| OULAD 100% | 0,9204 | 0,9114 | 0,8862 | 0,9154 | 0,9018 | 0,8964 | 0,9183 |

Bảng 4.5. AP trung bình chín lần chạy của Hybrid CNN–BiLSTM và các mô hình đối sánh.

![Đối sánh UCI](figures/fig08_parity_uci_ap.png)

Hình 4.11. AP trên UCI theo mốc thông tin.

![Đối sánh OULAD](figures/fig09_parity_oulad_ap.png)

Hình 4.12. AP trên OULAD theo mốc cắt.

Để kiểm tra vai trò từng thành phần, đề tài huấn luyện lại trên một mốc điển hình với các biến thể: chỉ nhánh bảng, chỉ CNN, chỉ BiLSTM. Trên OULAD 35%, mô hình đầy đủ đạt AP 0,809, cao hơn chỉ nhánh bảng (0,804), chỉ BiLSTM (0,785) và chỉ CNN (0,774). Trên UCI S1, mô hình đầy đủ đạt 0,799, cao nhất trong các biến thể đã thử. Trên UCI S0, mô hình đầy đủ gần bằng nhánh bảng, đúng với việc CNN và BiLSTM bị tắt. Số liệu công bố chính vẫn là kết quả huấn luyện hỗn hợp mốc ở Bảng 4.3 và Bảng 4.4.

Phân tích ý nghĩa kết quả:

- Trên các mốc đã có chuỗi, Hybrid CNN–BiLSTM đạt AP 0,821 tại S1 và 0,910 tại S2; trên OULAD từ 35% trở đi AP tăng từ 0,806 lên 0,920 trên cùng một mô hình. Kiểm định Wilcoxon ủng hộ sự khác biệt so với hồi quy logistic và rừng ngẫu nhiên.
- Mức tăng AP từ S0 lên S1 bằng 0,367 trùng với thời điểm G1 vào chuỗi và CNN, BiLSTM được bật, cho thấy đóng góp của phần thời gian chứ không chỉ của hồi quy trên đặc trưng tĩnh.
- AP trên OULAD tăng 0,158 từ 20% đến 100% trên một mô hình, nên không cần huấn luyện mô hình riêng cho từng mốc cắt.
- S0 và 20% không phải kết luận chính. Mốc 100% không dùng cho cảnh báo sớm và không đưa vào module khuyến nghị.
- MAE không dùng vì đây là phân lớp nhị phân. AP trên UCI và OULAD không so trực tiếp với nhau.

### 4.2.4. Trực quan hóa kết quả dự báo

Xác suất trên 66.685 dòng của các mốc 20–75% (ba phần chia, hạt giống 42) được dùng cho module khuyến nghị. Tập này không chứa nhãn gốc nên không vẽ ma trận nhầm lẫn từ chính file đó.

![Phân bố xác suất](figures/fig12_oof_score_hist.png)

Hình 4.13. Phân bố xác suất Hybrid CNN–BiLSTM trên tập kiểm định. Đường đứt là trung vị ngưỡng của từng mốc.

![Xác suất và độ bất định](figures/fig13_p_vs_entropy.png)

Hình 4.14. Quan hệ giữa xác suất và entropy nhị phân, mẫu 8.000 điểm.

Nhận xét:

Từ histogram, phân bố xác suất dịch theo mốc cắt: tại 20% tập trung thấp hơn tại 75%, cùng chiều với AP tăng ở mục 4.2.3. Entropy cao quanh xác suất 0,5, đúng vùng module khuyến nghị chuyển sang rà soát thủ công khi độ bất định lớn hoặc biên so với ngưỡng mỏng. Cùng một xác suất có thể cho nhãn vận hành khác nhau giữa các phần chia vì ngưỡng phụ thuộc phần chia (Hình 4.2). Không suy ra mô hình tách lớp hoàn hảo từ histogram không nhãn.

### 4.2.5. Trọng số cổng theo mốc quan sát

Khối lượng softmax trung bình của ba nhánh được gộp theo mốc trên chín lần chạy. Đây là cách đọc hành vi của kiến trúc lai: cổng học khi nào dùng nhánh nào. Kết quả không thay thế phân tích đóng góp từng điểm dữ liệu.

![Cổng theo mốc](figures/gate_weights_by_cutoff.png)

Hình 4.15. Khối lượng trung bình của nhánh bảng, CNN và BiLSTM theo mốc.

| Tập dữ liệu | Mốc | Nhánh bảng | CNN | BiLSTM |
|---|---|---:|---:|---:|
| UCI | S0 | 1,000 | 0,000 | 0,000 |
| UCI | S1 | 0,064 | 0,263 | 0,673 |
| UCI | S2 | 0,057 | 0,250 | 0,693 |
| OULAD | 20% | 0,315 | 0,232 | 0,453 |
| OULAD | 35% | 0,272 | 0,245 | 0,483 |
| OULAD | 50% | 0,232 | 0,251 | 0,517 |
| OULAD | 75% | 0,200 | 0,251 | 0,549 |
| OULAD | 100% | 0,172 | 0,237 | 0,591 |

Bảng 4.6. Khối lượng cổng trung bình trên chín lần chạy.

Nhận xét:

Tại UCI S0, nhánh bảng nhận toàn bộ khối lượng, CNN và BiLSTM bằng 0, đúng thiết kế Chương 3. Khi có G1 và G2, khối lượng dồn sang BiLSTM khoảng 0,67–0,69. Trên OULAD, nhánh bảng giảm từ 0,315 xuống 0,172 khi mốc tăng; BiLSTM tăng từ 0,453 lên 0,591. CNN ổn định quanh 0,23–0,25. Cổng không sụp về một nhánh trên OULAD, nhưng BiLSTM chiếm phần lớn khi chuỗi đủ dài. Kết quả này phù hợp giả thuyết H3.

---

## 4.3. Kết quả module khuyến nghị

Module khuyến nghị xếp hạng hành động hỗ trợ trên xác suất của Hybrid CNN–BiLSTM. Việc đánh giá thực hiện trên tập độc lập, không dùng để hiệu chỉnh module.

### 4.3.1. Tập đánh giá

Tập độc lập gồm 632 tình huống, 150 sinh viên, 2.398 lượt đánh giá, không có tình huống bỏ qua. Bốn trạng thái phát hành gồm đề xuất hành động, rà soát thủ công, không đủ bằng chứng và không còn hành động khả thi.

### 4.3.2. Chỉ số xếp hạng

| Phương án | NDCG@3 | P@1 | MRR | R@3 | Hành động không hợp lệ | Số hành động đứng đầu khác nhau |
|---|---:|---:|---:|---:|---:|---:|
| Module khuyến nghị | 0,88785 | 0,99206 | 0,99603 | 0,79947 | 0 | 5 |
| Quy tắc xếp hạng B1 | 0,86649 | 0,99683 | 0,99841 | 0,80357 | 0 | 4 |
| Quy tắc xếp hạng B0 | 0,81889 | 0,99365 | 0,99683 | 0,78981 | 0 | 2 |

Bảng 4.7. Kết quả trên 632 tình huống độc lập. Chênh lệch NDCG@3 của module khuyến nghị so với B1 là 0,02131, khoảng tin cậy 95% [0,01440; 0,02815] (bootstrap 2.000). So với B0: 0,06885, khoảng tin cậy [0,06051; 0,07748].

![Chỉ số xếp hạng](figures/fig14_rec_panel_c_metrics.png)

Hình 4.16. NDCG@3, Precision tại 1 và Recall tại 3 trên tập độc lập.

![Bootstrap](figures/fig17_rec_bootstrap_ndcg.png)

Hình 4.17. Phân bố chênh lệch NDCG@3, bootstrap 2.000.

### 4.3.3. Trạng thái phát hành và hành động đứng đầu

Trên 632 tình huống: đề xuất 94 (14,9%), rà soát thủ công 175 (27,7%), không đủ bằng chứng 363 (57,4%), không còn hành động khả thi 0.

![Trạng thái](figures/fig15_rec_routes.png)

Hình 4.18. Phân bố trạng thái phát hành.

Hành động đứng đầu: phục hồi tương tác 111, luyện tập truy xuất 64, ôn nội dung trọng tâm 36, duy trì nhịp học 31, hoàn thành bài đánh giá 27.

![Hành động đứng đầu](figures/fig16_rec_top1_actions.png)

Hình 4.19. Phân bố năm hành động ở vị trí đứng đầu.

Nhận xét:

Module khuyến nghị đạt NDCG@3 bằng 0,888 và không phát hành hành động vi phạm luật khả thi. Tỷ lệ không đủ bằng chứng 57,4% phản ánh cơ chế an toàn khi nhật ký tương tác còn mỏng hoặc xác suất dưới ngưỡng, không phải lỗi xếp hạng. Kết quả không được đọc như bằng chứng can thiệp làm thay đổi kết quả cuối môn. Mốc 100% không đưa vào module khuyến nghị.

### 4.3.4. Minh họa một tình huống

Với một lượt ghi danh OULAD tại mốc 20%, mô hình dự đoán cho xác suất 0,258, ngưỡng 0,18, nhãn vận hành dương và độ bất định 0,823. Module khuyến nghị chuyển sang rà soát thủ công vì độ bất định cao, phù hợp quy tắc đã mô tả ở Chương 3.
