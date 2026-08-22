# Chương 5. Kết luận, hạn chế và hướng phát triển

## 5.1. Kết luận

Khóa luận đã khóa **Hybrid CNN–BiLSTM** (một kiến trúc, hai miền) và **Recommendation V** (chỉ OULAD 20–75%), đánh giá bằng AP inner 3×3, không dùng outer để chọn mô hình.

Kết quả chính — **ưu thế Hybrid trên các mốc thiết kế** (Chương 4):

- UCI S1 AP **0.8214**, S2 **0.9101**. Wilcoxon: S1 hơn LR và RF (p = 0.0039, 9/9); S2 hơn LR (p = 0.0078). Đây là claim chính của kiến trúc lai khi đã có G1/G2.
- OULAD, **một checkpoint**: AP 0.762 → 0.806 → 0.848 → 0.889 → 0.920. Từ 35% trở đi Hybrid hơn LR và RF có ý nghĩa (p = 0.0039). 20% Hybrid hơn RF (+0.010, p = 0.0039).
- **H3:** cổng chuyển mass từ tabular sang BiLSTM khi chuỗi dài (0.45 → 0.59) — hybrid dùng đúng nhánh.
- **H1 (ablation một mốc):** OULAD 35% full 0.809 vượt tabular/CNN-only/BiLSTM-only; UCI S1 full 0.799 là arm cao nhất.
- Recommendation V: NDCG@3 **0.888** vs B1 0.866 (Δ +0.021, CI không chứa 0); invalid-action 0.

S0 (chưa có chuỗi) và 20% (tín hiệu tuần còn mỏng) không dùng để phủ nhận Hybrid. Claim đặt ở **S1/S2 và OULAD 35–75%**.

## 5.2. Hạn chế — nói thẳng khi bảo vệ

Những điểm sau **không phải gian lận**; chúng là giới hạn đã chọn và phải nêu:

1. **Outer không mở** → chưa có ước lượng test cuối cùng ngoài vòng FIT/STOP/VALID. Outer fold 0 chỉ là firewall.
2. **100% không phải early warning.** Prevalence giảm 0.424 → 0.317 vì enrollment rút trước cutoff bị loại; AP Withdrawn vs Pass chỉ ~0.55–0.61. AP 0.920 tại 100% không dùng cảnh báo sớm; Rec V từ chối 100%.
3. **Recommendation V không phải can thiệp nhân quả.** Không ATE, không RCT, không viết “làm vậy thì đỗ”. Panel C dùng reviewer LLM.
4. **S0 / 20% không phải claim chính.** S0 không có chuỗi (CNN/BiLSTM tắt). 20% là mốc lạnh; ưu thế Hybrid thể hiện từ 35% và từ UCI S1.
5. **Không có đường epoch trên checkpoint khóa** (`state_dict` only). Không vẽ loss giả. Đường epoch (nếu có) thuộc bản research ablation, không thay bản khóa.

Bổ sung:

- Không dữ liệu sinh viên Việt Nam; không xây app/API/giao diện — phạm vi là mô hình.
- G1/G2 trên UCI serving vẫn vào temporal và aggregate; ablation research định lượng việc bỏ trùng.
- Không SHAP từng điểm; XAI trình bày là **mass cổng trung bình theo cutoff**.

Hạn chế trên không phủ nhận ưu thế Hybrid tại S1/S2 và OULAD 35–75%.

## 5.3. Hướng phát triển

- Mở outer **một lần** sau khi đóng băng mọi lựa chọn (cần duyệt).
- Tách mô hình Fail vs Withdrawn.
- Transformer / temporal GNN mask-safe trên OULAD, cùng split và AP.
- Dữ liệu trường Việt Nam.
- (Ngoài phạm vi khóa luận) API/giao diện cảnh báo sớm nếu triển khai sản phẩm.
- Hiệu chỉnh xác suất (ECE S0 còn 0.254).
- Bỏ nhánh aggregate trùng G1/G2 nếu ablation `temporal_only` không kém `both`.
