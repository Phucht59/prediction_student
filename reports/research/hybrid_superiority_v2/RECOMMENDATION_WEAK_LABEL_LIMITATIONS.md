# Recommendation — weak label limitations

Gemini-derived labels là **weak supervision**, không phải chuyên gia độc lập.

## Cấm

- Gán target Risk/Non-Risk bằng Gemini
- Chọn kiến trúc Hybrid bằng Gemini score
- Gọi NDCG trên consensus Gemini là expert validation
- Tuyên bố khuyến nghị cải thiện điểm (không có intervention/user study)
- Gửi PII (tên, MSSV, email)
- Vượt 500 request/model/ngày (safe cap 480)

## Claim hợp lệ khi chưa có expert gold

> Hệ thống xếp hạng hành động hỗ trợ theo weak supervision, đo consistency/stability/feasibility và abstain khi bằng chứng không đủ.

Provenance V3 (không phải authority mới): ~632 student-stage cases, ~2398 Gemini reviews, weak-label NDCG@3 ≈ 0.88785, exact-best Top-1 ≈ 0.407, 94 RECOMMEND / 175 HUMAN_REVIEW / 363 INSUFFICIENT_EVIDENCE.

Hai alias người dùng cấp: Gemini 3.5 Flash Lite và Gemini 3.1 Flash Lite, mỗi cái tối đa 500 request/ngày. Model ID chính thức được resolve từ config/API, không đoán.

Prediction gate độc lập hoàn toàn với Gemini.
