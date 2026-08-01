# Dự đoán thành tích và cảnh báo sớm sinh viên

Repository khóa luận:

> **Xây dựng mô hình học kết hợp để dự đoán thành tích học tập sinh viên**

## Mô hình dự đoán cuối đã khóa

| Dataset | Mô hình cuối | Bài toán | Macro-F1 |
|---|---|---|---:|
| Student-Mat | CNN-BiLSTM | Phân loại Low / Medium / High | **0.901460** |
| Student-Por | CNN-BiLSTM | Phân loại Low / Medium / High | **0.862259** |
| OULAD | H1 Tabular Residual CNN-BiLSTM | At-risk / Not-at-risk | **0.894071** |

Các kết quả trên là **final authority** của release hiện tại. Những kết quả OULAD
`0.828084` và `0.798400` chỉ còn giá trị lịch sử, không phải kết quả cuối dùng
trong báo cáo hoặc demo.

## Vì sao chốt các mô hình này?

- Hai bộ Student-Mat và Student-Por giữ CNN-BiLSTM đã đóng băng, đúng mục tiêu
  nghiên cứu mô hình học kết hợp trên dữ liệu điểm sinh viên.
- OULAD dùng H1 Tabular Residual CNN-BiLSTM theo giao thức
  `STRICT_REAL_TIME`: loại toàn bộ score và aggregate suy ra từ score không xác
  minh được thời điểm công bố.
- OULAD kết hợp chuỗi hành vi theo thời gian với đặc trưng aggregate/static,
  dùng 3 outer folds và 5 seed cố định; threshold được chọn chỉ từ inner OOF.
- OULAD đạt Macro-F1 0.894071, gần tương đương MLP 0.895349. Vì vậy khóa luận
  không tuyên bố hybrid luôn thắng mọi mô hình ML; giá trị chính là kiến trúc
  kết hợp, khả năng cảnh báo theo giai đoạn và tính hợp lệ thời gian của feature.

## Hệ thống khuyến nghị cuối

Tên chính thức: **Hybrid CNN-BiLSTM Learning Support Recommender**  
Tên tiếng Việt: **Hệ thống khuyến nghị hỗ trợ học tập dựa trên CNN-BiLSTM hybrid**.

Đây không phải một mạng nơ-ron thứ hai được train để đoán hành động. Hệ thống sử
dụng kết quả dự đoán đã đóng băng của CNN-BiLSTM/H1, kết hợp với bằng chứng học
tập quan sát được trước cutoff và một policy xác định để tạo kế hoạch hỗ trợ có
thể kiểm tra lại.

```text
Dự đoán thành tích/rủi ro đã đóng băng
  + bằng chứng học tập trước cutoff
  + độ bất định và lineage
  -> policy riêng cho UCI hoặc OULAD
  -> kiểm tra điều kiện, xung đột và chống chỉ định
  -> chọn tối đa 4 hành động, tối đa 180 phút mỗi giai đoạn
  -> LearningPlan: FULL / PARTIAL / ABSTAIN / EVALUATION_ONLY
```

Hai nhánh nghiệp vụ được tách biệt:

- **RecommendHybridUCI**: Student-Mat và Student-Por tại S0, S1, S2; G3 bị cấm.
- **RecommendHybridOULAD**: định tuyến request 20–99% về mốc dự báo hợp lệ gần
  nhất trong quá khứ; request trước 20% phải abstain và FINAL chỉ dùng đánh giá.

Hệ thống không dùng sensitive attributes, future outcome, outer-test label hoặc
evidence sau cutoff. Không có neural action ranker, pseudo-label, random action
hay mô hình ước lượng hiệu quả can thiệp.

### Kết quả kỹ thuật khuyến nghị

- Final gate: `RECOMMEND_HYBRID_PHASE5_FINAL_PASS`.
- Safety và constraint violations: **0**.
- Evidence support và explanation lineage: **100%**.
- Deterministic replay và plan-hash match: **100%**.
- Bộ đánh giá kỹ thuật: 260 records; actionable coverage trên 240 record đủ điều
  kiện can thiệp là **92.08%**, abstention **7.92%**.
- Toàn bộ artefact release: 15,378 risk profile/plan; GENERATED 10,953,
  PARTIAL_EVIDENCE 1,209 và ABSTAINED 3,216, tương ứng generated-or-partial
  coverage **79.09%**.
- Trạng thái đánh giá chuyên gia: `PENDING_EXPERT_LABELS`.
- Không tuyên bố hệ thống đã được chứng minh làm tăng điểm hoặc có hiệu quả nhân
  quả.

## Kiến trúc tổng quát

```text
Dữ liệu sinh viên
  -> kiểm tra schema và thời điểm khả dụng
  -> tiền xử lý chỉ fit trên tập train
  -> nhánh temporal CNN-BiLSTM
  -> nhánh aggregate/static residual
  -> fusion và classifier
  -> xác suất rủi ro / mức thành tích
  -> risk profile
  -> policy khuyến nghị và constraint solver
  -> kế hoạch hỗ trợ học tập
  -> PostgreSQL và evidence kiểm chứng
```

### UCI Student Performance

- Nhãn: `Low`, `Medium`, `High` từ điểm cuối kỳ `G3`.
- `G3` chỉ dùng tạo nhãn, tuyệt đối không dùng làm predictor.
- Mô hình cuối: CNN-BiLSTM riêng cho Student-Mat và Student-Por.

### OULAD

- Nhãn: `At-risk` và `Not-at-risk`.
- 47 kênh temporal và 165 đặc trưng aggregate.
- Score values và score-derived aggregates bị loại khỏi strict protocol.
- Các mốc đánh giá: 20%, 35%, 50%, 75% và FINAL.
- Mô hình cuối có 160,492 tham số và topology đã đóng băng.

## Nguồn sự thật

- `configs/final/final_model_authority.yaml`: mô hình dự đoán và metric cuối.
- `configs/final/model_registry.yaml`: ba model ID dự đoán được công khai.
- `configs/final/recommendation.yaml`: authority của hệ thống khuyến nghị.
- `src/recommend_hybrid/`: policy, routing và constrained plan builder.
- `docs/recommend_hybrid/MODEL_CARD.md`: model card và giới hạn tuyên bố.
- `reports/recommend_hybrid/FINAL_VALIDATION.md`: validation cuối của recommender.
- `reports/final/RECOMMENDATION_RESULTS.md`: tổng hợp artefact khuyến nghị.
- `artifacts/final_release/`: metric replay, registry và checksum release.
- `artifacts/canonical_v3/`: evidence OULAD strict real-time.
- `artifacts/final/models/`, `artifacts/final/metrics/`: checkpoint và metric UCI.
- `reports/final/thesis_v3/`: 12 báo cáo cuối dùng để viết và bảo vệ khóa luận.

## Chạy kiểm tra release

```powershell
python project.py final status
python project.py final report
python project.py final validate
python scripts/final/validate_final_release.py
python scripts/recommend_hybrid/validate_phase5.py
pytest tests/audit/test_final_release.py
pytest tests/recommend_hybrid/phase5 -q
```

Các lệnh trên kiểm tra hoặc replay evidence; không tự động train lại mô hình.

## Cấu trúc chính

```text
configs/final/             cấu hình và model authority cuối
artifacts/final_release/   registry, metric replay và checksum
artifacts/canonical_v3/    evidence OULAD strict real-time
artifacts/final/           checkpoint, metric và evidence UCI/recommendation
data/                      manifest và dữ liệu được phép công khai
database/final/            schema và migration PostgreSQL
docs/recommend_hybrid/     kiến trúc, model card và data contract khuyến nghị
reports/final/thesis_v3/   bộ báo cáo khóa luận cuối
reports/recommend_hybrid/  validation khoa học của hệ thống khuyến nghị
scripts/final/             script replay và validation release
src/recommend_hybrid/      pipeline khuyến nghị UCI/OULAD
src/                       data, model, pipeline và database
tests/                     kiểm thử release và chức năng
```

## Nguyên tắc khoa học

- Không chọn seed tốt nhất từ outer test.
- Không tune threshold trên outer test.
- Không đưa feature tương lai vào thời điểm cảnh báo sớm.
- Không thay đổi metric đã đóng băng khi dọn repository.
- Không gọi coverage của recommender là accuracy.
- Không tuyên bố hiệu quả giáo dục khi chưa có nhãn chuyên gia và outcome thực tế.
- Kết quả lịch sử vẫn tồn tại trong Git history, không nằm trong release tree.
