# Module khuyến nghị (serving)

Bản khóa: `src/recommend_hybrid/serving/`.

Hybrid CNN–BiLSTM cung cấp `p`. Rec học nút thắt còn kéo dài 14 ngày (ASSESS / ENGAGE / COUNSEL), rồi phát hành trên hàng đợi top-K theo `p`.

```python
from src.recommend_hybrid.serving import PersistencePipeline
```

Luồng:

1. Hybrid khóa → `PredictionResult`
2. Top 10% theo `p` trong đợt (module × presentation × mốc)
3. Lọc khả thi cứng
4. Phân loại tồn tại 14 ngày
5. Lộ trình `Q_τ` = TMA/quiz còn hạn
6. `ACTION` / `QUEUE` / `COUNSEL` / `OUT_OF_BUDGET`

Nhãn học là log OULAD sau cutoff, không phải kết quả cuối môn. Artifact: `artifacts/recommend_hybrid/serving/`.

Bản Five-EBM cũ nằm ở `test_lab/archived_from_release/recommend_hybrid_v3`.
