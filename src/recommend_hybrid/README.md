# Recommendation V

Bản phát hành khóa luận: **`src/recommend_hybrid/v3/`** (Five-EBM-C0).

Không dùng `src/recommend_hybrid/final/` làm mô hình cuối. Đó là lineage Panel B cũ (NDCG@3 ≈ 0,953), **không** phải số khóa luận. Serving live đọc V3.

```python
from src.recommend_hybrid.v3.pipeline import RecommendationV3Pipeline
from src.recommend_hybrid.v3.ranker import FiveEBMC0Ranker
from src.recommend_hybrid.v3.contracts import RouteStatus
```

CLI / PostgreSQL: `src/database/live_runtime.py` nạp

- ranker: `artifacts/recommend_hybrid/v3/ranker/final_models/*.joblib`
- router: `artifacts/recommend_hybrid/v3/router/ROUTER_CONFIG.json`
- features: `artifacts/recommend_hybrid/v3/data/learner_stage_features.parquet`
- Hybrid OOF folds (OULAD serving rec): `artifacts/recommend_hybrid/v3/data/c0_inner_fold{0,1,2}_seed42.pt`

Luồng khóa:

1. Hybrid CNN–BiLSTM (`src/prediction/`) → `p, t, ŷ, u`
2. định tuyến rủi ro (`p` vs `t`, `u`, `margin`)
3. luật khả thi cứng (5 hành động)
4. năm EBM độc lập, cùng 17 cột, mục tiêu `expected_relevance / 3`
5. an toàn Top-1
6. `RECOMMEND` / `HUMAN_REVIEW` / `INSUFFICIENT_EVIDENCE` / `NO_FEASIBLE_ACTION`

Không gọi mô hình ngôn ngữ lúc vận hành. Không khuyến nghị UCI. Không khuyến nghị OULAD 100%.

Số khóa Panel C: NDCG@3 0,88785 vs B1 0,86649; invalid = 0.
`artifacts/recommend_hybrid/v3/release/FINAL_RELEASE_MANIFEST.json`
