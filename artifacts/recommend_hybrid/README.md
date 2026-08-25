# Trọng số serving — đừng lấy nhầm thư mục

| Dùng | Không dùng cho khóa luận |
|---|---|
| **`v3/`** Recommendation V, Panel C | `final/` Panel B cũ |
| `v3/ranker/final_models/*.joblib` | `final/ranker/final_models/` |
| `v3/router/ROUTER_CONFIG.json` | `final/router/` |
| `v3/data/c0_inner_fold*_seed42.pt` Hybrid C0 OULAD | checkpoint research/`test_lab/` |

Hybrid kiến trúc: `src/prediction/` + `configs/prediction/hybrid_final.json`.  
Không lấy `artifacts/prediction/final/development/hybrid_config.json` (bản cũ, CNN 128 kênh).
