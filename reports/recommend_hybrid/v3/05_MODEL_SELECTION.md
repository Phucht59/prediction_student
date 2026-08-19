# Block C — model selection

**STATUS: PASS**

Diagnosis: Five-EBM-C0 already has high NDCG, five distinct top-1 actions on the portable slice, and no tiny-margin collapse that would justify a residual reranker.

```text
FINAL_CANDIDATE = Five-EBM-C0
challenger_trained = false
```

Simplicity wins. Artifact: `artifacts/recommend_hybrid/v3/challenger/SELECTION.json`.
