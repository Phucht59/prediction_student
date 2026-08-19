# Recommendation V3

**Development authority:** Five-EBM-C0 on Phase4 Hybrid C0 OOF  
**Held-out authority:** pending Panel C Gemini (not run)

```text
Phase4 Hybrid C0
    → risk_probability + STOP threshold + H2(p)
    → cutoff-safe OULAD 20/35/50/75 evidence
    → C0-aligned risk router
    → hard feasibility
    → Five-EBM-C0
    → RECOMMEND Top-1 or HUMAN_REVIEW Top-3
    → deterministic personalized plan
```

UCI is out of scope. OULAD 100% is non-intervention. Panel B was not used for tuning. No prediction HPO. No Gemini runtime.

Development portable NDCG@3 = 0.961 (179 queries). Invalid-action rate = 0 on that slice.
