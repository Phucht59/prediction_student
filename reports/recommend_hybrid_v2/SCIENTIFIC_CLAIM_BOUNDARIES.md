# Scientific Claim Boundaries & Methodological Safeguards

## Important Disclaimer
> **"This is an internal model plausibility check, not evidence of causal intervention effectiveness."**

## Core Scientific Boundaries

1. **Frozen Hybrid Model Role**:
   - The Hybrid CNN–BiLSTM is a frozen risk prediction model.
   - It outputs risk probability P(at-risk), but does **NOT** prove causal mechanisms.

2. **Weak Supervision & Pseudo-Expert Labels**:
   - LLM ratings and Snorkel LabelModel outputs are **probabilistic silver labels** derived via weak supervision.
   - They do not represent prospective randomized control trial (RCT) evidence.

3. **Five-EBM Ranker Boundaries**:
   - Five-EBM learns rank relevance from weak supervision targets.
   - Offline ranking metrics (e.g. NDCG@3) evaluate ranking alignment with silver labels, not guaranteed real-world academic gain.

4. **Plausibility Simulator Boundaries**:
   - The simulator calculates **model-implied risk delta** by modifying pre-cutoff raw features.
   - It serves as an internal plausibility sanity check, **NOT** prospective causal intervention proof.

5. **Deployment Authority**:
   - `runtime_authorized` remains strictly `FALSE`. Prospective human expert trial or clinical evaluation is required before real-world deployment.
