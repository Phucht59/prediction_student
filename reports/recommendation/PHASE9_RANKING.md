# Phase 9 ranking

Operational contract after frozen Phase 8 models.

- Five EBM models score each case independently.
- Raw EBM scores are stored; operational relevance is clipped to [0, 3].
- Feasibility uses `recommendation.feasibility.v2`. A4 Progress Monitoring is FEASIBLE.
- INFEASIBLE actions are removed from releasable ranking.
- UNKNOWN is `NEEDS_VERIFICATION`, not silently treated as feasible.
- A5 keeps Phase 7 REVIEW. If A5 is in the released top-k, `release_status=REVIEW_REQUIRED` when feasible.
- Top 3 by clipped score, then raw score, then fixed action order.
- No LLM reordering.

Panel B inference (150 cases, not a labeled evaluation):

- A4 feasibility: 150/150 FEASIBLE.
- Plan status: 93 RECOMMEND, 57 REVIEW.
- A5 in top-1: 0.000; A5 in top-3: 0.593.
- A5 release_status: 103 REVIEW_REQUIRED, 47 NEEDS_VERIFICATION.

`PHASE9_DATA` remains blocked until Panel B automated reference jobs are run.
