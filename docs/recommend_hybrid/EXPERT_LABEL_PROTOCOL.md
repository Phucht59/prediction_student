# recommend_hybrid optional expert-evaluation protocol

## Current status

Expert evaluation is `OPTIONAL` and the Phase 2 pipeline is retained as `FUTURE_EXTENSION`. Expert labels are unavailable; real reviewers, completed case reviews and action ratings are all zero. Recommendation training is `NOT_APPLICABLE`, and Phase 3 is not blocked by expert labels.

The existing pilot has 60 cases and two blank independent-reviewer templates. If future expert evaluation is authorized, its minimum intended case-review volume is 60 × 2 = 120 independent case reviews. Templates remain unmodified and contain no fabricated rating.

## Preserved future protocol

Cases remain blinded to identity, outcome, model internals and exact probability. Each future action rating uses the ordinal scale 3, 2, 1, 0, -1 and requires action/case/expert identity, approval, missing-action, safety, escalation, reason and comment fields. An unsafe -1 rating requires a safety concern and adjudication.

The importer continues to validate approved expert IDs, known case/action pairs, completeness, duplicate records, score/status vocabularies, safety consistency and raw-file immutability. These records may support later optional evaluation, but must not be converted into pseudo-labels or presented as current evidence.

## Phase 3 boundary

The evidence-based policy uses versioned domain rules, observed pre-cutoff evidence and frozen CNN-BiLSTM prediction context. It does not import expert files, train a ranker, compute expert approval rate, or claim user satisfaction/causal effectiveness.
