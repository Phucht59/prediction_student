# Ordinal Model V3 protocol

V3 evaluates whether ordinal supervision and train-only continuous-G3 auxiliary supervision improve ordered three-class prediction. It does not attempt to force a deep model above the deterministic G2 rule.

The future full run is fixed to the shared five outer folds, three inner folds, seeds 42/52/62/72/82, inner mean Macro-F1 selection, and outer-validation inference once. Track A uses late-stage `[G1,G2]`; Track B uses early-warning `[G1]`. Pre-assessment and all-feature tracks are excluded. Legacy-79 identities must be independently disjoint from the development manifest.

M0/M1 share backbone capacity and differ only in nominal versus rank-consistent ordinal head/loss. M2/M3 share the same multi-task backbone/search space and differ only in ordinal versus nominal classification. M4 is the fixed S3 sequence backbone with an ordinal head and remains a research comparator. Continuous-G3 models form a separate target-supervision comparison group.

Full V3 is not authorized by this phase. Only one-fold/one-seed late-stage smoke is permitted.
