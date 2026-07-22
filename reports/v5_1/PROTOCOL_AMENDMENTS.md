# V5.1 Protocol Amendments

## 2026-07-18 — OULAD target-table checksum typo

- Timing: before OULAD component screening and before any OULAD outer-fold evaluation.
- Scope: corrected the frozen SHA-256 text for
  `data/processed/study_c_oulad/targets/F2_MIDDLE.parquet` from a 63-character
  transcription to the observed 64-character digest
  `f5dabf9719c2c8ce038e2bca89d11da82508eee2ca65c92c2bca40d7bad9da80`.
- Evidence: the corrected digest matches the immutable V5 OULAD configuration,
  the on-disk file, and the remaining Study C source hashes all verified without
  changes.
- Scientific impact: none. No dataset, split, feature, target, model, search
  space, seed, or evaluation rule changed. No OULAD trial had started.

## 2026-07-18 — OULAD stage-gated compute reduction

- Timing: after four architecture-screening trials on outer-training fold 0,
  before any OULAD outer-fold result was evaluated.
- Preserved evidence: the existing Optuna database and study name are retained.
  Trials 0–3 remain COMPLETE. Trial 4, which was created during the pause after
  trial 3 completed but produced no result, remains PRUNED with an administrative
  pause reason. No trial number is deleted or reused.
- Screening scope: component screening uses outer-training fold 0 only. The
  architecture stage stops at eight completed trials unless the best inner
  Macro-F1 exceeds 0.8300 and the best trial is within the three most recent
  completed trials; only then may it extend to at most twelve completed trials.
  At most two configurations are confirmed.
- Conditional stages: masked-week pretraining opens only when the confirmed
  architecture reaches inner Macro-F1 0.8305. Pretraining, augmentation, or loss
  changes require a same-fold mean inner improvement of at least 0.001.
  Augmentation and loss screening also require documented training-side evidence
  of overfit or class imbalance and use at most four registered configurations.
- Focused stage: the former 60–120-trial-per-fold search is removed. A single
  outer-training-fold-0 study runs 16 total trials and may extend to at most 24
  only while recent improvement continues. Its locked configuration is reused
  without retuning on all three outer folds.
- Final fairness: the selected full hybrid is evaluated on all three immutable
  outer folds and all five registered seeds. CNN-only and BiLSTM-only are each
  evaluated once per outer fold with seed 42. Eligible V5 XGBoost/MLP evidence is
  reused only after input-contract and checksum verification.
- Scientific impact: this amendment reduces compute before viewing any OULAD
  outer result. It does not change the target, feature cutoff, split manifest,
  group isolation, primary metric, fixed seeds, threshold scope, or future lock.
- Resume-integrity note: the first restart reused the deterministic sampler seed,
  causing trials 5 and 6 to exactly repeat COMPLETE trials 0 and 1. They remain
  immutable COMPLETE rows but are excluded from the unique-configuration budget.
  Trial 7 matched trial 2 and was stopped and retained as PRUNED before producing
  a result. The study records the mapping `{5: 0, 6: 1, 7: 2}`. Subsequent resumes
  offset the sampler seed by the preserved trial-row count and prune any exact
  parameter duplicate before model fitting. Stage gates count unique COMPLETE
  configurations, so these administrative duplicates cannot shorten or bias the
  registered budget.

## 2026-07-19 — OULAD full-fold screening recovery and hard cap

- Timing: before any OULAD outer-fold result, focused search, or final evaluation
  was accessed. The architecture study had 11 unique COMPLETE configurations and
  a sequence of PRUNED rows that had nevertheless already finished all three
  registered inner folds.
- Root cause: the median pruner received the three fold scores only after all
  fold training had completed. It therefore saved no compute and compared an
  individual fold against its step median instead of comparing the registered
  mean across folds. This could label a configuration PRUNED even when its
  full-fold mean exceeded the best COMPLETE mean.
- Recovery rule: an immutable historical PRUNED row is eligible for architecture
  ranking only when it contains all three inner-fold scores. Its score is the
  arithmetic mean of those three registered values. Partial and administrative
  PRUNED rows remain excluded. Exact parameter duplicates remain de-duplicated.
- Forward rule: architecture pruning is disabled and the 8/12 stage gates count
  unique fully evaluated configurations. This is a hard compute cap. Existing
  fully evaluated evidence is reused without retraining; at most two candidates
  receive the previously registered fixed-seed confirmation.
- Scientific impact: this corrects budget accounting and preserves already-paid
  inner-fold evidence. Target, data, split, metric, search space, registered
  seeds, outer-test isolation, and the Future OULAD lock are unchanged.
