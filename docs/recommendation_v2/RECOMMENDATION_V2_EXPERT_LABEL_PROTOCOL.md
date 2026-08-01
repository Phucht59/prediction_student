# Recommendation V2 expert-label protocol

## Current status

`expert_status = PENDING_REAL_EXPERT_LABELS` and `recommendation_training_status = BLOCKED`. Existing files prepare 60 cases and two reviewer-specific templates, but zero reviewers have submitted ratings and zero cases are scored. A template is not an expert label.

## Unit of rating

Blind reviewers to outcome, model identity, exact probability, internal record ID and sensitive demographics. Show the same cutoff-safe evidence summary, stage, uncertainty band and complete candidate catalog for each case. Randomize candidate order independently from ranker order. Preserve a hidden case/action key for reconciliation.

Each rating contains:

```text
case_id
action_id
expert_id
relevance_score
approval_status
missing_action
safety_concern
escalation_required
reason_support
comment
```

Required validation: registered case/action/expert IDs; one rating per `(case_id, action_id, expert_id)`; no missing relevance; explicit safety/escalation booleans; non-empty comment for unsafe, rejected or missing-action reports; timestamps and protocol/catalog versions stored separately in the submission manifest.

## Relevance scale

- `3` = Rất phù hợp / highly suitable
- `2` = Phù hợp / suitable
- `1` = Có thể cân nhắc / may be considered
- `0` = Không phù hợp / unsuitable
- `-1` = Không an toàn / unsafe

This ordinal scale is suitable for pairwise/listwise ranking and is more useful than the existing binary action relevance template. It must coexist with, not replace, the current plan-level 1–5 score if thesis reporting still requires that metric. `-1` must also force `safety_concern=true`; it is not merely a lower relevance grade. `approval_status` should use `APPROVED`, `MODIFIED`, `REJECTED`, `NEEDS_MORE_EVIDENCE` for plan decisions; action-level disposition should be separately validated as include/modify/exclude if needed.

## Sampling and reviewer design

Stratify cases by stage, predicted class/risk band, uncertainty/abstention, module/presentation and observed-evidence completeness. Split all records for a student together. Include enough overlap for at least two independent reviewers per adjudication subset; do not disclose each other's scores. Record expertise and conflicts of interest without placing identity attributes in ranker features.

Quality gates before training:

1. At least two real reviewers and the approved minimum number of fully rated overlapping cases.
2. Schema, completeness, duplicate and blinding checks pass.
3. Weighted kappa/Krippendorff alpha for ordinal relevance and agreement for unsafe/escalation flags are reported; threshold is chosen by the architecture owner before viewing final model results.
4. Disagreements involving `-1`, safety or escalation are adjudicated, never averaged away.
5. Missing-action text is mapped only to a catalog review queue; it does not automatically create an action or label.
6. Raw and adjudicated label manifests are immutable, checksummed and versioned.

## Training use

Phase 3 may train a ranker on real ratings using student-grouped train/validation/test splits. Preserve ordinal information through graded relevance or preference pairs; never convert current rules, ranker predictions or model risk labels into expert labels. Compare against the governed rule baseline and report NDCG@K, Precision@K, Recall@K, coverage, unsafe-action rate and reviewer-stratified uncertainty.

No labels are created in Phase 1.
