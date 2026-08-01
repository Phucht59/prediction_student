# recommend_hybrid expert-label protocol

Current status: `PENDING_REAL_EXPERT_LABELS`; Phase 3 training status: `BLOCKED`. Phase 2 exported 60 real-data pilot cases and blank templates for `expert_01` and `expert_02`; reviewer count, scored cases, action ratings and fabricated labels remain zero.

## Blinding and export

Cases are sampled from canonical five-seed hybrid predictions at `MIDDLE_50` and reconstructed from real OULAD events satisfying `event_day < cutoff_day`. HMAC case IDs use an unpersisted 256-bit export secret. Student identity, course identity, fold, seed, checkpoint filename, internal model alias, outer/future label, protected attributes and future outcome are absent.

The approved Phase 1 protocol withholds exact probability, so reviewers receive risk/confidence/uncertainty/disagreement bands. This resolves the broader Phase 2 case description in favor of the locked blinding authority. Candidate order is independently randomized per reviewer. The source export and two reviewer templates are immutable inputs to later real review; templates contain no prefilled ratings.

## Rating schema

Each real action rating requires `case_id`, `action_id`, `expert_id`, `relevance_score`, `approval_status`, `missing_action`, `safety_concern`, `escalation_required`, `reason_support`, and `comment`.

- 3: highly suitable
- 2: suitable
- 1: may be considered
- 0: unsuitable
- -1: unsafe; requires `safety_concern=true` and adjudication

Action approval is `APPROVE`, `PARTIAL`, `UNSURE`, or `REJECT`. Case-plan status is `APPROVED`, `MODIFIED`, `REJECTED`, or `NEEDS_MORE_EVIDENCE`.

## Import and training gate

The importer validates approved experts, known case/action pairs, required fields, score/status vocabularies, duplicate `(case_id, action_id, expert_id)` records, unsafe-score consistency and contradictory approval. It hashes the raw file before and after and writes a separate normalized artifact; raw review files are never edited.

Rules, model predictions, test fixtures and language-model judgments are not expert labels. Training remains blocked until real reviewer submissions, overlap/agreement analysis, safety adjudication and immutable raw/adjudicated manifests pass the later phase gate.
