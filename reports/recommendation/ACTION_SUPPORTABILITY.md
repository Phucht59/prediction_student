# Action supportability decision

## Current A4 candidate status

Progress Monitoring is the current A4 candidate under Gemini-family
revalidation. The previous Gemma source is historical evidence only and is
not reused. Gemini 3.5 labels already exist; Gemini 3.1 labels are prepared
as a second source. Academic Help-Seeking is not used for this validation.

A4 Content Review remains in the A1-A5 action catalog but is locked as `UNSUPPORTED_BY_CURRENT_STATE`.
Reason: `Current Student State lacks observable content-level evidence.`

This is an empirical supportability/data-observability decision: the current Student State does not expose content-level evidence. It is not model-performance cherry-picking and does not remove an action because of an agreement score.

## Evidence

- Gemini A4 ABSTAIN: `500/500` unique Panel A cases.
- Gemma completed A4 ABSTAIN: `484/484` completed cases.
- Gemma raw failed records recovered offline: `16`; their supported A1/A2/A3/A5 function-call arguments were reparsed from `raw_response`.
- No API request was made during recovery.

## Action support contract

| Action | Status | Use in weak-label comparison/training |
|---|---|---|
| A1 | SUPPORTED | Included |
| A2 | SUPPORTED | Included |
| A3 | SUPPORTED | Included |
| A4 | UNSUPPORTED_BY_CURRENT_STATE | Excluded |
| A5 | SUPPORTED | Included; REVIEW |

A4 is excluded from weak-label training, Snorkel, EBM training, and ranking-model evaluation. A5 remains supported but is flagged `REVIEW` because prior weak-source agreement was weak.

The normalized comparison below contains `2000` supported-action pairs (A1/A2/A3/A5), with no A4 rows.
