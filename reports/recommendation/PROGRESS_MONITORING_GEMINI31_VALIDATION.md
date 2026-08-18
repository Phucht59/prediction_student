# A4 Progress Monitoring: Gemini 3.1 vs Gemini 3.5

These are two distinct LLM weak-label sources from the Gemini model family, not fully independent annotators.

- Decision: `REVIEW`
- Gemini 3.1: `gemini-3.1-flash-lite`
- Gemini 3.5: `gemini-3.5-flash-lite`
- Cases compared: `500`
- Exact agreement: `281/500` (0.562000)
- Weighted Cohen kappa on numeric non-ABSTAIN pairs: `0.713713806908109`
- Gemini 3.1 ABSTAIN rate: `0.000000`; distribution `{'0': 82, '1': 214, '2': 92, '3': 112, 'ABSTAIN': 0}`
- Gemini 3.5 ABSTAIN rate: `0.000000`; distribution `{'0': 54, '1': 178, '2': 212, '3': 56, 'ABSTAIN': 0}`
- Gemini 3.1 max class share: `0.428000`; collapsed: `False`
- Observable Student State variation detected: `True`

No automatic hard agreement threshold is applied. `REVIEW` means the agreement is available for substantive review rather than being declared materially reasonable by an invented cutoff.

## Agreement by stage

| Stage | N | Exact | Weighted kappa | Gemini 3.1 ABSTAIN | Gemini 3.5 ABSTAIN |
|---|---:|---:|---:|---:|---:|
| 20pct | 133 | 0.526316 | 0.5991561181434599 | 0.000000 | 0.000000 |
| 35pct | 129 | 0.581395 | 0.7002102622745214 | 0.000000 | 0.000000 |
| 50pct | 122 | 0.532787 | 0.7402597402597403 | 0.000000 | 0.000000 |
| 75pct | 116 | 0.612069 | 0.7928571428571428 | 0.000000 | 0.000000 |

## Agreement by risk band

| Risk band | N | Exact | Weighted kappa | Gemini 3.1 ABSTAIN | Gemini 3.5 ABSTAIN |
|---|---:|---:|---:|---:|---:|
| high | 109 | 0.513761 | 0.2804832482251838 | 0.000000 | 0.000000 |
| low | 272 | 0.522059 | 0.3341810783316378 | 0.000000 | 0.000000 |
| medium | 119 | 0.697479 | 0.5274411974340698 | 0.000000 | 0.000000 |

## Confusion matrix

Rows = Gemini 3.1; columns = Gemini 3.5.

| Gemini 3.1 \ Gemini 3.5 | 0 | 1 | 2 | 3 | ABSTAIN |
|---|---:|---:|---:|---:|---:|
| 0 | 31 | 43 | 8 | 0 | 0 |
| 1 | 23 | 121 | 70 | 0 | 0 |
| 2 | 0 | 13 | 76 | 3 | 0 |
| 3 | 0 | 1 | 58 | 53 | 0 |
| ABSTAIN | 0 | 0 | 0 | 0 | 0 |

## Student State variation diagnostic

Thresholds below are diagnostic heuristics only; they are not a label-selection gate.

| Feature | Numeric-label mean range | Coverage range | Variation observed |
|---|---:|---:|---|
| risk_probability | 2.000000 | 0.000000 | True |
| inactive_streak | 1.299583 | 0.000000 | True |
| active_days_ratio | 1.437968 | 0.000000 | True |
| recent_activity | 1.017169 | 0.000000 | True |
| activity_trend | 0.368000 | 0.000000 | True |
| assessment_completion | 1.608037 | 0.000000 | True |
| missing_assessments | 1.697668 | 0.000000 | True |
| course_progress | 0.110819 | 0.000000 | False |
