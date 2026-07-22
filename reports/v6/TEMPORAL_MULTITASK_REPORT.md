# V6 temporal multi-task report

| Candidate | Macro-F1 | PR-AUC | Survival C-index | Outcome Macro-F1 | Gate |
|---|---:|---:|---:|---:|---:|
| W0 | 0.824403 | 0.891603 | 0.561538 | 0.595127 | true |
| W1 | 0.824123 | 0.891832 | 0.574320 | 0.605478 | true |

Selected: **W0**. Withdrawal is the only time-to-event target;
Fail is a masked final-outcome class and is never assigned a fabricated event
time. All screening is confined to outer-training fold 0.
