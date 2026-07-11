# Final report audit (read-only)

**Audited document:** `C:\Users\THPhu\Downloads\KLTN_CNN_BiLSTM_Tran_Hoang_Phuc_FINAL.docx`  
**Audit basis:** frozen evidence bundle, `14_FINAL_FACTS.json`, frozen selected
configuration, model-selection artifacts, final project documentation and final
source code.  
**Scope:** factual, protocol, traceability and claim audit only. The DOCX was
not modified.

## PASS

| Area | Result | Evidence used |
| --- | --- | --- |
| Dataset identity and target | Correctly identifies UCI `student-mat`, Portuguese secondary-school mathematics, 395 records, G3 thresholds and the three scenarios. | `02_DATASET_AND_PROBLEM.md`; dataset/model-selection manifest |
| Final architecture | Correctly names the final model as **CNN-BiLSTM single seed 42**, with G1/G2 two-step input, 16 channels, kernel 1, hidden size 32, two dropout values and 13,059 parameters. It does not present the ensemble as final. | `selected_config.json`; `04_MODEL_ARCHITECTURE.md` |
| Final protocol | Correctly states locked test 79, outer 5 folds, inner 3 folds, 30 Optuna trials, frozen configuration and no locked-test model selection. | selected configuration; protocol manifest |
| Final CNN-BiLSTM results | Abstract, Chapter 4 tables and conclusion use the frozen values: nested Macro-F1 0.8781 +/- 0.0448; locked Accuracy 0.9114; Macro-F1 0.9262; weighted F1 0.9122; QWK 0.9152; ordinal MAE 0.0886; Brier 0.1683; ECE 0.0591; macro PR-AUC 0.9699. | final `run_manifest.json`; `classification_report.json`; `14_FINAL_FACTS.json` |
| Error analysis | The report's confusion matrix, 7 one-step errors, zero two-step errors and test support 26/38/15 agree with frozen predictions. | final evidence predictions and run manifest |
| Baseline conclusion | Correctly reports G2 rule locked Macro-F1 0.9365 and HGB locked Macro-F1 0.9463; it explicitly avoids claiming CNN-BiLSTM superiority. | `baseline_results.csv`; `MODEL_COMPARISON_PROTOCOL.md` |
| HGB protocol distinction | Correctly separates HGB 0.8969 train-pool OOF from 0.8690 same-outer-fold nested comparison. | `MODEL_COMPARISON_PROTOCOL.md` |
| Ablations | Fixed CNN-BiLSTM single model (0.8422/0.9098), ensemble 11 seed (0.8505/0.8876), imbalance variants and final selected model are distinguished. | `deep_ablation_results.csv`; final selected config |
| Early scenarios | Correctly gives early-warning 0.6974 and pre-assessment 0.4344 OOF Macro-F1 and avoids direct equivalence with late-stage. | `baseline_results.csv`; `scenario_results.csv` |
| Recommendation | Correctly calls `student_mat_rule_policy_v3` a rule-based advisory policy, reports 79 valid/contradiction-free outputs, and says expert review is pending. | `recommendation_evaluation.json`; `08_RECOMMENDATION_SYSTEM.md` |
| PostgreSQL status | Live migration 003, 395-row target lineage and DB-first verification are complete. | `FINAL_PROJECT_AUDIT.md`; DB-first evidence |
| Reproducibility | Frozen configuration produced 0 class mismatches; probability drift is at most `2.78e-08`; 62 tests pass without skips. | `reproducibility_manifest.json`; audit docs |
| Structure | The document contains a coherent title/abstract, five chapters, appendix material, 41 tables and listed figures. Captions/tables observed are consistent with project terminology. | DOCX structural inspection |

## NEEDS_CORRECTION

1. **Appendix E key name:** the text describes the target key as
   `dataset_version_id` plus `source_record_id`. Migration 003 and the final
   schema use `dataset_version_id` plus **`record_id`**. Replace
   `source_record_id` with `record_id` wherever it denotes the physical column;
   `source record identity` remains acceptable as descriptive prose.

2. **Dataset-variable table wording:** the row containing `paid` says that the
   listed study/support variables can be used as permitted risk factors by the
   policy. The final automated recommendation policy excludes `paid` (along
   with sex, school, address, guardian, alcohol and going-out variables). Split
   the table wording into “available dataset features” and “variables permitted
   in automated recommendation rules,” or explicitly mark `paid` as excluded.

3. **Reference/caption final pass:** table and figure titles are structurally
   present, but their page numbers and all external bibliography metadata need a
   final human Word/department-format review. The local renderer could not run
   because LibreOffice/soffice is unavailable, so this audit does not certify
   visual pagination, visual table overflow or bibliography style compliance.

## UNSUPPORTED CLAIM

No unsupported quantitative final-model claim was found in the audited DOCX.

The following statements must remain qualified, as the repository does not
support stronger versions of them:

| Do not claim | Reason |
| --- | --- |
| Recommendation improves student outcomes or has expert validation | Expert evaluation is `not_collected`; structural evaluation is not intervention evidence. |
| Expert review is complete | No independent expert scores have been collected. |
| CNN-BiLSTM outperforms G2/HGB or is the best deployable predictor | G2 and HGB have higher locked-test Macro-F1. |
| Results generalize to Vietnamese university students | The evidence is Portuguese secondary-school mathematics data only. |
| G1/G2 model a long-term learning history | The frozen input sequence has length two. |

## Audit checks performed

- Frozen metrics and configuration were compared with final DOCX text/tables.
- Legacy values `0.9256` and `0.8736`, old 16-trial/four-inner-fold protocol,
  kernel 3 and an “ensemble final” claim were not found as final claims.
- Selected config SHA-256 and frozen prediction checksum were checked against
  project facts.
- No DOCX content was changed, copied, re-saved or exported.
