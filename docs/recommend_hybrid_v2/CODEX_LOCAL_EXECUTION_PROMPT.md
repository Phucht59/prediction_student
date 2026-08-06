# Codex Local Execution Prompt — Recommendation Module V2

You are an execution agent, not the system designer. Follow this document exactly. Do not replace the architecture, simplify the scientific protocol, invent labels, weaken gates, tune on test data, or modify the frozen Hybrid model.

## 0. Operating scope

Repository: `Phucht59/prediction_student`

Required branch: `Module_recomend`

Authoritative base commit: `f2a06a7b77b2bf7519cc2ff70b3997c703ae6819`

The following files are protocol authorities and must be read before changing code:

- `docs/recommend_hybrid_v2/SCIENTIFIC_PROTOCOL.md`
- `configs/recommend_hybrid/explainable_v2.yaml`
- `configs/recommend_hybrid/actions_v2.yaml`
- `configs/recommend_hybrid/feature_contract_v2.yaml`
- `configs/recommend_hybrid/literature_sources_v2.yaml`
- `src/recommend_hybrid/explainable_v2/`

Do not create another branch. Do not merge. Do not rebase onto `main`. Work only on `Module_recomend`.

Do not launch duplicate jobs. Before every long run, check active Python processes and existing run manifests. All long-running jobs must be resume-safe and use one supervisor PID manifest.

## 1. Non-negotiable scientific constraints

1. The frozen Hybrid CNN–BiLSTM is the sole risk authority.
2. Do not change Hybrid architecture, parameter count, checkpoints, thresholds already frozen for model evaluation, preprocessing authority, outer folds, or OOF lineage.
3. The V2 recommendation module does not train another risk model.
4. The ranker operates on the five canonical actions only:
   - `ASSESSMENT_COMPLETION`
   - `RECOVER_ENGAGEMENT`
   - `STUDY_REGULARITY`
   - `TARGETED_CONTENT_REVIEW`
   - `QUIZ_RETRIEVAL_PRACTICE`
5. `NO_ACTION`, `MONITOR`, and `HUMAN_REVIEW` are routes, not actions.
6. Never use these as ranker features:
   - action identity;
   - final result;
   - post-cutoff behavior;
   - weak-label confidence, entropy, or conflict;
   - OOD score;
   - current or previous action-head output;
   - causal ATE/CATE;
   - raw student identifier;
   - sensitive demographic attributes.
7. All behavioral ranker features must end strictly before `cutoff_day`.
8. Known future course schedule fields may be used only for feasibility, as defined in `feature_contract_v2.yaml`.
9. Fit imputation, preprocessing, label aggregation, feature thresholds, score calibration, OOD detection, hyperparameters, risk routing thresholds, and safety-router thresholds using training/validation data only.
10. Open each outer test fold only after all choices for that fold are frozen.
11. Keep `runtime_authorized: false` in every manifest and report.
12. Do not make expert-validation, deployment-effectiveness, or causal claims.

Any conflict between existing old code and V2 protocol must be resolved in favor of V2 without deleting old artifacts. Old action-head code remains a comparator only.

## 2. Preflight and repository integrity

Run and save outputs:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD
git lfs status
git lfs fsck
python --version
nvidia-smi
```

Requirements:

- Branch must be exactly `Module_recomend`.
- Working tree must be understood before edits.
- Do not stage unrelated user files.
- Verify all LFS artifacts needed by the frozen Hybrid pipeline are present.
- Create `artifacts/recommend_hybrid/explainable_v2/run_state/preflight.json` containing command results, Python version, package versions, CUDA availability, GPU name, branch, HEAD, and timestamps.

Install the locked recommendation environment:

```bash
python -m pip install -r requirements-recommend-v2.txt
```

If an existing project virtual environment is authoritative, install into that environment rather than creating an incompatible one. Record exact versions with `pip freeze`.

Run the existing and new static checks before data work:

```bash
python scripts/recommend_hybrid/validate_explainable_v2_static.py
pytest -q tests/recommend_hybrid/explainable_v2
```

Also run the project’s existing Ruff command if configured. If no Ruff configuration exists, run syntax compilation and do not invent a style configuration:

```bash
python -m compileall src/recommend_hybrid/explainable_v2 scripts/recommend_hybrid
```

Fix real failures. Do not relax assertions to make tests pass.

## 3. Build the V2 learner-stage feature table

Create a new script:

`src/recommend_hybrid/explainable_v2/data_builder.py`

and an executable wrapper:

`scripts/recommend_hybrid/explainable_v2/build_feature_table.py`

Output:

- `artifacts/recommend_hybrid/explainable_v2/data/learner_stage_features.parquet`
- `artifacts/recommend_hybrid/explainable_v2/data/feature_lineage.parquet`
- `artifacts/recommend_hybrid/explainable_v2/data/FEATURE_TABLE_MANIFEST.json`

Use existing canonical OULAD builders, frozen OOF Hybrid predictions/checkpoints, and existing raw data. Reuse validated code where safe, especially the pre-cutoff behavior logic in `scripts/recommend_hybrid/causal/build_oulad_landmark_rows.py`, but do not copy follow-up or treatment fields into V2 ranker features.

The feature table must contain one row per unique `student_key + course_key + stage`, not one row per action. Required identity fields:

- `query_id`
- `student_key`
- `course_key`
- `code_module`
- `code_presentation`
- `outer_fold`
- `stage`
- `cutoff_day`

Required model/ranker fields are exactly those in `feature_contract_v2.yaml` and `FEATURE_COLUMNS` in `ranker.py`.

Required routing/feasibility fields may also be present but must be separated in the manifest.

Risk fields:

- Obtain at-risk probabilities from the five frozen Hybrid checkpoints belonging to the learner’s frozen outer fold and stage.
- `risk_probability` = arithmetic mean across five seed probabilities.
- `seed_disagreement` = population standard deviation across those five probabilities.
- `hybrid_uncertainty` = normalized binary entropy of the mean probability.
- Verify checkpoint SHA-256, architecture hash, and fold lineage against the existing manifest before inference.
- Never run a checkpoint trained on the learner’s test fold as an in-sample model.

Behavioral fields:

- Use data strictly before the cutoff.
- Do not interpret zero as missing or missing as zero unless the feature contract explicitly permits it.
- For assessment progress, if no assessment was scheduled before cutoff, retain missing and a missing indicator; do not assign 1.0.
- `assessments_due` and `time_to_deadline_days` use verified schedules only and exclude already submitted assessments.
- `assessment_window_open` means a verified scheduled future opportunity only. Do not claim knowledge of extensions or institution reopening.
- `knowledge_gap_evidence` may use only scores released before cutoff and a threshold fitted from inner-training data. If topic-level evidence is unavailable, do not claim a topic-level gap.
- `quiz_available` and `study_material_available` must be verified from course/VLE metadata.

Lineage:

Every generated value must have a lineage row including source table, source column, aggregation, observation start/end, cutoff, and split used to fit any statistic.

Run `assert_pre_cutoff_lineage`. Post-cutoff violations must equal zero.

Manifest must include:

- row and unique student counts;
- counts by stage, outer fold, module, and presentation;
- missingness by feature and stage;
- feature min/max/quantiles;
- duplicate query count;
- student overlap matrix across folds;
- source file hashes;
- frozen Hybrid checkpoint hashes;
- architecture hash;
- post-cutoff violation count;
- outcome-in-feature flag, which must be false.

Do not continue if duplicate queries, student overlap, checkpoint mismatch, or post-cutoff usage is found.

## 4. Risk stratification without relearning risk

Create:

`scripts/recommend_hybrid/explainable_v2/select_risk_policy.py`

Hybrid remains frozen. Use final outcomes only to evaluate and calibrate the frozen risk predictions; final outcomes must never enter action labels or ranker features.

For each outer fold:

1. Use the outer training pool to establish all train-fitted transformations.
2. Use a grouped inner validation split to choose LOW/HIGH thresholds from the grids in `explainable_v2.yaml`.
3. Optimize a preregistered validation objective combining calibrated precision/recall and intervention budget. Record the exact formula before reading outer test metrics.
4. Report LOW, BORDERLINE, HIGH coverage, PR-AUC, Brier score, ECE, precision, recall, and alerts per 1,000 learners.
5. High uncertainty or seed disagreement must route to BORDERLINE/MONITOR regardless of mean risk.
6. Apply frozen selected thresholds once to the outer test fold.

Write fold-specific thresholds and evidence to:

- `artifacts/recommend_hybrid/explainable_v2/risk_policy/outer_<fold>.json`
- `reports/recommend_hybrid_v2/RISK_STRATIFICATION_RESULTS.md`

Do not modify Hybrid checkpoints or retrain Hybrid.

## 5. Build action candidates and deterministic feasibility

Create:

`scripts/recommend_hybrid/explainable_v2/build_action_candidates.py`

For every learner-stage query, cross with the five canonical actions and apply `feasibility.py`.

Output one row per query-action:

`artifacts/recommend_hybrid/explainable_v2/data/action_candidates.parquet`

Columns must include:

- query and split identity;
- action ID;
- eligible flag;
- reason codes;
- all admissible learner context fields;
- no relevance target yet.

Run audits proving:

- exactly five candidate rows before filtering per query;
- deterministic candidate order;
- invalid automatic action count zero after filtering;
- no action ID in Five-EBM learner context columns;
- no final result or label diagnostic in ranker columns.

## 6. Implement V2 weak labeling functions

Create package:

`src/recommend_hybrid/explainable_v2/labeling_functions/`

Each labeling function must return `ABSTAIN`, 0, 1, 2, or 3 for one query-action. Every function requires metadata:

- source ID;
- source family;
- literature registry reference where applicable;
- feature dependencies;
- stage applicability;
- positive and negative conditions;
- abstention conditions;
- version.

Do not hard-code universal thresholds taken from unrelated studies. Behavioral thresholds must be estimated only from inner-training distributions, stored per stage where needed, then frozen before validation/test application.

Required source families:

### A. Literature-grounded family

Use the claims and limits in `literature_sources_v2.yaml`. Literature functions provide directional evidence only. They may abstain frequently.

### B. OULAD behavioral family

Implement distinct, auditable rules for:

- assessment need;
- disengagement recovery;
- irregular study pattern;
- content review need;
- quiz/retrieval practice need.

Rules must distinguish `RECOVER_ENGAGEMENT` from `STUDY_REGULARITY`. A learner with nearly no recent activity should favor recovery; a learner with adequate total activity but highly uneven timing may favor regularity.

### C. Policy/feasibility family

- Ineligible or contraindicated actions vote 0 or unsafe according to protocol.
- Feasibility alone must not vote an action as highly relevant.

### D. LLM weak-annotator family

Implement deterministic exporter/importer and schema validation. Do not fabricate LLM responses.

Create:

- `scripts/recommend_hybrid/explainable_v2/export_llm_annotation_cases.py`
- `scripts/recommend_hybrid/explainable_v2/import_llm_annotations.py`
- `configs/recommend_hybrid/llm_annotation_protocol_v2.yaml`

The LLM case must:

- contain only pre-cutoff evidence;
- hide final result;
- hide exact Hybrid probability, showing only a blinded risk band if needed;
- hide old and new ranker outputs;
- randomize action order deterministically per reviewer/run;
- use pseudonymous case IDs;
- permit abstention;
- require relevance 0–3, evidence codes, rationale, and safety/escalation flags.

Training LLM panel A and held-out benchmark panel B must be source-disjoint by model family or prompt family and use different case IDs/order secrets.

If valid LLM API credentials or imported annotation files are unavailable:

- do not invent annotations;
- complete the exporter/importer and all rule-only preliminary experiments;
- set final scientific status to `BLOCKED_PENDING_LLM_WEAK_LABELS`;
- do not claim the final model is selected.

## 7. Aggregate weak labels correctly

Create:

`scripts/recommend_hybrid/explainable_v2/fit_weak_label_models.py`

For every outer fold and action:

1. Fit the Snorkel LabelModel on inner-training votes only.
2. Apply it to inner validation and outer test without refitting.
3. Produce class probabilities for relevance 0–3, expected relevance, confidence, entropy, source-family count, and retained/abstained status.
4. Require at least two independent source families for retained training targets.
5. Use label confidence only as a training sample weight, never as an input feature.
6. Produce pairwise source agreement and correlation audits.
7. Run leave-one-source-family-out label aggregation and later model ablations.
8. Keep test labels frozen and never use test agreement to alter rules.

Output:

- `artifacts/recommend_hybrid/explainable_v2/labels/outer_<fold>_<action>.parquet`
- `artifacts/recommend_hybrid/explainable_v2/labels/SOURCE_AUDIT.json`
- `reports/recommend_hybrid_v2/WEAK_LABEL_EVIDENCE.md`

Block release when one source family effectively determines all retained labels or when action-stage purity recreates the previous shortcut.

## 8. Held-out pseudo-expert benchmark

Build a source-disjoint benchmark across all four stages, risk levels, modules/presentations, clear cases, and deliberately ambiguous cases.

Target size: 800 unique learner-stage cases if data and annotation budget permit; minimum 480. Sampling must be stratified and frozen before annotation.

Each case must be judged repeatedly by panel B with action order randomization. Report:

- annotation completion and abstention;
- pairwise weighted agreement;
- repeated-prompt self-consistency;
- relevance distribution per action/stage;
- ambiguous-case entropy;
- safety/escalation flags.

Call this artifact `HELD_OUT_PSEUDO_EXPERT_BENCHMARK`, never expert ground truth.

Do not use it for hyperparameter tuning, threshold selection, rule revision, or label model fitting. It is opened only after candidate models and release gates are frozen.

## 9. Train and optimize candidate recommenders

Create a resume-safe supervisor:

`scripts/recommend_hybrid/explainable_v2/run_model_selection.py`

All models train on identical grouped splits and retained weak targets. Evaluation is query-level, never row-level.

Required candidates:

1. Global popularity ranking.
2. Literature/behavior rule-severity ranking.
3. Action-stage-only baseline.
4. Independent logistic relevance models.
5. Five independent EBM relevance models.
6. LambdaMART challenger.
7. Frozen old four-stage neural action-head comparator evaluated on the same eligible queries when mapping is valid.

### Five-EBM optimization

- Use one independent EBM per action.
- Do not include action ID.
- Search only the grids in `explainable_v2.yaml`.
- Use grouped inner cross-validation by student.
- Optimize mean validation NDCG@3 across complete query sets, not per-row loss alone.
- Use label confidence as sample weight only.
- Record every trial, fold score, parameter set, random seed, wall-clock information, and failure.
- Use deterministic, resume-safe SQLite or JSONL state.
- Do not select hyperparameters using outer test or pseudo-expert benchmark.

### Logistic baseline

Search a small preregistered regularization grid and use the same missingness/preprocessing policy.

### Action-stage-only baseline

It may use action, stage, module, and presentation, but no learner context. Its purpose is shortcut detection, not deployment.

### LambdaMART

Use LightGBM ranking groups defined by query. Tune on grouped inner validation only. Use a bounded, preregistered search of tree complexity, learning rate, number of estimators, minimum leaf size, feature fraction, bagging fraction, and L1/L2 regularization. Do not allow query rows to split across folds.

### Old neural action-head

Do not retrain it on outer test. Evaluate existing OOF predictions where action mapping and query identity are valid. Clearly report any queries it cannot represent.

### Score calibration

Fit per-action isotonic calibrators on validation predictions only using `calibration.py`. Safety routing must consume calibrated score outputs. Raw clipped EBM output is not a calibrated probability.

## 10. Mandatory ranking evaluation and shortcut audits

Use `metrics.py` and query-level data.

Primary endpoint:

- NDCG@3.

Secondary:

- Precision@1;
- MRR;
- Recall@3;
- pairwise accuracy;
- invalid-action rate;
- action diversity;
- top-1 stability across folds and seeds;
- coverage and abstention.

Required analyses:

1. Per outer fold.
2. Overall OOF.
3. Per stage.
4. Per module/presentation.
5. Per risk band where meaningful.
6. Missingness subgroups.
7. Bootstrap 95% confidence intervals clustered by learner/query.
8. Paired full-minus-action-stage-only bootstrap.
9. Context permutation using `audits.py`; eligibility, target, action identity, and query set remain fixed.
10. Leave-one-label-source-family-out model training/evaluation.
11. Feature ablation:
    - behavior only;
    - Hybrid probability/uncertainty only;
    - behavior + Hybrid;
    - stage excluded;
    - each action evidence family removed.
12. Top-action concentration and action distribution by stage.
13. Explanation stability under seeds and small permissible perturbations.

A candidate is ineligible when:

- invalid action rate is nonzero;
- it fails to beat rule, popularity, or action-stage-only baselines;
- full-minus-action-stage-only bootstrap CI includes only non-positive values;
- context permutation does not materially reduce ranking quality;
- one label-source family is solely responsible for performance;
- any student or post-cutoff leakage is found.

Use `model_selection.py`. Select the simplest release-gate-passing model statistically indistinguishable from the empirical best. Do not choose a neural or tree model merely because it is more complex.

## 11. Optimize risk and safety routing

Create:

`scripts/recommend_hybrid/explainable_v2/select_safety_policy.py`

Use validation data only to select:

- minimum top-1 score;
- minimum top-1 minus top-2 margin;
- maximum Hybrid uncertainty;
- maximum seed disagreement;
- maximum weak-label conflict;
- maximum OOD score.

Use the exact grids in `explainable_v2.yaml` unless a code-level incompatibility is proven and documented before test evaluation.

Optimize a declared validation utility that penalizes:

- incorrect automatic recommendation;
- invalid action;
- over-coverage;
- excessive HUMAN_REVIEW load;
- missed high-confidence recommendation.

Never optimize solely for coverage or solely for NDCG.

Report route coverage:

- NO_ACTION;
- MONITOR;
- RECOMMEND;
- HUMAN_REVIEW;
- alerts per 1,000 learners.

## 12. Learning-plan builder

Create:

`src/recommend_hybrid/explainable_v2/plan_builder.py`

Every RECOMMEND output must include:

- primary action;
- up to two secondary actions;
- top evidence contributions from EBM;
- plain Vietnamese rationale;
- concrete duration;
- measurable target;
- next review stage;
- contraindication/availability evidence;
- frozen Hybrid risk band and uncertainty summary;
- model, fold, calibration, threshold, and artifact lineage;
- explicit statement that recommendation effectiveness is not causal evidence.

Do not expose raw embeddings or sensitive features in explanations.

## 13. Constrained Hybrid plausibility simulator

Build only after model selection is frozen.

Create package:

`src/recommend_hybrid/explainable_v2/simulator/`

and runner:

`scripts/recommend_hybrid/explainable_v2/run_plausibility_simulator.py`

The simulator is not a label generator and not a causal estimator.

Rules:

1. Transform raw behavioral events, not isolated standardized model columns.
2. Use only mutable behaviors associated with the chosen action.
3. Derive LOW/MEDIUM/HIGH doses from empirical inner-training distributions, stratified by stage and module/presentation where sample sizes allow.
4. Require empirical support; unsupported scenarios are abstained.
5. Preserve immutable history and all data before the current cutoff.
6. Generate behavior only in the current-to-next-stage window.
7. Re-run the canonical feature engineering pipeline at the next validated stage.
8. Re-run all five frozen Hybrid seeds valid for that fold/stage.
9. Compare chosen action against every other feasible action and a placebo/no-change scenario.
10. Never modify final result or directly edit risk probability.
11. Report model-implied risk delta, not treatment effect.
12. For a stage without a scientifically valid next-stage checkpoint, abstain rather than inventing a simulation target.

Action transformations:

- ASSESSMENT_COMPLETION: simulate submission of an actually scheduled, still-open assessment before the next stage; do not fabricate assessment scores.
- RECOVER_ENGAGEMENT: add empirically supported active-day/session patterns on verified VLE resources.
- STUDY_REGULARITY: redistribute supported activity across more days without unrealistic click inflation.
- TARGETED_CONTENT_REVIEW: add supported content interactions only when relevant material and knowledge-gap evidence exist.
- QUIZ_RETRIEVAL_PRACTICE: add supported quiz sessions only for available, already studied content.

Required simulator metrics:

- supported scenario rate;
- positive-response rate;
- median model-implied risk delta;
- adverse-response rate;
- chosen-action versus alternative-action concordance;
- seed consistency;
- dose consistency;
- placebo response.

Any adverse or unsupported top action must trigger HUMAN_REVIEW in the offline demonstration; do not silently replace evidence.

## 14. Final reports and artifacts

Required reports:

- `reports/recommend_hybrid_v2/DATA_AND_LINEAGE_AUDIT.md`
- `reports/recommend_hybrid_v2/RISK_STRATIFICATION_RESULTS.md`
- `reports/recommend_hybrid_v2/WEAK_LABEL_EVIDENCE.md`
- `reports/recommend_hybrid_v2/MODEL_COMPARISON.md`
- `reports/recommend_hybrid_v2/SHORTCUT_AND_ABLATION_AUDIT.md`
- `reports/recommend_hybrid_v2/PSEUDO_EXPERT_BENCHMARK.md`
- `reports/recommend_hybrid_v2/PLAUSIBILITY_SIMULATOR_RESULTS.md`
- `reports/recommend_hybrid_v2/FINAL_RECOMMENDATION_V2_RESULTS.md`
- `reports/recommend_hybrid_v2/MODEL_CARD.md`

Required machine-readable artifacts:

- complete configuration snapshot;
- split manifest;
- feature manifest and lineage;
- label source registry and votes;
- source correlation and ablation results;
- trial database/log;
- fitted per-action models;
- calibrators;
- selected routing thresholds;
- OOF prediction table;
- baseline prediction tables on the identical query set;
- bootstrap samples or deterministic bootstrap seed manifest;
- explanation artifacts;
- simulator scenarios and output;
- SHA-256 manifest for every final artifact;
- final release validation JSON.

The final release JSON must include:

```json
{
  "status": "PASS_OR_BLOCKED_WITH_EXACT_REASON",
  "scientific_status": "OFFLINE_WEAK_SUPERVISION_VALIDATED_OR_BLOCKED",
  "runtime_authorized": false,
  "expert_validated": false,
  "causal_effect_identified": false,
  "frozen_hybrid_modified": false
}
```

Do not write PASS if LLM benchmark evidence is required but unavailable, if any mandatory gate fails, or if tests did not run.

## 15. Validation commands

At minimum run:

```bash
python scripts/recommend_hybrid/validate_explainable_v2_static.py
pytest -q tests/recommend_hybrid/explainable_v2
python -m compileall src/recommend_hybrid/explainable_v2 scripts/recommend_hybrid/explainable_v2
```

Run existing project validation relevant to frozen Hybrid and recommendation artifacts. Run Ruff using the project’s existing command/configuration. Record every command, exit code, and output path in:

`artifacts/recommend_hybrid/explainable_v2/FINAL_VALIDATION.json`

## 16. Git safety and final publication

Before commit:

```bash
git status --short
git diff --check
git diff --stat
git lfs status
git lfs fsck
```

Review every modified file. Do not include caches, API keys, raw private prompts, temporary LLM responses containing identifiers, database files, or unrelated user changes.

Commit intentionally to `Module_recomend` and push that branch. Do not merge and do not mark a PR ready for review unless all final gates pass.

## 17. Final response format

Return only verified facts in this structure:

```text
BRANCH: Module_recomend
BASE_COMMIT: f2a06a7b77b2bf7519cc2ff70b3997c703ae6819
FINAL_COMMIT: <sha>
PUSHED: YES/NO
WORKING_TREE_CLEAN: YES/NO
FROZEN_HYBRID_MODIFIED: NO/YES
STATIC_TESTS: PASS/FAIL
UNIT_TESTS: PASS/FAIL
FULL_LOCAL_TRAINING: PASS/FAIL/BLOCKED
LLM_WEAK_LABELS: AVAILABLE/MISSING
PSEUDO_EXPERT_BENCHMARK: PASS/FAIL/BLOCKED
SELECTED_MODEL: <name or NONE>
PRIMARY_NDCG_AT_3: <value or N/A>
ACTION_STAGE_ONLY_NDCG_AT_3: <value or N/A>
FULL_MINUS_BASELINE_CI_95: <value or N/A>
CONTEXT_PERMUTATION_DELTA: <value or N/A>
INVALID_ACTION_RATE: <value or N/A>
SIMULATOR_STATUS: PASS/FAIL/BLOCKED
SCIENTIFIC_STATUS: <exact status>
RUNTIME_AUTHORIZED: FALSE
PR: <URL or NONE>
```

Do not add optimistic interpretation when a gate is blocked. Do not fabricate metrics. Do not claim future work was completed.
