# Constrained Counterfactual Learning Recommender

Status: `IMPLEMENTATION_CANDIDATE_NOT_FINAL_RELEASE`

## 1. Motivation

The original weak-supervision experiment used Snorkel labeling functions to
create silver action labels. The diagnostic audit showed that action and stage
identity almost completely determined those labels. A neural ranker could
therefore reproduce the labeling rules without learning whether an action was
useful for a particular learner.

The candidate implemented here does not train another recommender from silver
labels. It uses the frozen Hybrid CNN–BiLSTM prediction model as a read-only risk
authority and ranks feasible learning actions by model-estimated risk reduction.

## 2. Architecture

```text
Observed OULAD data before the cutoff
                  |
                  v
Frozen Hybrid CNN–BiLSTM baseline risk
                  |
                  v
Evidence policy candidate generator
                  |
                  v
Feasible action counterfactual simulation
                  |
                  v
Canonical 47-channel and 165-feature rebuild
                  |
                  v
Frozen Hybrid CNN–BiLSTM counterfactual risk
                  |
                  v
Risk-reduction utility ranking
                  |
                  v
Existing workload, prerequisite and safety constraints
                  |
                  v
Learning-support plan with evidence and explanation
```

The policy remains responsible for candidate eligibility, cutoff routing,
evidence requirements, abstention and urgent human escalation. The
counterfactual component only orders eligible, model-scorable actions.

## 3. Counterfactual utility

For student state `x` and action `a`:

```text
risk_reduction(a) = risk(x) - risk(counterfactual(x, a))

utility(a) = positive_risk_reduction
             * evidence_strength
             * uncertainty_penalty
             / workload_factor
```

An action is rejected when the estimated reduction is below the configured
minimum. The score is an ordering utility, not a treatment-effect estimate and
not a probability that the intervention will work.

## 4. Action simulation

Only mutable OULAD behavior channels can be changed:

- VLE clicks and active days;
- content, quiz and assessment-related activity;
- submitted assessment count;
- inactivity indicators derived from observed activity.

The following information is protected:

- released score channels;
- static student/course features;
- cutoff and stage context;
- final result, target and unregistration outcome;
- all post-cutoff observations.

Human-support actions such as instructor contact and advisor escalation are not
assigned synthetic feature effects. They remain policy decisions and safety
fallbacks.

## 5. Frozen preprocessing authority

The temporal sequence is represented in the original 47-channel model space.
The aggregate branch was trained using a fold-specific 165-feature
standardization and the static branch was trained using fold-specific numeric
scaling and categorical levels.

Each checkpoint stores that preprocessing state. The prediction adapter:

1. verifies that all selected seed checkpoints share the same preprocessing
   hash;
2. inverse-transforms the baseline aggregate to raw feature space;
3. rebuilds the changed 161 temporal aggregate features;
4. preserves the four raw context features;
5. reapplies the frozen aggregate transform;
6. transforms the static columns with the frozen numeric and categorical state.

No preprocessing parameter is estimated from an outer-validation student.

## 6. Training-fold reference profiles

Action targets such as activity p50 or p65 are estimated only from the training
partition of the same outer fold, course presentation and prediction stage.
Padding weeks are excluded. Targets and final outcomes are not accepted by the
reference-profile builder API.

Each profile records:

- outer fold;
- course presentation;
- stage;
- number of training students and observed weeks;
- percentile values and fallback status;
- deterministic profile hash.

## 7. Safety and fallback

The existing constraint solver is retained. It enforces action count, workload,
prerequisites, conflicts, evidence and stage applicability.

Critical or high-priority human support is never demoted by the model ranking.
The system returns the existing deterministic policy plan when any required
scientific input is missing, including:

- frozen model inputs;
- training-fold reference profile;
- frozen checkpoint authority;
- sufficient policy evidence;
- a positive model-estimated risk reduction.

Fallback reasons are stored in the result contract.

## 8. Data and checkpoint availability

Raw OULAD tables and the generated stage bundle are intentionally excluded from
Git. Full evaluation must run on the project machine that owns:

- `data/raw/*.csv` registered in `data/manifests/extension_raw_manifest.json`;
- `data/processed/study_c_oulad/manifests/split_manifest.csv`;
- canonical training checkpoints referenced by the recommendation manifest;
- OOF predictions required by historical trajectory validation.

GitHub CI separately runs a real release-checkpoint smoke with a deterministic
contract-valid synthetic tensor. That smoke validates checkpoint loading,
architecture authority, frozen preprocessing and counterfactual scoring, but is
not reported as educational-effect evidence.

## 9. Validation and evaluation

### 9.1 Technical validation

```bash
python scripts/recommend_hybrid/validate_counterfactual.py
```

This runs focused unit/integration tests and static scientific gates.

### 9.2 Local authority preflight

```bash
python scripts/recommend_hybrid/preflight_counterfactual_evaluation.py \
  --verify-hashes
```

The command fails before evaluation when raw data, the frozen split, checkpoint
files, training authority or OOF predictions are missing or inconsistent.

### 9.3 Outer-fold counterfactual evaluation

```bash
python scripts/recommend_hybrid/evaluate_counterfactual_recommender.py \
  --folds 0,1,2 \
  --stages E1_EARLY_20PCT,E2_EARLY_35PCT,M1_MIDDLE_FROZEN,L1_LATE_75PCT \
  --seeds 42,1201,2026,3407,7319 \
  --max-records-per-fold-stage 100
```

`M1_MIDDLE_FROZEN` is the canonical bundle key. The OOF/reporting alias remains
`M1_MIDDLE_50PCT`; code and tests keep these namespaces separate.

Main metrics:

- scored coverage and fallback rate;
- mean and median top-action risk reduction;
- Success@0.01 and Success@0.05;
- decision-threshold crossing rate;
- workload and selected-action count;
- action frequency and concentration;
- fixed-seed bootstrap confidence interval.

### 9.4 Historical trajectory validation

```bash
python scripts/recommend_hybrid/evaluate_historical_trajectories.py
```

This compares later observed behavior and outcomes between learners whose
behavior moved in the recommended direction and those whose behavior did not.
It is an observational association check only. Its outputs are not used to
train, tune, score or select an action.

### 9.5 Candidate release build

```bash
python scripts/recommend_hybrid/build_counterfactual_candidate_release.py \
  --verify-hashes \
  --max-records-per-fold-stage 100 \
  --bootstrap-replicates 1000
```

The runner executes preflight, technical validation, outer-fold evaluation and
historical validation, then creates a checksum registry. A successful run is
marked `CANDIDATE_VALIDATED`, not `FINAL_RELEASE`.

Use `--max-records-per-fold-stage 0` only when a complete all-row evaluation is
required and the available compute budget is sufficient.

## 10. Scientific claim boundary

Supported wording:

> The system recommends feasible learning actions that the frozen prediction
> model estimates may reduce academic-risk probability for the current learner.

Not supported:

- the action is optimal;
- the action causally improves grades;
- the recommendation is expert validated;
- the estimated risk reduction equals a real intervention effect;
- observational trajectory differences prove effectiveness.

The release claim identifier is:

```text
MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT
```

Historical analysis uses:

```text
OBSERVATIONAL_ASSOCIATION_ONLY_NOT_CAUSAL_EFFECT
```

## 11. Release gate

The module must remain an implementation candidate until all of the following
are complete:

1. focused CI passes;
2. registered release-checkpoint smoke passes;
3. local preflight passes with hashes verified;
4. outer-fold evaluation artifacts are generated from real checkpoints and raw
   OULAD data;
5. preprocessing, no-leakage and deterministic-replay checks pass;
6. results are reviewed for degenerate action concentration;
7. thesis claims are updated to the exact boundaries in this document.
