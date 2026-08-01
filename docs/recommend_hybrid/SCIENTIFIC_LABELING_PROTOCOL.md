# Scientific labeling protocol

## Locked scope

The labeling unit is `student_state x candidate_action`. Each unit must carry a prediction context from the frozen Hybrid CNN-BiLSTM family: `cnn_bilstm_mat`, `cnn_bilstm_por`, or `h1_tabular_residual_oulad`, with the dataset-specific authority, class probabilities, uncertainty, requested cutoff, and checkpoint lineage. Baselines and other model families are not accepted prediction authorities.

Only evidence strictly before the requested cutoff is admissible. UCI supports S0 (no G1/G2), S1 (G1), and S2 (G1/G2); G3 is outcome-only. OULAD supports past-only anchors at 20%, 35%, 50%, and 75% for requests from 20% through 99%. Requests before 20% abstain, and FINAL is evaluation-only.

## Weak-supervision labels

The target classes are `INAPPROPRIATE=0`, `CONDITIONAL=1`, and `APPROPRIATE=2`. `LF_ABSTAIN=-1` means that one labeling function has insufficient evidence to vote; it is not a fourth target class. Runtime `ABSTAIN` is a refusal decision for low coverage, low confidence, conflicting evidence, excessive prediction uncertainty, missing anchor, or a failed safety gate.

Phase 2 will implement independent labeling-function families for published educational evidence, state-action fit, stage/cutoff constraints, prediction uncertainty, prerequisites/contraindications, workload, safety/human review, and prohibited future or sensitive data. Their overlapping and conflicting votes may be combined by a Snorkel Label Model into probabilistic silver labels. Low-coverage cases remain abstentions; high-conflict or high-uncertainty cases require conservative handling or human review. Phase 1 does not run the Label Model or create silver labels.

Silver labels are programmatically inferred, probabilistic training targets. They are not expert-reviewed gold labels. No current artifact is described as expert validation, user acceptance, deployment evidence, causal effect, or proven educational effectiveness.

## Split and evaluation lock

Splits are grouped by `student_key`: a student cannot occur in both train and test, and every stage for that student remains in the same split. The test split is used only after the protocol and thresholds are locked. Later Accuracy, Macro-F1, and NDCG@3 measure agreement with held-out silver labels only; they do not measure real-world educational benefit.

The source registry records group/context-level claims and their limitations. The action map keeps actions without adequate registered evidence as `INSUFFICIENT_EVIDENCE`; such actions are not silently removed or treated as supported.
