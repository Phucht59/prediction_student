# Model Improvement Plan V3

The V2 benchmark indicates an information/sequence-length limitation rather than a case for increasing CNN–BiLSTM capacity. G2 alone reaches Macro-F1 0.8977; tuned CNN–BiLSTM reaches 0.7984 and loses every fold to G2. G1 plus G2 does not improve over G2 in the tested logistic baseline. HGB/MLP are competitive but do not beat G2.

Priority 1: ordinal small MLP with a pre-registered ordinal head, compared against nominal small MLP and G2 rule on the existing V2 folds. Accept only if Macro-F1 rises without worsening QWK, ordinal MAE, two-step errors, High-class F1, or seed variance.

Priority 2: reduced-capacity BiLSTM ablation only as a research control; BatchNorm-to-LayerNorm is justified only if a pre-registered stability ablation identifies normalization as the source of variance. Do not add layers, attention, or Transformer machinery to length two.

Priority 3: hybrid or multi-task model only after an independently reviewed availability contract admits additional features or new longitudinal grades are obtained. It must include separate ablations for sequence branch, tabular branch, fusion, and any auxiliary G3 regression head.

Stop a candidate after one full V2 benchmark if it is below G2 on most folds or adds seed variance without a compensating ordinal/class-wise benefit. Legacy-79 is never used for acceptance, rejection, seed selection, or threshold choice.
