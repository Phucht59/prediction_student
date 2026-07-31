# Training and preprocessing policy audit

## H0

- Three inner folds.
- Train-only masked temporal mean/std; padded values excluded.
- Median imputation plus StandardScaler for aggregate/static numeric features.
- AdamW, LR 0.00146274001349, weight decay
  1.99048860226e-07, dropout
  0.285489, batch 256.
- Standard BCE and constant survival/outcome weights 0.15/0.15.
- P1 pretraining, then exactly 8 final epochs.

## H1

- Two inner folds.
- Raw temporal values passed through the model's projection/LayerNorm.
- Aggregate nanmean/nanstd and static fill-zero mean/std preprocessing.
- Per-fold Phase 3/early-warning hyperparameters and auxiliary weights.
- No pretraining.
- Checkpoint selection minimizes inner endpoint NLL; refit epochs 10/12/5.

Every transformer remains train-only. No preprocessing leakage was found.
The issue is recipe non-equivalence, not fit-scope contamination.
