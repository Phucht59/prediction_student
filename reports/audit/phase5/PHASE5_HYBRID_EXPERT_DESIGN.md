# Phase 5 — Hybrid Expert Design

H1 preserves the complete H0 CNN-BiLSTM and A0 scalar-gated path. The train-only preprocessed 165 aggregate plus 13 static dimensions also enter a compact `178→48→32` expert. Its scalar risk logit bypasses shared-representation compression through `z_final = z_hybrid + sigmoid(a) * z_tabular`. Alpha starts at 0.05. H1 has 160,492 parameters (+6.85%), inside the +15% budget.
