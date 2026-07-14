# Scientific rationale

Benchmark V2 shows nominal Small MLP and HGB outperform CNN–BiLSTM on `[G1,G2]`, while V2.2 shows training budget—not kernel-one—is the main neural sanity effect. V3 therefore prioritizes a small tabular backbone, ordered class supervision, and a separately labelled continuous-G3 auxiliary signal. RMSE, MAE and R² are reported only for models with a regression head and on inverse-transformed raw 0–20 predictions without clipping.
