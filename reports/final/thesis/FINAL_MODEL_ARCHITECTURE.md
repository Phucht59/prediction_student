# Final H1 model architecture

Candidate: `H1_TABULAR_RESIDUAL_EXPERT`

Parameters: **160,492**

Architecture hash: `df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e`

```text
Temporal behavioral sequence (47 channels)
        ↓
CNN kernels 2/3/5, 32 channels, dilation 1
        ↓
Bidirectional LSTM (hidden 64, one layer)
        ↓
masked mean/max temporal pooling → projection 64
        │
        ├───────────────────────────┐
        │                           │
aggregate 165 + static 13           │
        ↓                           │
Tabular Residual Expert             │
        │                           │
        └──────────┬────────────────┘
                   ↓
z_final = z_hybrid + sigmoid(a) × z_tabular
                   ↓
           at-risk probability
```

## 1. Input

The temporal input has **47** channels.
Aggregate and runtime static dimensions are **165** and **13**. Features are
stage-safe and preprocessing is fitted on training partitions only.

## 2. CNN

Input projection: **48**. Parallel kernels:
**[2, 3, 5]**. Conv channels: **32**,
dilation: **1**, with residual processing.

## 3. BiLSTM

One bidirectional LSTM layer with hidden size **64**.

## 4. Temporal pooling

`masked_mean_max` followed by a **64**-
dimensional projection.

## 5. Aggregate/static branch

Aggregate hidden size **64**, static hidden size
**32**, fusion width **64**,
and scalar gated-residual fusion.

## 6. Tabular Residual Expert

Linear(178,48) → LayerNorm(48) → GELU → Dropout → Linear(48,32) → GELU → Linear(32,1). The input is the concatenated 165 aggregate and 13 static
features.

## 7. Fusion/logit

`z_final=z_hybrid+sigmoid(a)*z_tabular`. The coefficient
uses `learnable_bounded_sigmoid` and starts at
0.05.

## 8. Output

The primary output is the binary at-risk logit/probability. Survival and
outcome heads remain training auxiliaries.
