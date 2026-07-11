# Ablation interpretation note

The current CNN/Bi-LSTM ablations use fixed configurations and have different
parameter counts from the 13,059-parameter frozen final model. Therefore a
lower BiLSTM-only score only describes that specific ablation configuration; it
does not establish that CNN is causally essential or that Bi-LSTM is generally
unsuitable. Independent tuning and parameter-matched ablations are future work.
