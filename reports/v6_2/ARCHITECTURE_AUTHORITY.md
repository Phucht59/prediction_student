# Architecture authority

## UCI Student Performance

The authoritative Student-Mat and Student-Por model is the frozen V5.1
CNN–BiLSTM family. It treats G1/G2 as a two-step sequence, combines the temporal
encoder with the registered context branch, and uses fold-selected fusion and
auxiliary objectives. The final evidence is complete OOF aggregation over five
outer folds and the fixed seed ensemble; classification is primary.

## OULAD

The authoritative final OULAD prediction model remains the frozen V6 serial
architecture: 47 temporal channels → projection → multi-kernel CNN → residual →
bidirectional LSTM → masked pooling, combined with aggregate/static branches
through gated residual fusion. V6.1 tested parameter-matched CNN, dilation,
serial skip, and parallel CNN || BiLSTM candidates on the permitted development
partition. No candidate passed the preregistered gate, so V6.2 does **not**
switch the final architecture to parallel and performs no new outer evaluation.
