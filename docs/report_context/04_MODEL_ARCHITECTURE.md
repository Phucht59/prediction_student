# Final CNN-BiLSTM architecture

Input is one two-step sequence `[G1, G2]` per record. The frozen configuration
uses Conv1D with 16 channels and kernel size 1, sequence dropout 0.197248,
BiLSTM hidden dimension 32, one layer, pooled sequence representation, head
dropout 0.456984 and a linear three-logit classifier. Softmax is used for
inference probabilities; training uses weighted cross entropy as configured.

| Stage | Shape per example | Frozen setting |
| --- | --- | --- |
| input | sequence length 2, scalar value per step | G1, G2 |
| Conv1D | 16 channels x 2 steps | kernel 1 |
| sequence dropout | unchanged | 0.197248 |
| BiLSTM | bidirectional hidden representation | hidden 32, 1 layer |
| pooling/head dropout | vector | 0.456984 |
| linear head | 3 logits | Low/Medium/High |

Optimizer learning rate is 0.0046677, weight decay 0.0003541, batch size 32,
maximum 40 epochs, early-stopping patience 12 and scheduler patience 3. There
are 13,059 trainable parameters. The sequence has only two assessment points;
this architecture must not be interpreted as modeling long-term temporal
dependencies. The final model is single seed 42; the 11-seed ensemble is an
ablation only.
