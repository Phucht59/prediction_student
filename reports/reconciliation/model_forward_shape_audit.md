# Code-level forward and parameter audit

The active model class is unchanged between the historical and V2 source paths. A code-created historical final-config model has 13,059 trainable parameters and, for input `[N=2,L=2,F=1]`, executes Conv input `[2,1,2]` → Conv/BatchNorm/ReLU output `[2,16,2]` → BiLSTM input `[2,2,16]`, hidden `[directions=2,N=2,H=32]` → linear output `[2,3]`. This confirms `hidden[-2] || hidden[-1]` and Low/Medium/High output shape.

V2 tuned fold 0 uses 32 channels and kernel size 2, yielding 17,251 parameters and Conv output `[2,32,3]` because symmetric padding expands an even-sized kernel on a length-two input. The LSTM therefore receives length 3. This is a real configuration/representation difference, not a tensor-order bug. Historical and legacy-refit config use kernel 1 and length 2.
