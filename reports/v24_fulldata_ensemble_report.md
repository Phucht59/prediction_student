# V24 Full-Data Ensemble Report

**Strategy**: Fixed 70/15/15 split, full data training per member, TTA
**Architecture**: SE-MultiScale CNN-BiLSTM + Deep Residual MLP ensemble
**Training**: Focal Loss + Mixup + CosineAnnealing + diverse seeds/archs

## Final Results

| Dataset | Members | Uniform F1 | Top-K F1 | Weighted F1 | **Best F1** | V18 Best | Paper F1 | Beat V18? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xapi | 63 | 0.7254 | 0.7265 | 0.7269 | **0.7269** | 0.7841 | 0.8447 | ❌ |

## Key Innovations vs V23
- **Full data training**: No K-Fold data splitting → each member uses 70% train data
- **SE Block**: Squeeze-and-Excitation channel attention in CNN
- **Stacked attention**: Multiple MHSA layers for richer sequence modeling
- **Deep Residual MLP**: Complementary architecture for ensemble diversity
- **Test-Time Augmentation**: 3x TTA with feature noise
- **Top-K selection**: Best half of members for final ensemble

## Member Details

| Dataset | Member | Feature | Model | Seed | Val F1 |
| --- | --- | --- | --- | --- | --- |
| xapi | full/deep_res_mlp/seed314 | full | deep_res_mlp | 314 | 0.7888 |
| xapi | full/deep_res_mlp/seed42 | full | deep_res_mlp | 42 | 0.7848 |
| xapi | full/deep_res_mlp/seed999 | full | deep_res_mlp | 999 | 0.7785 |
| xapi | full/deep_res_mlp/seed1234 | full | deep_res_mlp | 1234 | 0.7721 |
| xapi | full/secnn_bilstm/seed123 | full | secnn_bilstm | 123 | 0.7712 |
| xapi | full/deep_res_mlp/seed777 | full | deep_res_mlp | 777 | 0.7702 |
| xapi | full/deep_res_mlp/seed2024 | full | deep_res_mlp | 2024 | 0.7702 |
| xapi | paper/secnn_bilstm/seed1234 | paper | secnn_bilstm | 1234 | 0.7660 |
| xapi | paper/deep_res_mlp/seed2024 | paper | deep_res_mlp | 2024 | 0.7659 |
| xapi | behavior8/secnn_bilstm/seed999 | behavior8 | secnn_bilstm | 999 | 0.7606 |
