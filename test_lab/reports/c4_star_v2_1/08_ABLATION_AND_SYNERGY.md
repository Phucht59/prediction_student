# 08 Ablation and synergy

Ladder M0–M7 is the mechanism ablation (fold 0 seed 42).
UCI: GroupDRO (M4+) slightly better J than M0 but still far below CatBoost.
OULAD: KD (M2+) reduces warm-loss penalty vs M0/M1; HPO then found M4 with n_warm_loss=0 on fold 0.
CNN-only / BiLSTM-only sequence-only roster was not a separate 3×3 (C4 branch_mode ablations remain fold-0).
Do not claim CNN–BiLSTM synergy on UCI: T≤2 and CatBoost still dominates S2.
