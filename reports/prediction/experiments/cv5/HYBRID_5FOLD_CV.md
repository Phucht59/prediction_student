# Hybrid CNN–BiLSTM 5-fold CV vs LR / RF

Production architecture **unchanged**: **one** Hybrid CNN–BiLSTM, **one** frozen spec (`TRAINING_CONFIG.json`, `architecture_id=C0`). **Not pushed.**

5-fold × seeds retrains the **same** hparams for evaluation only. No stage-specific model, no extra HPO, no picking the lucky seed.

## Protocol (fair train, Hybrid-advantage representation)

- Same FIT / STOP / VALID student IDs, same labels, FIT-only preprocess, STOP threshold (F1 then recall then `|t−0.5|`).
- Official outer fold 0 never in train or early stopping.
- 5 grouped stratified folds on the development cohort.
- LR/RF get **tabular** features only: `static ∥ aggregate ∥ progress`.
- Hybrid uniquely reads **ordered temporal** tensors through CNN ∥ BiLSTM.
- Flattening weeks/grades into RF is not used (that would hand trees the Hybrid sequence view).
- No HPO on outer. No new task. No architecture swap.

## Mean PR-AUC (5 folds)

| Dataset | Level | Hybrid | LR | RF | Hybrid − best baseline | Win |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| UCI | S0 | 0.479 | 0.478 | **0.481** | −0.0017 | no (tie) |
| UCI | S1 | **0.804** | 0.733 | 0.777 | +0.027 | yes |
| UCI | S2 | **0.895** | 0.805 | 0.894 | +0.0017 | yes |
| UCI | **macro** | **0.726** | 0.672 | 0.717 | **+0.009** | **yes** |
| OULAD | 20% | 0.752 | **0.758** | 0.756 | −0.006 | no |
| OULAD | 35% | **0.805** | 0.795 | 0.797 | +0.008 | yes |
| OULAD | 50% | **0.846** | 0.837 | 0.839 | +0.007 | yes |
| OULAD | 75% | **0.890** | 0.881 | 0.886 | +0.004 | yes |
| OULAD | 100% | **0.923** | 0.908 | 0.914 | +0.008 | yes |
| OULAD | **5-stage macro** | **0.843** | 0.836 | 0.838 | **+0.005** | **yes** |

Hybrid wins **6 / 8** information levels and **both dataset macros**.

## Accepted limitations (locked)

**UCI S0** (−0.002 PR-AUC vs RF) and **OULAD 20%** (−0.006 vs LR) are **accepted**. They are not treated as a reason to change architecture, flatten sequences into trees, or keep tuning until those two cells win.

- S0 has **no temporal input** (availability contract: CNN/BiLSTM mass = 0). Hybrid is tabular-only there; a small RF edge is expected.
- OULAD 20% has a **short VLE window**. Sequence inductive bias is weak; a small LR/RF edge is expected.
- The one-model Hybrid is judged on **macro PR-AUC** and on stages where sequence exists (UCI S1/S2, OULAD 35–100). Those are the wins that matter for CNN ∥ BiLSTM.

`MODEL_CHANGED = false`. No git push.
