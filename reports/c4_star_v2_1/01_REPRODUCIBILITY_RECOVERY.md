# 01 Reproducibility recovery

Parent protocol `eb5f4cfbf4e1`. C4 protocol `ce758268ce0c`.

Numbers below are **recomputed** as mean-of-run AP from raw OOF (not copied from Markdown). Pooled-row AP differs and is not the protocol statistic.

## UCI baseline mean-of-run AP (3×3)

```json
{
  "S0": {
    "CatBoost": 0.5009713365027397,
    "DT": 0.4466145738781581,
    "LR": 0.4649642235403744,
    "MLP": 0.44214861300612385,
    "RF": 0.48629895379405724,
    "SVM": 0.4379743074753967,
    "XGB": 0.4550505666440529
  },
  "S1": {
    "CatBoost": 0.7693939811848465,
    "DT": 0.7345847070064621,
    "LR": 0.7416838321874065,
    "MLP": 0.7021355286758938,
    "RF": 0.7211343947865287,
    "SVM": 0.7383814824872343,
    "XGB": 0.7430178103255514
  },
  "S2": {
    "CatBoost": 0.9066812358700744,
    "DT": 0.8842841254968836,
    "LR": 0.8762892044077405,
    "MLP": 0.8396081865175851,
    "RF": 0.9047654273571337,
    "SVM": 0.8842752571079178,
    "XGB": 0.8989973866780557
  }
}
```

CatBoost S0/S1/S2 lock 0.5010 / 0.7694 / 0.9067 is **VERIFIED** against OOF.

## OULAD baseline mean-of-run AP (SPEED: fold0 × 2 seeds)

```json
{
  "100pct": {
    "CatBoost": 0.9223063436963366,
    "DT": 0.890916187920384,
    "LR": 0.9240451333898554,
    "MLP": 0.9230560962156411,
    "RF": 0.9223648652627766,
    "SVM": 0.9243688581520625,
    "XGB": 0.9259512199917121
  },
  "20pct": {
    "CatBoost": 0.766541258438799,
    "DT": 0.6974109353982441,
    "LR": 0.7683617160669742,
    "MLP": 0.7660563780802494,
    "RF": 0.7483666042837493,
    "SVM": 0.7656891902025144,
    "XGB": 0.7658982979514888
  },
  "35pct": {
    "CatBoost": 0.8070300341886554,
    "DT": 0.7558040748837669,
    "LR": 0.8086826592702222,
    "MLP": 0.8078166903672241,
    "RF": 0.7891334780917689,
    "SVM": 0.8034742501600078,
    "XGB": 0.8056536049200227
  },
  "50pct": {
    "CatBoost": 0.8557322877721052,
    "DT": 0.8079514317723889,
    "LR": 0.8558637057978324,
    "MLP": 0.8531568349593313,
    "RF": 0.8493668746433054,
    "SVM": 0.8545435524159122,
    "XGB": 0.8562628731609592
  },
  "75pct": {
    "CatBoost": 0.8983982841370268,
    "DT": 0.8561257170779578,
    "LR": 0.8988558433965916,
    "MLP": 0.8984596310897048,
    "RF": 0.8938655090238938,
    "SVM": 0.8964664720033488,
    "XGB": 0.8980229904137227
  }
}
```

SPEED lock XGB 100% 0.9260 / LR 20% 0.7684 is **VERIFIED**. This ceiling is **not confirmatory** (truncated HPO).

## OULAD C0-R hybrid OOF

```json
{
  "100pct": 0.9230144821448701,
  "20pct": 0.7608985261148193,
  "35pct": 0.8088641083472928,
  "50pct": 0.8575801721726197,
  "75pct": 0.8968728663023159
}
```

## Missing / UNVERIFIED

- UCI Hybrid per-record OOF parquet: **missing** (only robust JSON means). Robust C0-R JSON exists; per-row UCI Hybrid OOF is UNVERIFIED.
- OULAD diagnose shuffle/reverse: **missing** (SPEED skipped).
- Outer test predictions: **absent** (correct).
- Ablation `full` 8-epoch AP~0.32 is under-convergence, not a synergy result.

## Outer test

No `confirmation.json` pass. `outer_test_used` flags in locks are false.
