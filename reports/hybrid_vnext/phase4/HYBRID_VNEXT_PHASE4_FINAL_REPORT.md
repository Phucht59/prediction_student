# HYBRID VNEXT PHASE 4 — SUPERIORITY FINAL REPORT

**Status:** `NOT_READY_FOR_FINAL_EVAL`

One Hybrid C0, shared structural tuple 128/64/128, UCI evaluated at S0→S1→S2, OULAD evaluated at 20→35→50→75→100.
Active baselines: LR / DT / RF / SVM / MLP. XGBoost is not an active comparator. Outer unused unless both inner gates pass.

## A. Executive conclusion

`NOT_READY_FOR_FINAL_EVAL`

- UCI Hybrid `0.7288` vs `RF` `0.7320` (Δ `-0.0033`, positive stages `2`, ok=`False`)
- OULAD 5-stage Hybrid `0.8451` vs `LR` `0.8392` (Δ `0.0059`, early Δ `0.0051`, positive `4/5`, ok=`True`)
- A tie is not a win. Best seed was not selected. Authority was not updated unless FINAL_HYBRID_ACCEPTED.
- Stop reason: UCI loses to RF on macro (−0.0033) and S0 (−0.0447). OULAD would pass vs LR, but both datasets are required. Outer was not opened.

## B. One-model contract

```json
{
  "allowed_dataset_differences": [
    "input_dimensions",
    "FIT-only preprocessing statistics",
    "categorical vocabulary",
    "learned weights",
    "FIT-derived class prior"
  ],
  "architecture": "C0",
  "forbidden": [
    "dataset_specific_topology",
    "stage_specific_model",
    "stage_specific_checkpoint",
    "separate_oulad_100_model"
  ],
  "oulad": {
    "one_fitted_model": true,
    "states": [
      "20pct",
      "35pct",
      "50pct",
      "75pct",
      "100pct"
    ]
  },
  "outer_test_used": false,
  "public_class": "Hybrid",
  "same_architecture_for_uci_and_oulad": true,
  "same_fusion": true,
  "same_structural_config": true,
  "same_training_strategy_family": true,
  "shared_structural_config": {
    "bilstm_hidden": 128,
    "cnn_channels": 64,
    "d_fuse": 128
  },
  "topology_hash": "b2377c362eeca6deba96903cc5a9375ac05e4f9110767583ae6539490ca1b08f",
  "uci": {
    "one_fitted_model": true,
    "states": [
      "S0",
      "S1",
      "S2"
    ]
  }
}
```

## C. C0 topology integrity

- topology_hash `b2377c362eeca6deba96903cc5a9375ac05e4f9110767583ae6539490ca1b08f`
- parallel CNN ∥ BiLSTM, 3-way masked softmax, availability [1, temporal, temporal]
- Structural HPO was not reopened.

## D. Leakage audit

- pass=`True`
- UCI: `{"fit_stop_valid_disjoint": true, "forbidden": [], "g3_in_predictors": false, "s0_has_g1g2": false, "s1_has_g2_as_latest": false}`
- OULAD: `{"has_100pct": true, "forbidden_in_100pct": [], "n_100pct": 22522, "n_20pct": 26697}`

## E. Overfitting audit

```json
{
  "oulad": {
    "100pct": {
      "gap_mean": 0.02497360222611238,
      "pr_auc": 0.9203951339729622,
      "status": "WELL_FIT",
      "std": 0.005977943763684316
    },
    "20pct": {
      "gap_mean": 0.02497360222611238,
      "pr_auc": 0.7623615500208676,
      "status": "WELL_FIT",
      "std": 0.005977943763684316
    },
    "35pct": {
      "gap_mean": 0.02497360222611238,
      "pr_auc": 0.8058455497428858,
      "status": "WELL_FIT",
      "std": 0.005977943763684316
    },
    "50pct": {
      "gap_mean": 0.02497360222611238,
      "pr_auc": 0.8483327197196115,
      "status": "WELL_FIT",
      "std": 0.005977943763684316
    },
    "75pct": {
      "gap_mean": 0.02497360222611238,
      "pr_auc": 0.8884580908854048,
      "status": "WELL_FIT",
      "std": 0.005977943763684316
    }
  },
  "outer_test_used": false,
  "uci": {
    "S0": {
      "gap_mean": 0.06026939062517876,
      "pr_auc": 0.454744821940755,
      "status": "MIXED",
      "std": 0.024140540307607195
    },
    "S1": {
      "gap_mean": 0.06026939062517876,
      "pr_auc": 0.8214149119441972,
      "status": "WELL_FIT",
      "std": 0.024140540307607195
    },
    "S2": {
      "gap_mean": 0.06026939062517876,
      "pr_auc": 0.9101038055944977,
      "status": "WELL_FIT",
      "std": 0.024140540307607195
    }
  }
}
```

## F. Active baseline ceiling

```json
{
  "uci": {
    "macro": {
      "DT": 0.6682073502839286,
      "DT_std": 0.020551309019107542,
      "LR": 0.7120087568345516,
      "LR_std": 0.03411219566596775,
      "MLP": 0.6953138062473472,
      "MLP_std": 0.047449650341875635,
      "RF": 0.7320438916039742,
      "RF_std": 0.03550767348748176,
      "SVM": 0.7257507294082339,
      "SVM_std": 0.027494438299021616
    },
    "strongest": {
      "macro": 0.7320438916039742,
      "name": "RF"
    }
  },
  "oulad": {
    "macro": {
      "DT": 0.7995536231434563,
      "DT_5stage": 0.7995536231434563,
      "DT_early": 0.7778951117565642,
      "DT_std": 0.005678202761250044,
      "LR": 0.8391909147988836,
      "LR_5stage": 0.8391909147988837,
      "LR_early": 0.8211292853422337,
      "LR_std": 0.005492721511977353,
      "MLP": 0.7940960872850645,
      "MLP_5stage": 0.7940960872850645,
      "MLP_early": 0.7685098649652354,
      "MLP_std": 0.006115861144955971,
      "RF": 0.8373087718871017,
      "RF_5stage": 0.8373087718871016,
      "RF_early": 0.8177772578944504,
      "RF_std": 0.004341686410350088,
      "SVM": 0.8273233815148792,
      "SVM_5stage": 0.8273233815148793,
      "SVM_early": 0.8087072066285788,
      "SVM_std": 0.004583052256043683
    },
    "strongest": {
      "macro": 0.8391909147988836,
      "name": "LR"
    }
  }
}
```

## G. XGBoost removal

- Active roster: `['LR', 'DT', 'RF', 'SVM', 'MLP']`
- Historical provenance preserved: `True`
- Active surface hits: `[]`

## H. SVM integration

```json
{
  "outer_test_used": false,
  "per_dataset": {
    "oulad": {
      "screen_macro": 0.8337172772354083,
      "spec": {
        "C": 1.0,
        "class_weight": "balanced",
        "kernel": "linear"
      }
    },
    "uci": {
      "screen_macro": 0.6946364501779289,
      "spec": {
        "C": 1.0,
        "class_weight": "balanced",
        "gamma": "scale",
        "kernel": "rbf"
      }
    }
  }
}
```

## I. Training superiority ladder

- Selected strategy: `{"name": "L1_control", "outer_test_used": false, "spec": {"curriculum": "C3", "ema": 0.7, "hard_stage_weights": false, "lambda_rank": 0.0, "name": "L1_control", "notes": "Phase3 C0 control, mixed states", "stage_norm": false, "trunc_p": 0.0, "weight_hi": 1.5, "weight_lo": 0.75}}`
- Screen: `{
  "L1_control": {
    "uci": 0.7103179844378973,
    "oulad": 0.8533133182742338
  },
  "L2_stagenorm": {
    "uci": 0.7103179844378973,
    "oulad": 0.8533133182742338
  },
  "L3_C1": {
    "uci": 0.32369132546042195,
    "oulad": 0.8518237217237046
  },
  "L3_C2": {
    "uci": 0.6552615594991883,
    "oulad": 0.8109829315120969
  },
  "L4_hard": {
    "uci": 0.7105282724831775,
    "oulad": 0.8525040069569346
  },
  "L5_trunc": {
    "uci": 0.7065086454182125,
    "oulad": 0.8532121377953027
  },
  "L6_rank05": {
    "uci": 0.710690906811381,
    "oulad": 0.8525775063702603
  }
}`

## J. UCI S0
`0.454744821940755`

## K. UCI S1
`0.8214149119441972`

## L. UCI S2
`0.9101038055944977`

## M. UCI information-growth curve

See `INFORMATION_GROWTH_ANALYSIS.csv`.

## N. OULAD 20
`0.7623615500208676`

## O. OULAD 35
`0.8058455497428858`

## P. OULAD 50
`0.8483327197196115`

## Q. OULAD 75
`0.8884580908854048`

## R. OULAD 100
`0.9203951339729622`

## S. OULAD information-growth curve

See `INFORMATION_GROWTH_ANALYSIS.csv`. 100% remains one state of the same checkpoint. Length≈Withdrawn shortcut is diagnosed, not used as a feature.

## T. Dataset-shift robustness

```json
{
  "do_not_compare_raw_pr": true,
  "oulad_margin_vs_strongest": 0.005887694069462768,
  "outer_test_used": false,
  "uci_margin_vs_strongest": -0.0032893784441575535
}
```

## U. Gate/branch diagnostics

See `GATE_DIAGNOSTICS.csv` when written from robust payloads. UCI S0 must keep tabular_mass=1.

## V. Robust inner superiority

```json
{
  "oulad": {
    "best_baseline": 0.8391909147988837,
    "best_baseline_early": 0.8211292853422337,
    "best_baseline_name": "LR",
    "delta_5stage": 0.005887694069462657,
    "delta_early": 0.005120192249958744,
    "hybrid_5stage": 0.8450786088683464,
    "hybrid_early": 0.8262494775921925,
    "ok": true,
    "positive_stages": 4,
    "stage_delta": {
      "100pct": 0.00895770134747853,
      "20pct": -0.0008519605153730625,
      "35pct": 0.007253779898226176,
      "50pct": 0.008395005889263008,
      "75pct": 0.005683943727718743
    },
    "worst_stage_delta": -0.0008519605153730625
  },
  "outer_test_used": false,
  "ready": false,
  "status": "NOT_READY_FOR_FINAL_EVAL",
  "tie_is_win": false,
  "uci": {
    "best_baseline": 0.7320438916039742,
    "best_baseline_name": "RF",
    "delta": -0.0032893784441575535,
    "hybrid": 0.7287545131598167,
    "ok": false,
    "positive_stages": 2,
    "stage_delta": {
      "S0": -0.04472696432201256,
      "S1": 0.03193402449922156,
      "S2": 0.002924804490318511
    },
    "worst_stage_delta": -0.04472696432201256
  }
}
```

## W. Final outer results if allowed

Outer opens only after both UCI and OULAD strict gates pass.

## X. Paired bootstrap if allowed

Not computed unless outer ran.

## Y. Authority decision

`src/prediction` is updated only if FINAL_HYBRID_ACCEPTED.

## Z. Limitations

- UCI S0 has no temporal signal; trees remain strong on static tabular features.
- OULAD 100% length≈Withdrawn shortcut exists and is reported, not exploited.
- Numeric HPO is applied only after a winning training family is identified.
- Tie ≠ win.

## Required scientific answers

- Q1 UCI beat strongest active? `False`
- Q2 OULAD beat strongest active? `True`
- Q3 same Hybrid across dataset nature? `true` (one C0 / one strategy family)
- Q4 information growth: see section M/S
- Q5 temporal gate: see diagnostics
- Q6 overfit: see section E
- Q7 leakage-safe? `True`
- Q8 mechanism: `L1_control`
- Q9 3×3 stability: UCI std `0.0241` OULAD std `0.0060`
- Q10 defensible superiority? `False`

