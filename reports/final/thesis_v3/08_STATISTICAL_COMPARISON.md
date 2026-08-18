# FINAL AUTHORITY — Paired statistical comparisons

```json
{
  "uci_frozen": {
    "student_mat": [
      {
        "dataset": "student_mat",
        "comparator": "cnn_only",
        "cnn_bilstm_macro_f1": 0.9014601961315334,
        "comparator_macro_f1": 0.8707924528301887,
        "delta_cnn_bilstm_minus_comparator": 0.0306677433013446,
        "ci_95_lower": 0.0042450753802595,
        "ci_95_upper": 0.0579717945645931,
        "verdict": "CNN_BILSTM_HIGHER",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_mat",
        "comparator": "bilstm_only",
        "cnn_bilstm_macro_f1": 0.9014601961315334,
        "comparator_macro_f1": 0.8397186892906668,
        "delta_cnn_bilstm_minus_comparator": 0.0617415068408665,
        "ci_95_lower": 0.0296150069425421,
        "ci_95_upper": 0.0979229555043613,
        "verdict": "CNN_BILSTM_HIGHER",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_mat",
        "comparator": "logistic_regression",
        "cnn_bilstm_macro_f1": 0.9014601961315334,
        "comparator_macro_f1": 0.8793180019908625,
        "delta_cnn_bilstm_minus_comparator": 0.0221421941406708,
        "ci_95_lower": -0.0022944535367629,
        "ci_95_upper": 0.0473148231782832,
        "verdict": "PRACTICAL_TIE",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_mat",
        "comparator": "decision_tree",
        "cnn_bilstm_macro_f1": 0.9014601961315334,
        "comparator_macro_f1": 0.906654414071294,
        "delta_cnn_bilstm_minus_comparator": -0.0051942179397604,
        "ci_95_lower": -0.0213108884719567,
        "ci_95_upper": 0.0119061313165606,
        "verdict": "PRACTICAL_TIE",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_mat",
        "comparator": "random_forest",
        "cnn_bilstm_macro_f1": 0.9014601961315334,
        "comparator_macro_f1": 0.9013865704188284,
        "delta_cnn_bilstm_minus_comparator": 7.362571270486118e-05,
        "ci_95_lower": -0.0199319648418434,
        "ci_95_upper": 0.0213148896697804,
        "verdict": "PRACTICAL_TIE",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_mat",
        "comparator": "hist_gradient_boosting",
        "cnn_bilstm_macro_f1": 0.9014601961315334,
        "comparator_macro_f1": 0.8785461958764588,
        "delta_cnn_bilstm_minus_comparator": 0.0229140002550746,
        "ci_95_lower": -0.0009560874646757,
        "ci_95_upper": 0.0487338427559681,
        "verdict": "PRACTICAL_TIE",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_mat",
        "comparator": "svm",
        "cnn_bilstm_macro_f1": 0.9014601961315334,
        "comparator_macro_f1": 0.8142714606002981,
        "delta_cnn_bilstm_minus_comparator": 0.0871887355312353,
        "ci_95_lower": 0.0489625417936565,
        "ci_95_upper": 0.1268419106557987,
        "verdict": "CNN_BILSTM_HIGHER",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_mat",
        "comparator": "xgboost",
        "cnn_bilstm_macro_f1": 0.9014601961315334,
        "comparator_macro_f1": 0.8880001338150677,
        "delta_cnn_bilstm_minus_comparator": 0.0134600623164656,
        "ci_95_lower": -0.0066423317222973,
        "ci_95_upper": 0.0353468271193678,
        "verdict": "PRACTICAL_TIE",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      }
    ],
    "student_por": [
      {
        "dataset": "student_por",
        "comparator": "cnn_only",
        "cnn_bilstm_macro_f1": 0.8622587167738002,
        "comparator_macro_f1": 0.8468079089978452,
        "delta_cnn_bilstm_minus_comparator": 0.0154508077759549,
        "ci_95_lower": 0.002459251162534,
        "ci_95_upper": 0.029252749365096,
        "verdict": "CNN_BILSTM_HIGHER",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_por",
        "comparator": "bilstm_only",
        "cnn_bilstm_macro_f1": 0.8622587167738002,
        "comparator_macro_f1": 0.784278331756525,
        "delta_cnn_bilstm_minus_comparator": 0.0779803850172752,
        "ci_95_lower": 0.0507090182802036,
        "ci_95_upper": 0.1055571834408966,
        "verdict": "CNN_BILSTM_HIGHER",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_por",
        "comparator": "logistic_regression",
        "cnn_bilstm_macro_f1": 0.8622587167738002,
        "comparator_macro_f1": 0.8205410888586723,
        "delta_cnn_bilstm_minus_comparator": 0.0417176279151279,
        "ci_95_lower": 0.0149976025489263,
        "ci_95_upper": 0.0679141502683394,
        "verdict": "CNN_BILSTM_HIGHER",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_por",
        "comparator": "decision_tree",
        "cnn_bilstm_macro_f1": 0.8622587167738002,
        "comparator_macro_f1": 0.8487179154783511,
        "delta_cnn_bilstm_minus_comparator": 0.0135408012954491,
        "ci_95_lower": -0.0119359326239226,
        "ci_95_upper": 0.0386585728172048,
        "verdict": "PRACTICAL_TIE",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_por",
        "comparator": "random_forest",
        "cnn_bilstm_macro_f1": 0.8622587167738002,
        "comparator_macro_f1": 0.8692436817866236,
        "delta_cnn_bilstm_minus_comparator": -0.0069849650128234,
        "ci_95_lower": -0.0290815495823771,
        "ci_95_upper": 0.0150501046024147,
        "verdict": "PRACTICAL_TIE",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_por",
        "comparator": "hist_gradient_boosting",
        "cnn_bilstm_macro_f1": 0.8622587167738002,
        "comparator_macro_f1": 0.850629682857171,
        "delta_cnn_bilstm_minus_comparator": 0.0116290339166291,
        "ci_95_lower": -0.0165469353836426,
        "ci_95_upper": 0.0397279156605013,
        "verdict": "PRACTICAL_TIE",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_por",
        "comparator": "svm",
        "cnn_bilstm_macro_f1": 0.8622587167738002,
        "comparator_macro_f1": 0.7824770193282303,
        "delta_cnn_bilstm_minus_comparator": 0.0797816974455698,
        "ci_95_lower": 0.0423770186768478,
        "ci_95_upper": 0.1188877077110788,
        "verdict": "CNN_BILSTM_HIGHER",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      },
      {
        "dataset": "student_por",
        "comparator": "xgboost",
        "cnn_bilstm_macro_f1": 0.8622587167738002,
        "comparator_macro_f1": 0.8663880491374553,
        "delta_cnn_bilstm_minus_comparator": -0.004129332363655,
        "ci_95_lower": -0.0276183989974055,
        "ci_95_upper": 0.0198358374778691,
        "verdict": "PRACTICAL_TIE",
        "bootstrap_unit": "record_id",
        "resamples": 5000
      }
    ]
  },
  "oulad_canonical_v3": {
    "oulad": {
      "best_ml": "mlp",
      "ci_lower": -0.00422168555820811,
      "ci_upper": 0.0016682982257444365,
      "delta_macro_f1": -0.0012781366204605016,
      "replicates": 5000
    },
    "student_mat": {
      "best_ml": "decision_tree",
      "ci_lower": -0.08594023882135603,
      "ci_upper": -0.02339929604282368,
      "delta_macro_f1": -0.053813439481764314,
      "replicates": 5000
    },
    "student_por": {
      "best_ml": "random_forest",
      "ci_lower": -0.03846405704645306,
      "ci_upper": 0.004419037218355515,
      "delta_macro_f1": -0.017313134825627685,
      "replicates": 5000
    }
  }
}
```
