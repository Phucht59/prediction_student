# FINAL AUTHORITY — Provenance

```json
{
  "student_mat": {
    "authority": {
      "macro_f1": 0.9014601961315334,
      "checkpoint_directory": "artifacts/final/models/cnn_bilstm_mat",
      "prediction_artifact": "artifacts/final/comparator_completion/student_mat/oof_predictions.parquet",
      "configuration": "configs/final/cnn_bilstm_mat.yaml",
      "metric_artifact": "artifacts/final/metrics/cnn_bilstm_mat.json"
    },
    "architecture_config": {
      "model_id": "cnn_bilstm_mat",
      "display_name": "CNN-BiLSTM MAT",
      "dataset": "student_mat",
      "task": "three_class_student_performance",
      "input_contract": {
        "source": "UCI Student Performance student-mat",
        "classes": [
          "Low",
          "Medium",
          "High"
        ],
        "preprocessing": "fit_on_training_partition_only"
      },
      "architecture": {
        "family": "CNN-BiLSTM",
        "transfer_learning": true,
        "shared_trunk": true,
        "subject_specific_head": true
      },
      "selection": {
        "selected_on": "inner_validation_only",
        "outer_test_used_for_selection": false
      },
      "training_protocol": {
        "outer_folds": 5,
        "fixed_seeds": [
          42,
          1201,
          2026,
          3407,
          7319
        ],
        "ensemble": "mean_probability",
        "training_enabled_in_release": false
      },
      "artifacts": {
        "metrics": "artifacts/final/metrics/cnn_bilstm_mat.json",
        "predictions": "artifacts/final/predictions/cnn_bilstm_mat/oof_predictions.parquet",
        "checkpoints": "artifacts/final/models/cnn_bilstm_mat",
        "tuning_evidence": "artifacts/final/tuning_evidence/cnn_bilstm_mat"
      },
      "provenance": {
        "source_version": "v5_1",
        "original_candidate": "cnn_bilstm_v5_1_transfer_selected_ensemble",
        "role": "canonical_prediction_evidence"
      }
    },
    "metric": {
      "status": "COMPLETE",
      "dataset": "student-mat",
      "candidate": "cnn_bilstm_v5_1_transfer_selected_ensemble",
      "metrics": {
        "candidate": "cnn_bilstm_v5_1_transfer_selected_ensemble",
        "records": 395,
        "accuracy": 0.8911392405063291,
        "balanced_accuracy": 0.9020888215665611,
        "macro_f1": 0.9014601961315334,
        "weighted_f1": 0.8916765397427795,
        "macro_pr_auc": 0.9441838635944574,
        "nll": 0.36351269622218274,
        "confusion_matrix": [
          [
            119,
            11,
            0
          ],
          [
            25,
            165,
            2
          ],
          [
            0,
            5,
            68
          ]
        ],
        "per_class": {
          "low": {
            "precision": 0.8263888888888888,
            "recall": 0.9153846153846154,
            "f1": 0.8686131386861314,
            "support": 130
          },
          "medium": {
            "precision": 0.9116022099447514,
            "recall": 0.859375,
            "f1": 0.8847184986595175,
            "support": 192
          },
          "high": {
            "precision": 0.9714285714285714,
            "recall": 0.9315068493150684,
            "f1": 0.951048951048951,
            "support": 73
          }
        }
      },
      "seed_metrics": [
        {
          "candidate": "bilstm_only_v5_1",
          "seed": 42,
          "records": 395,
          "accuracy": 0.8430379746835444,
          "balanced_accuracy": 0.8561263318112632,
          "macro_f1": 0.8459830082565514,
          "weighted_f1": 0.8425079518471339,
          "macro_pr_auc": 0.8980234568407878,
          "nll": 0.5011542292943157,
          "confusion_matrix": [
            [
              117,
              13,
              0
            ],
            [
              27,
              152,
              13
            ],
            [
              0,
              9,
              64
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8125,
              "recall": 0.9,
              "f1": 0.8540145985401459,
              "support": 130
            },
            "medium": {
              "precision": 0.8735632183908046,
              "recall": 0.7916666666666666,
              "f1": 0.8306010928961749,
              "support": 192
            },
            "high": {
              "precision": 0.8311688311688312,
              "recall": 0.8767123287671232,
              "f1": 0.8533333333333334,
              "support": 73
            }
          },
          "rmse": 3.9711855049159426,
          "r2": 0.24675419534944698
        },
        {
          "candidate": "bilstm_only_v5_1",
          "seed": 2026,
          "records": 395,
          "accuracy": 0.8227848101265823,
          "balanced_accuracy": 0.8377513610818405,
          "macro_f1": 0.8278849337774058,
          "weighted_f1": 0.822153683463879,
          "macro_pr_auc": 0.8890693885090443,
          "nll": 0.5211344283730062,
          "confusion_matrix": [
            [
              115,
              15,
              0
            ],
            [
              32,
              147,
              13
            ],
            [
              0,
              10,
              63
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.782312925170068,
              "recall": 0.8846153846153846,
              "f1": 0.8303249097472925,
              "support": 130
            },
            "medium": {
              "precision": 0.8546511627906976,
              "recall": 0.765625,
              "f1": 0.8076923076923077,
              "support": 192
            },
            "high": {
              "precision": 0.8289473684210527,
              "recall": 0.863013698630137,
              "f1": 0.8456375838926175,
              "support": 73
            }
          },
          "rmse": 3.9369145108627857,
          "r2": 0.25969899184048284
        },
        {
          "candidate": "bilstm_only_v5_1",
          "seed": 3407,
          "records": 395,
          "accuracy": 0.8202531645569621,
          "balanced_accuracy": 0.8428495638683996,
          "macro_f1": 0.8274283795613435,
          "weighted_f1": 0.8193570110243206,
          "macro_pr_auc": 0.8868649946968558,
          "nll": 0.5176155393928163,
          "confusion_matrix": [
            [
              113,
              17,
              0
            ],
            [
              32,
              145,
              15
            ],
            [
              0,
              7,
              66
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7793103448275862,
              "recall": 0.8692307692307693,
              "f1": 0.8218181818181818,
              "support": 130
            },
            "medium": {
              "precision": 0.8579881656804734,
              "recall": 0.7552083333333334,
              "f1": 0.8033240997229917,
              "support": 192
            },
            "high": {
              "precision": 0.8148148148148148,
              "recall": 0.9041095890410958,
              "f1": 0.8571428571428571,
              "support": 73
            }
          },
          "rmse": 3.558547237972269,
          "r2": 0.3951581365712593
        },
        {
          "candidate": "cnn_bilstm_v5_1",
          "seed": 42,
          "records": 395,
          "accuracy": 0.8556962025316456,
          "balanced_accuracy": 0.8732971841704719,
          "macro_f1": 0.860852473649878,
          "weighted_f1": 0.8551905644117229,
          "macro_pr_auc": 0.9273878370858887,
          "nll": 0.414503824873332,
          "confusion_matrix": [
            [
              117,
              13,
              0
            ],
            [
              26,
              154,
              12
            ],
            [
              0,
              6,
              67
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8181818181818182,
              "recall": 0.9,
              "f1": 0.8571428571428571,
              "support": 130
            },
            "medium": {
              "precision": 0.8901734104046243,
              "recall": 0.8020833333333334,
              "f1": 0.8438356164383561,
              "support": 192
            },
            "high": {
              "precision": 0.8481012658227848,
              "recall": 0.9178082191780822,
              "f1": 0.881578947368421,
              "support": 73
            }
          },
          "rmse": 3.508314661122135,
          "r2": 0.4121135519594964
        },
        {
          "candidate": "cnn_bilstm_v5_1",
          "seed": 1201,
          "records": 395,
          "accuracy": 0.8405063291139241,
          "balanced_accuracy": 0.8620525260508137,
          "macro_f1": 0.8455720740141158,
          "weighted_f1": 0.8396854301577937,
          "macro_pr_auc": 0.9108605160097646,
          "nll": 0.4382043915695596,
          "confusion_matrix": [
            [
              116,
              14,
              0
            ],
            [
              28,
              149,
              15
            ],
            [
              0,
              6,
              67
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8055555555555556,
              "recall": 0.8923076923076924,
              "f1": 0.8467153284671532,
              "support": 130
            },
            "medium": {
              "precision": 0.8816568047337278,
              "recall": 0.7760416666666666,
              "f1": 0.8254847645429363,
              "support": 192
            },
            "high": {
              "precision": 0.8170731707317073,
              "recall": 0.9178082191780822,
              "f1": 0.864516129032258,
              "support": 73
            }
          },
          "rmse": 3.935122754930169,
          "r2": 0.2603726853521706
        },
        {
          "candidate": "cnn_bilstm_v5_1",
          "seed": 2026,
          "records": 395,
          "accuracy": 0.8556962025316456,
          "balanced_accuracy": 0.883097339304531,
          "macro_f1": 0.8597839870870189,
          "weighted_f1": 0.8543356171366555,
          "macro_pr_auc": 0.9217667596869136,
          "nll": 0.4316341488763931,
          "confusion_matrix": [
            [
              122,
              8,
              0
            ],
            [
              29,
              147,
              16
            ],
            [
              0,
              4,
              69
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8079470198675497,
              "recall": 0.9384615384615385,
              "f1": 0.8683274021352313,
              "support": 130
            },
            "medium": {
              "precision": 0.9245283018867925,
              "recall": 0.765625,
              "f1": 0.8376068376068376,
              "support": 192
            },
            "high": {
              "precision": 0.8117647058823529,
              "recall": 0.9452054794520548,
              "f1": 0.8734177215189873,
              "support": 73
            }
          },
          "rmse": 3.480868610483403,
          "r2": 0.42127581300543104
        },
        {
          "candidate": "cnn_bilstm_v5_1",
          "seed": 3407,
          "records": 395,
          "accuracy": 0.8607594936708861,
          "balanced_accuracy": 0.8809093636576514,
          "macro_f1": 0.8662012736145136,
          "weighted_f1": 0.8599592266440894,
          "macro_pr_auc": 0.92721772887997,
          "nll": 0.40972581654678214,
          "confusion_matrix": [
            [
              122,
              8,
              0
            ],
            [
              30,
              151,
              11
            ],
            [
              0,
              6,
              67
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8026315789473685,
              "recall": 0.9384615384615385,
              "f1": 0.8652482269503546,
              "support": 130
            },
            "medium": {
              "precision": 0.9151515151515152,
              "recall": 0.7864583333333334,
              "f1": 0.84593837535014,
              "support": 192
            },
            "high": {
              "precision": 0.8589743589743589,
              "recall": 0.9178082191780822,
              "f1": 0.8874172185430463,
              "support": 73
            }
          },
          "rmse": 3.6187140929906465,
          "r2": 0.37453225865241724
        },
        {
          "candidate": "cnn_bilstm_v5_1",
          "seed": 7319,
          "records": 395,
          "accuracy": 0.8556962025316456,
          "balanced_accuracy": 0.8789573820395739,
          "macro_f1": 0.8621474807810673,
          "weighted_f1": 0.854935839806662,
          "macro_pr_auc": 0.9161879962403985,
          "nll": 0.4301016674418975,
          "confusion_matrix": [
            [
              117,
              13,
              0
            ],
            [
              27,
              152,
              13
            ],
            [
              0,
              4,
              69
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8125,
              "recall": 0.9,
              "f1": 0.8540145985401459,
              "support": 130
            },
            "medium": {
              "precision": 0.8994082840236687,
              "recall": 0.7916666666666666,
              "f1": 0.8421052631578947,
              "support": 192
            },
            "high": {
              "precision": 0.8414634146341463,
              "recall": 0.9452054794520548,
              "f1": 0.8903225806451613,
              "support": 73
            }
          },
          "rmse": 3.481393553406132,
          "r2": 0.42110124733938703
        },
        {
          "candidate": "cnn_bilstm_v5_1_transfer_selected",
          "seed": 42,
          "records": 395,
          "accuracy": 0.8911392405063291,
          "balanced_accuracy": 0.9029168130195527,
          "macro_f1": 0.9027565022311513,
          "weighted_f1": 0.8918924120174206,
          "macro_pr_auc": 0.9394002587026068,
          "nll": 0.3780721944480207,
          "confusion_matrix": [
            [
              120,
              10,
              0
            ],
            [
              27,
              164,
              1
            ],
            [
              0,
              5,
              68
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8163265306122449,
              "recall": 0.9230769230769231,
              "f1": 0.8664259927797834,
              "support": 130
            },
            "medium": {
              "precision": 0.9162011173184358,
              "recall": 0.8541666666666666,
              "f1": 0.8840970350404312,
              "support": 192
            },
            "high": {
              "precision": 0.9855072463768116,
              "recall": 0.9315068493150684,
              "f1": 0.9577464788732394,
              "support": 73
            }
          }
        },
        {
          "candidate": "cnn_bilstm_v5_1_transfer_selected",
          "seed": 1201,
          "records": 395,
          "accuracy": 0.8835443037974684,
          "balanced_accuracy": 0.8947426384498302,
          "macro_f1": 0.8956349206349207,
          "weighted_f1": 0.883967989594239,
          "macro_pr_auc": 0.9398795271510044,
          "nll": 0.3651500276275045,
          "confusion_matrix": [
            [
              113,
              17,
              0
            ],
            [
              23,
              167,
              2
            ],
            [
              0,
              4,
              69
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8308823529411765,
              "recall": 0.8692307692307693,
              "f1": 0.849624060150376,
              "support": 130
            },
            "medium": {
              "precision": 0.8882978723404256,
              "recall": 0.8697916666666666,
              "f1": 0.8789473684210526,
              "support": 192
            },
            "high": {
              "precision": 0.971830985915493,
              "recall": 0.9452054794520548,
              "f1": 0.9583333333333334,
              "support": 73
            }
          }
        },
        {
          "candidate": "cnn_bilstm_v5_1_transfer_selected",
          "seed": 2026,
          "records": 395,
          "accuracy": 0.8962025316455696,
          "balanced_accuracy": 0.9043869277602155,
          "macro_f1": 0.9034516816164926,
          "weighted_f1": 0.8965026709190352,
          "macro_pr_auc": 0.9351650955830699,
          "nll": 0.3697327791668166,
          "confusion_matrix": [
            [
              121,
              9,
              0
            ],
            [
              23,
              166,
              3
            ],
            [
              0,
              6,
              67
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8402777777777778,
              "recall": 0.9307692307692308,
              "f1": 0.8832116788321168,
              "support": 130
            },
            "medium": {
              "precision": 0.9171270718232044,
              "recall": 0.8645833333333334,
              "f1": 0.8900804289544236,
              "support": 192
            },
            "high": {
              "precision": 0.9571428571428572,
              "recall": 0.9178082191780822,
              "f1": 0.9370629370629371,
              "support": 73
            }
          }
        },
        {
          "candidate": "cnn_bilstm_v5_1_transfer_selected",
          "seed": 3407,
          "records": 395,
          "accuracy": 0.8810126582278481,
          "balanced_accuracy": 0.8923142781875658,
          "macro_f1": 0.888688811200672,
          "weighted_f1": 0.8811603211644137,
          "macro_pr_auc": 0.9386549702011863,
          "nll": 0.3778645657795454,
          "confusion_matrix": [
            [
              119,
              11,
              0
            ],
            [
              25,
              162,
              5
            ],
            [
              0,
              6,
              67
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8263888888888888,
              "recall": 0.9153846153846154,
              "f1": 0.8686131386861314,
              "support": 130
            },
            "medium": {
              "precision": 0.9050279329608939,
              "recall": 0.84375,
              "f1": 0.8733153638814016,
              "support": 192
            },
            "high": {
              "precision": 0.9305555555555556,
              "recall": 0.9178082191780822,
              "f1": 0.9241379310344827,
              "support": 73
            }
          }
        },
        {
          "candidate": "cnn_bilstm_v5_1_transfer_selected",
          "seed": 7319,
          "records": 395,
          "accuracy": 0.8886075949367088,
          "balanced_accuracy": 0.9006988350310268,
          "macro_f1": 0.897951948215785,
          "weighted_f1": 0.8887828565226654,
          "macro_pr_auc": 0.9350640479997434,
          "nll": 0.35673675477750155,
          "confusion_matrix": [
            [
              116,
              14,
              0
            ],
            [
              22,
              166,
              4
            ],
            [
              0,
              4,
              69
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8405797101449275,
              "recall": 0.8923076923076924,
              "f1": 0.8656716417910447,
              "support": 130
            },
            "medium": {
              "precision": 0.9021739130434783,
              "recall": 0.8645833333333334,
              "f1": 0.8829787234042553,
              "support": 192
            },
            "high": {
              "precision": 0.9452054794520548,
              "recall": 0.9452054794520548,
              "f1": 0.9452054794520548,
              "support": 73
            }
          }
        },
        {
          "candidate": "cnn_only_v5_1",
          "seed": 42,
          "records": 395,
          "accuracy": 0.8632911392405064,
          "balanced_accuracy": 0.8740194356632713,
          "macro_f1": 0.8676480781680903,
          "weighted_f1": 0.863124644386074,
          "macro_pr_auc": 0.9260952119848894,
          "nll": 0.3996273297730598,
          "confusion_matrix": [
            [
              115,
              15,
              0
            ],
            [
              22,
              160,
              10
            ],
            [
              0,
              7,
              66
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8394160583941606,
              "recall": 0.8846153846153846,
              "f1": 0.8614232209737828,
              "support": 130
            },
            "medium": {
              "precision": 0.8791208791208791,
              "recall": 0.8333333333333334,
              "f1": 0.8556149732620321,
              "support": 192
            },
            "high": {
              "precision": 0.868421052631579,
              "recall": 0.9041095890410958,
              "f1": 0.8859060402684564,
              "support": 73
            }
          },
          "rmse": 2.874678153914672,
          "r2": 0.6052929992392979
        },
        {
          "candidate": "cnn_only_v5_1",
          "seed": 2026,
          "records": 395,
          "accuracy": 0.8556962025316456,
          "balanced_accuracy": 0.871641201264489,
          "macro_f1": 0.8615991752264837,
          "weighted_f1": 0.8553510407823222,
          "macro_pr_auc": 0.9227746748988711,
          "nll": 0.3983109603025831,
          "confusion_matrix": [
            [
              115,
              15,
              0
            ],
            [
              25,
              156,
              11
            ],
            [
              0,
              6,
              67
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8214285714285714,
              "recall": 0.8846153846153846,
              "f1": 0.8518518518518519,
              "support": 130
            },
            "medium": {
              "precision": 0.8813559322033898,
              "recall": 0.8125,
              "f1": 0.8455284552845529,
              "support": 192
            },
            "high": {
              "precision": 0.8589743589743589,
              "recall": 0.9178082191780822,
              "f1": 0.8874172185430463,
              "support": 73
            }
          },
          "rmse": 4.377503763526405,
          "r2": 0.08472957667135006
        },
        {
          "candidate": "cnn_only_v5_1",
          "seed": 3407,
          "records": 395,
          "accuracy": 0.8708860759493671,
          "balanced_accuracy": 0.8857159583186981,
          "macro_f1": 0.8772179636625173,
          "weighted_f1": 0.8706635310814399,
          "macro_pr_auc": 0.9366421378748587,
          "nll": 0.37680283993850283,
          "confusion_matrix": [
            [
              116,
              14,
              0
            ],
            [
              23,
              160,
              9
            ],
            [
              0,
              5,
              68
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8345323741007195,
              "recall": 0.8923076923076924,
              "f1": 0.862453531598513,
              "support": 130
            },
            "medium": {
              "precision": 0.8938547486033519,
              "recall": 0.8333333333333334,
              "f1": 0.862533692722372,
              "support": 192
            },
            "high": {
              "precision": 0.8831168831168831,
              "recall": 0.9315068493150684,
              "f1": 0.9066666666666666,
              "support": 73
            }
          },
          "rmse": 4.702821059254738,
          "r2": -0.05636324216622546
        }
      ],
      "ablation_metrics": [
        {
          "candidate": "bilstm_only_v5_1_ensemble",
          "records": 395,
          "accuracy": 0.8354430379746836,
          "balanced_accuracy": 0.8517459899309214,
          "macro_f1": 0.8397186892906668,
          "weighted_f1": 0.8346502630893999,
          "macro_pr_auc": 0.8950111903355852,
          "nll": 0.5100781377909076,
          "confusion_matrix": [
            [
              118,
              12,
              0
            ],
            [
              31,
              148,
              13
            ],
            [
              0,
              9,
              64
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7919463087248322,
              "recall": 0.9076923076923077,
              "f1": 0.8458781362007168,
              "support": 130
            },
            "medium": {
              "precision": 0.8757396449704142,
              "recall": 0.7708333333333334,
              "f1": 0.8199445983379502,
              "support": 192
            },
            "high": {
              "precision": 0.8311688311688312,
              "recall": 0.8767123287671232,
              "f1": 0.8533333333333334,
              "support": 73
            }
          },
          "rmse": 3.7393485381215044,
          "r2": 0.3321356436489704
        },
        {
          "candidate": "cnn_bilstm_v5_1_ensemble",
          "records": 395,
          "accuracy": 0.8582278481012658,
          "balanced_accuracy": 0.8775172696405574,
          "macro_f1": 0.8613798439885395,
          "weighted_f1": 0.8574589673764137,
          "macro_pr_auc": 0.9265559333389951,
          "nll": 0.4219948223985343,
          "confusion_matrix": [
            [
              120,
              10,
              0
            ],
            [
              26,
              152,
              14
            ],
            [
              0,
              6,
              67
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.821917808219178,
              "recall": 0.9230769230769231,
              "f1": 0.8695652173913043,
              "support": 130
            },
            "medium": {
              "precision": 0.9047619047619048,
              "recall": 0.7916666666666666,
              "f1": 0.8444444444444444,
              "support": 192
            },
            "high": {
              "precision": 0.8271604938271605,
              "recall": 0.9178082191780822,
              "f1": 0.8701298701298701,
              "support": 73
            }
          },
          "rmse": 3.3597160914508852,
          "r2": 0.4608600288037077
        },
        {
          "candidate": "cnn_bilstm_v5_1_transfer_selected_ensemble",
          "records": 395,
          "accuracy": 0.8911392405063291,
          "balanced_accuracy": 0.9020888215665611,
          "macro_f1": 0.9014601961315334,
          "weighted_f1": 0.8916765397427795,
          "macro_pr_auc": 0.9441838635944574,
          "nll": 0.36351269622218274,
          "confusion_matrix": [
            [
              119,
              11,
              0
            ],
            [
              25,
              165,
              2
            ],
            [
              0,
              5,
              68
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8263888888888888,
              "recall": 0.9153846153846154,
              "f1": 0.8686131386861314,
              "support": 130
            },
            "medium": {
              "precision": 0.9116022099447514,
              "recall": 0.859375,
              "f1": 0.8847184986595175,
              "support": 192
            },
            "high": {
              "precision": 0.9714285714285714,
              "recall": 0.9315068493150684,
              "f1": 0.951048951048951,
              "support": 73
            }
          }
        },
        {
          "candidate": "cnn_only_v5_1_ensemble",
          "records": 395,
          "accuracy": 0.8658227848101265,
          "balanced_accuracy": 0.8777576542559419,
          "macro_f1": 0.8707924528301887,
          "weighted_f1": 0.8656364302205238,
          "macro_pr_auc": 0.9300255199405209,
          "nll": 0.38917112922124664,
          "confusion_matrix": [
            [
              114,
              16,
              0
            ],
            [
              21,
              161,
              10
            ],
            [
              0,
              6,
              67
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.8444444444444444,
              "recall": 0.8769230769230769,
              "f1": 0.8603773584905661,
              "support": 130
            },
            "medium": {
              "precision": 0.8797814207650273,
              "recall": 0.8385416666666666,
              "f1": 0.8586666666666667,
              "support": 192
            },
            "high": {
              "precision": 0.8701298701298701,
              "recall": 0.9178082191780822,
              "f1": 0.8933333333333333,
              "support": 73
            }
          },
          "rmse": 3.634845226089821,
          "r2": 0.3689435378513032
        }
      ],
      "seed_stability": {
        "mean": 0.8976967727798044,
        "std": 0.005369907977256859,
        "min": 0.888688811200672,
        "max": 0.9034516816164926
      },
      "future_accessed": false
    },
    "prediction_sha256": "d7810e249a44d05230579db7362e49407874f0374b6f6e788978411ea7c8e76c",
    "metric_sha256": "ee4e998ca6e8173f374d78984ba5eba52105c1ae57501e93397bfc4e08aff776",
    "configuration_sha256": "3ec5452d5539fb96ce0982838c229c9317cf3a6acedb23e3eaf9de3ef9f27a69",
    "checkpoint_count": 25,
    "checkpoint_sha256": {
      "outer_0_seed_1201.pt": "cffce69ea266295dc08b482e5c0f949c5a74a04fc3d7ecf7c9b237fd27276b90",
      "outer_0_seed_2026.pt": "0415ee6021c56879427572e0a413d3ff7c59a8d6d1234bab278cbc5c3725376c",
      "outer_0_seed_3407.pt": "3ec7f5b0da63f44e537a1ae5b49f49c728e1244df4f654f27a21c298a195057c",
      "outer_0_seed_42.pt": "199517b13a2be2373b51e06ddd470b92a6fb20ad5b5ea7160b28228e28a27047",
      "outer_0_seed_7319.pt": "9a26059bfd870e768c8772ff07806daf5c714f1ac4f0c0da86c24c54298d387f",
      "outer_1_seed_1201.pt": "4eec714d23072ea415758edfa377c696c364f76add67cbca1ec0e16781ac9295",
      "outer_1_seed_2026.pt": "83529618c137c7c5186a2f7437025a1cb5d0d2e390e49c8b29fce4b872e48c6c",
      "outer_1_seed_3407.pt": "19589a7f7457b50986045e5a9fe6b3bb5f27518d3f504cac5bca7e9565293857",
      "outer_1_seed_42.pt": "83c68c8a666ea834abcadcb37cc931564227a6e124ec947c5fbf7b494f2544e4",
      "outer_1_seed_7319.pt": "d9e97fd6515c87b84fec072009a7b602e38515d107a0984d0c1b20888161c507",
      "outer_2_seed_1201.pt": "2a8bfac5a491cdf729d4ef0bd3e0d50ffc55ed4762db02a6079dac5d82a466e9",
      "outer_2_seed_2026.pt": "f907e6d80d12900f0de971c58a459c99e5c01aebbaf9f0e61a44f171ac1258a9",
      "outer_2_seed_3407.pt": "1fd7ed4245635a819a686cf38f366c67552fce53f9269b02a2b626d8d87b09fd",
      "outer_2_seed_42.pt": "3635d5c39a65b48d32252a4d97504d1d4bce9d8657151da55a4e47fe226f06fa",
      "outer_2_seed_7319.pt": "68e79c107516728f181e004fc6a1a1af99f4019fca98dc5271c7b641548e1d4d",
      "outer_3_seed_1201.pt": "cf24bb87b2d65c65d654a83282be9263c27127813f73526bdc2c59dbe66f4e48",
      "outer_3_seed_2026.pt": "f4e385b756f678e817a99477d141da65e30a9e6c0ab3e3dc3d074a974da38653",
      "outer_3_seed_3407.pt": "b4daa1ab0c793e79b32b79e7269a4b261c8c3c7fe54d66438897817ef45bfcba",
      "outer_3_seed_42.pt": "7e5c36c6794a90ee9328f01875aee7848306a30994600325b2f57f28691098b1",
      "outer_3_seed_7319.pt": "ccae4f5650bd1ff9e89d47b026efb8636a3d5086e4a46708174e4977df2f6481",
      "outer_4_seed_1201.pt": "213c51eb77b6bae629d96110b7e0de4ba307a6225397d1f21a00aec5e34c66e1",
      "outer_4_seed_2026.pt": "fa74456657dd842897dd63628e832975767c2bc3f0a7f9fde0e6e1a0a1a6e347",
      "outer_4_seed_3407.pt": "6bda20c96f167c20bef904504f1578c52fe8e5b01637d707678bfa610f100a98",
      "outer_4_seed_42.pt": "561c3f79d0dc56ac8d2a11997ef1971fc75d7c16389234d4741f9484581cdda9",
      "outer_4_seed_7319.pt": "6492dfa6533822e877c8260c977a5deb56de1d4f7f121d8c71f003b7b6737b8f"
    },
    "protocol_snapshot": {
      "schema_version": "student_mat_v5_1_protocol_v1",
      "protocol_status": "frozen_before_outer_evaluation",
      "dataset": "student-mat",
      "source": {
        "path": "data/raw/student-mat.csv",
        "sha256": "e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80",
        "delimiter": ";",
        "expected_rows": 395
      },
      "target": {
        "source_column": "G3",
        "classification": {
          "low": [
            0,
            9
          ],
          "medium": [
            10,
            14
          ],
          "high": [
            15,
            20
          ]
        },
        "regression": "raw_G3_0_to_20",
        "G3_as_input": "PROHIBITED"
      },
      "features": {
        "temporal_source": [
          "G1",
          "G2"
        ],
        "temporal_channels": [
          "normalized_grade",
          "stage_indicator",
          "signed_change_from_G1",
          "absolute_change_from_G1",
          "signed_distance_to_boundary_10",
          "signed_distance_to_boundary_15",
          "change_direction"
        ],
        "construction": "deterministic_from_G1_G2_only",
        "ml_receives_equivalent_flattened_temporal_features": true,
        "primary_safe_context": [
          "failures",
          "studytime",
          "schoolsup",
          "famsup",
          "paid",
          "activities",
          "internet",
          "higher",
          "traveltime",
          "freetime",
          "goout",
          "health"
        ],
        "primary_context_excludes": [
          "absences"
        ],
        "sensitivity_only_context": [
          "absences"
        ],
        "sensitive_primary_exclusions": [
          "sex",
          "age",
          "address",
          "famsize",
          "Pstatus",
          "Mjob",
          "Fjob"
        ],
        "preprocessing": "fit_current_training_partition_only",
        "unknown_category": "ignore_with_all_zero_onehot"
      },
      "splits": {
        "outer_manifest": "artifacts/v5/student_mat/split_manifest.csv",
        "outer_manifest_sha256": "3b1dbfc8e359f415e70e1e607a0f14b44d26bfc1bd0616650d0c2346509171f5",
        "outer_folds": 5,
        "inner_folds": 3,
        "inner_method": "StratifiedGroupKFold_with_conservative_quasi_identity",
        "split_seed": 42,
        "cross_subject_validation_overlap": "PROHIBITED"
      },
      "architecture": {
        "temporal": {
          "input_projection": [
            16,
            24,
            32
          ],
          "parallel_kernels": [
            [
              1,
              2
            ]
          ],
          "channels_per_branch": [
            8,
            16,
            24,
            32,
            48
          ],
          "activation": [
            "GELU",
            "ReLU"
          ],
          "normalization": "LayerNorm",
          "residual_projection": "required",
          "dropout": [
            0.05,
            0.15,
            0.25,
            0.4
          ],
          "bilstm_layers": [
            1,
            2
          ],
          "bilstm_hidden": [
            16,
            24,
            32,
            48,
            64
          ]
        },
        "context": {
          "layers": [
            1,
            2
          ],
          "hidden": [
            16,
            24,
            32,
            48,
            64,
            96
          ],
          "normalization": "LayerNorm",
          "max_size_relative_to_temporal": "not_larger"
        },
        "fusion_candidates": [
          "concatenation",
          "gated",
          "film_residual"
        ],
        "objectives": [
          "classification_only",
          "classification_plus_huber_regression",
          "classification_plus_huber_regression_plus_ordinal"
        ],
        "regression_weight_candidates": [
          0.05,
          0.1,
          0.2
        ],
        "ordinal_weight_candidates": [
          0.025,
          0.05,
          0.1
        ],
        "parameter_limit": 1500000
      },
      "transfer": {
        "source_dataset": "student-por",
        "candidates": [
          "standalone",
          "por_pretrain_mat_freeze_unfreeze",
          "shared_trunk_subject_specific_heads"
        ],
        "option_a": {
          "freeze_epochs": [
            3,
            5
          ],
          "unfreeze_learning_rate_fraction": [
            0.2,
            0.35
          ]
        },
        "option_b": {
          "subject_embedding_dim": [
            4,
            8
          ],
          "heads": [
            "math",
            "portuguese"
          ]
        },
        "quasi_identity_columns": [
          "school",
          "sex",
          "age",
          "address",
          "famsize",
          "Pstatus",
          "Medu",
          "Fedu",
          "Mjob",
          "Fjob",
          "reason",
          "nursery",
          "internet"
        ],
        "source_group_overlapping_target_validation": "excluded",
        "target_preprocessing_fit_on_target_train_only": true,
        "transferred_objects": "neural_weights_only",
        "retain_only_on_stable_inner_improvement": true
      },
      "comparators": {
        "primary": "decision_tree",
        "standard": [
          "logistic_regression",
          "decision_tree",
          "random_forest",
          "svm"
        ],
        "extended": [
          "hist_gradient_boosting"
        ],
        "fair_raw_information": [
          "G1",
          "G2",
          "primary_safe_context"
        ],
        "imbalance_tabular": [
          "none",
          "class_weight",
          "random_oversampling",
          "smote",
          "adasyn"
        ],
        "imbalance_deep": [
          "none",
          "class_weight",
          "random_sample_duplication",
          "focal"
        ]
      },
      "search": {
        "round_a_trials_per_component_group": 12,
        "round_a_component_groups": [
          "fusion",
          "objective",
          "transfer",
          "imbalance"
        ],
        "round_a_seeds": [
          42,
          2026,
          3407
        ],
        "round_b_trials_min": 40,
        "round_b_trials_max": 80,
        "storage": "artifacts/v5_1/student_mat/optuna.db",
        "sampler_seed": 3407,
        "plateau_non_improving_trials": 15
      },
      "evaluation": {
        "seeds": [
          42,
          1201,
          2026,
          3407,
          7319
        ],
        "primary_metric": "macro_f1",
        "primary_comparator": "decision_tree",
        "baseline_v5_macro_f1": 0.8799168720699821,
        "directional_target_macro_f1": 0.89,
        "metrics": [
          "accuracy",
          "macro_f1",
          "weighted_f1",
          "per_class_precision_recall_f1",
          "macro_pr_auc",
          "rmse",
          "r2"
        ],
        "performance_target_is_gate": false
      },
      "artifacts": {
        "root": "artifacts/v5_1/student_mat",
        "report_root": "reports/v5_1/student_mat",
        "future_accessed": false
      }
    }
  },
  "student_por": {
    "authority": {
      "macro_f1": 0.8622587167738002,
      "checkpoint_directory": "artifacts/final/models/cnn_bilstm_por",
      "prediction_artifact": "artifacts/final/comparator_completion/student_por/oof_predictions.parquet",
      "configuration": "configs/final/cnn_bilstm_por.yaml",
      "metric_artifact": "artifacts/final/metrics/cnn_bilstm_por.json"
    },
    "architecture_config": {
      "model_id": "cnn_bilstm_por",
      "display_name": "CNN-BiLSTM POR",
      "dataset": "student_por",
      "task": "three_class_student_performance",
      "input_contract": {
        "source": "UCI Student Performance student-por",
        "classes": [
          "Low",
          "Medium",
          "High"
        ],
        "preprocessing": "fit_on_training_partition_only"
      },
      "architecture": {
        "family": "CNN-BiLSTM",
        "ensemble": true
      },
      "selection": {
        "selected_on": "inner_validation_only",
        "outer_test_used_for_selection": false
      },
      "training_protocol": {
        "outer_folds": 5,
        "fixed_seeds": [
          42,
          1201,
          2026,
          3407,
          7319
        ],
        "ensemble": "mean_probability",
        "training_enabled_in_release": false
      },
      "artifacts": {
        "metrics": "artifacts/final/metrics/cnn_bilstm_por.json",
        "predictions": "artifacts/final/predictions/cnn_bilstm_por/oof_predictions.parquet",
        "checkpoints": "artifacts/final/models/cnn_bilstm_por",
        "tuning_evidence": "artifacts/final/tuning_evidence/cnn_bilstm_por"
      },
      "provenance": {
        "source_version": "v5_1",
        "original_candidate": "cnn_bilstm_v5_1_ensemble",
        "role": "canonical_prediction_evidence"
      }
    },
    "metric": {
      "status": "COMPLETE",
      "dataset": "student-por",
      "candidate": "cnn_bilstm_v5_1_ensemble",
      "metrics": {
        "candidate": "cnn_bilstm_v5_1_ensemble",
        "records": 649,
        "accuracy": 0.889060092449923,
        "balanced_accuracy": 0.8675763663148155,
        "macro_f1": 0.8622587167738002,
        "weighted_f1": 0.8896472207203324,
        "macro_pr_auc": 0.914678708867879,
        "nll": 0.3078934731046347,
        "confusion_matrix": [
          [
            78,
            22,
            0
          ],
          [
            27,
            379,
            12
          ],
          [
            0,
            11,
            120
          ]
        ],
        "per_class": {
          "low": {
            "precision": 0.7428571428571429,
            "recall": 0.78,
            "f1": 0.7609756097560976,
            "support": 100
          },
          "medium": {
            "precision": 0.9199029126213593,
            "recall": 0.9066985645933014,
            "f1": 0.9132530120481928,
            "support": 418
          },
          "high": {
            "precision": 0.9090909090909091,
            "recall": 0.916030534351145,
            "f1": 0.9125475285171103,
            "support": 131
          }
        },
        "rmse": 2.3496525355971873,
        "r2": 0.4702200964858734
      },
      "seed_metrics": [
        {
          "candidate": "bilstm_only_v5_1",
          "seed": 42,
          "records": 649,
          "accuracy": 0.802773497688752,
          "balanced_accuracy": 0.7771038630580615,
          "macro_f1": 0.7604282636457361,
          "weighted_f1": 0.8050459981983534,
          "macro_pr_auc": 0.8514356198564577,
          "nll": 0.4107187058553136,
          "confusion_matrix": [
            [
              62,
              38,
              0
            ],
            [
              50,
              342,
              26
            ],
            [
              0,
              14,
              117
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.5535714285714286,
              "recall": 0.62,
              "f1": 0.5849056603773585,
              "support": 100
            },
            "medium": {
              "precision": 0.868020304568528,
              "recall": 0.8181818181818182,
              "f1": 0.8423645320197044,
              "support": 418
            },
            "high": {
              "precision": 0.8181818181818182,
              "recall": 0.8931297709923665,
              "f1": 0.8540145985401459,
              "support": 131
            }
          },
          "rmse": 2.3663216107176526,
          "r2": 0.4626766274620612
        },
        {
          "candidate": "bilstm_only_v5_1",
          "seed": 2026,
          "records": 649,
          "accuracy": 0.8289676425269645,
          "balanced_accuracy": 0.7976488062627074,
          "macro_f1": 0.7861483881286638,
          "weighted_f1": 0.8285662487850093,
          "macro_pr_auc": 0.8779683061175655,
          "nll": 0.39058441132974664,
          "confusion_matrix": [
            [
              62,
              38,
              0
            ],
            [
              36,
              355,
              27
            ],
            [
              0,
              10,
              121
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.6326530612244898,
              "recall": 0.62,
              "f1": 0.6262626262626263,
              "support": 100
            },
            "medium": {
              "precision": 0.8808933002481389,
              "recall": 0.8492822966507177,
              "f1": 0.8647990255785627,
              "support": 418
            },
            "high": {
              "precision": 0.8175675675675675,
              "recall": 0.9236641221374046,
              "f1": 0.8673835125448028,
              "support": 131
            }
          },
          "rmse": 3.0027308561545833,
          "r2": 0.13479104534665876
        },
        {
          "candidate": "bilstm_only_v5_1",
          "seed": 3407,
          "records": 649,
          "accuracy": 0.8366718027734977,
          "balanced_accuracy": 0.8041719322595177,
          "macro_f1": 0.7948448179558353,
          "weighted_f1": 0.8362509057268862,
          "macro_pr_auc": 0.8540534156372303,
          "nll": 0.39612303969853974,
          "confusion_matrix": [
            [
              63,
              37,
              0
            ],
            [
              35,
              359,
              24
            ],
            [
              0,
              10,
              121
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.6428571428571429,
              "recall": 0.63,
              "f1": 0.6363636363636364,
              "support": 100
            },
            "medium": {
              "precision": 0.8842364532019704,
              "recall": 0.8588516746411483,
              "f1": 0.8713592233009708,
              "support": 418
            },
            "high": {
              "precision": 0.8344827586206897,
              "recall": 0.9236641221374046,
              "f1": 0.8768115942028986,
              "support": 131
            }
          },
          "rmse": 2.4155700425982873,
          "r2": 0.44007809014782306
        },
        {
          "candidate": "cnn_bilstm_v5_1",
          "seed": 42,
          "records": 649,
          "accuracy": 0.8767334360554699,
          "balanced_accuracy": 0.8594496998916444,
          "macro_f1": 0.8490640266765621,
          "weighted_f1": 0.8781181102224425,
          "macro_pr_auc": 0.9072672890984892,
          "nll": 0.31417006271176595,
          "confusion_matrix": [
            [
              78,
              22,
              0
            ],
            [
              33,
              372,
              13
            ],
            [
              0,
              12,
              119
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7027027027027027,
              "recall": 0.78,
              "f1": 0.7393364928909952,
              "support": 100
            },
            "medium": {
              "precision": 0.916256157635468,
              "recall": 0.8899521531100478,
              "f1": 0.9029126213592233,
              "support": 418
            },
            "high": {
              "precision": 0.9015151515151515,
              "recall": 0.9083969465648855,
              "f1": 0.9049429657794676,
              "support": 131
            }
          },
          "rmse": 2.7472258385305444,
          "r2": 0.27576932255886644
        },
        {
          "candidate": "cnn_bilstm_v5_1",
          "seed": 1201,
          "records": 649,
          "accuracy": 0.8859784283513097,
          "balanced_accuracy": 0.8607402266944252,
          "macro_f1": 0.8587652045851004,
          "weighted_f1": 0.8862971301782381,
          "macro_pr_auc": 0.9154911773248265,
          "nll": 0.30933706235222747,
          "confusion_matrix": [
            [
              78,
              22,
              0
            ],
            [
              25,
              380,
              13
            ],
            [
              0,
              14,
              117
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7572815533980582,
              "recall": 0.78,
              "f1": 0.7684729064039408,
              "support": 100
            },
            "medium": {
              "precision": 0.9134615384615384,
              "recall": 0.9090909090909091,
              "f1": 0.9112709832134293,
              "support": 418
            },
            "high": {
              "precision": 0.9,
              "recall": 0.8931297709923665,
              "f1": 0.896551724137931,
              "support": 131
            }
          },
          "rmse": 2.166322285865282,
          "r2": 0.5496664253861395
        },
        {
          "candidate": "cnn_bilstm_v5_1",
          "seed": 2026,
          "records": 649,
          "accuracy": 0.8813559322033898,
          "balanced_accuracy": 0.8620115173429758,
          "macro_f1": 0.8529426343027918,
          "weighted_f1": 0.8818516939886659,
          "macro_pr_auc": 0.9084213266767267,
          "nll": 0.3095358964603327,
          "confusion_matrix": [
            [
              76,
              24,
              0
            ],
            [
              28,
              374,
              16
            ],
            [
              0,
              9,
              122
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7307692307692307,
              "recall": 0.76,
              "f1": 0.7450980392156863,
              "support": 100
            },
            "medium": {
              "precision": 0.918918918918919,
              "recall": 0.8947368421052632,
              "f1": 0.9066666666666666,
              "support": 418
            },
            "high": {
              "precision": 0.8840579710144928,
              "recall": 0.9312977099236641,
              "f1": 0.9070631970260223,
              "support": 131
            }
          },
          "rmse": 3.058823474669155,
          "r2": 0.10216398784363834
        },
        {
          "candidate": "cnn_bilstm_v5_1",
          "seed": 3407,
          "records": 649,
          "accuracy": 0.8859784283513097,
          "balanced_accuracy": 0.8761250106529334,
          "macro_f1": 0.8616981079175016,
          "weighted_f1": 0.8876746452753023,
          "macro_pr_auc": 0.9161103489824423,
          "nll": 0.31147897323425744,
          "confusion_matrix": [
            [
              82,
              18,
              0
            ],
            [
              33,
              373,
              12
            ],
            [
              0,
              11,
              120
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7130434782608696,
              "recall": 0.82,
              "f1": 0.7627906976744186,
              "support": 100
            },
            "medium": {
              "precision": 0.927860696517413,
              "recall": 0.8923444976076556,
              "f1": 0.9097560975609756,
              "support": 418
            },
            "high": {
              "precision": 0.9090909090909091,
              "recall": 0.916030534351145,
              "f1": 0.9125475285171103,
              "support": 131
            }
          },
          "rmse": 2.854356242923684,
          "r2": 0.2181840216658706
        },
        {
          "candidate": "cnn_bilstm_v5_1",
          "seed": 7319,
          "records": 649,
          "accuracy": 0.889060092449923,
          "balanced_accuracy": 0.8786781840096425,
          "macro_f1": 0.8647926454007075,
          "weighted_f1": 0.8900382149591783,
          "macro_pr_auc": 0.9088937475985753,
          "nll": 0.319753258929417,
          "confusion_matrix": [
            [
              81,
              19,
              0
            ],
            [
              28,
              374,
              16
            ],
            [
              0,
              9,
              122
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7431192660550459,
              "recall": 0.81,
              "f1": 0.7751196172248804,
              "support": 100
            },
            "medium": {
              "precision": 0.9303482587064676,
              "recall": 0.8947368421052632,
              "f1": 0.9121951219512195,
              "support": 418
            },
            "high": {
              "precision": 0.8840579710144928,
              "recall": 0.9312977099236641,
              "f1": 0.9070631970260223,
              "support": 131
            }
          },
          "rmse": 2.8141574757284715,
          "r2": 0.24005006117441208
        },
        {
          "candidate": "cnn_only_v5_1",
          "seed": 42,
          "records": 649,
          "accuracy": 0.8782742681047766,
          "balanced_accuracy": 0.8543865736513386,
          "macro_f1": 0.8485529860714115,
          "weighted_f1": 0.8787941245298898,
          "macro_pr_auc": 0.9181295876731003,
          "nll": 0.298189590615781,
          "confusion_matrix": [
            [
              75,
              25,
              0
            ],
            [
              29,
              375,
              14
            ],
            [
              0,
              11,
              120
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7211538461538461,
              "recall": 0.75,
              "f1": 0.7352941176470589,
              "support": 100
            },
            "medium": {
              "precision": 0.9124087591240876,
              "recall": 0.8971291866028708,
              "f1": 0.9047044632086851,
              "support": 418
            },
            "high": {
              "precision": 0.8955223880597015,
              "recall": 0.916030534351145,
              "f1": 0.9056603773584906,
              "support": 131
            }
          },
          "rmse": 2.735247935581556,
          "r2": 0.2820708458345813
        },
        {
          "candidate": "cnn_only_v5_1",
          "seed": 2026,
          "records": 649,
          "accuracy": 0.8767334360554699,
          "balanced_accuracy": 0.8569138147241803,
          "macro_f1": 0.8484857857990377,
          "weighted_f1": 0.8774784997965747,
          "macro_pr_auc": 0.920127155937486,
          "nll": 0.30367885073453055,
          "confusion_matrix": [
            [
              77,
              23,
              0
            ],
            [
              29,
              373,
              16
            ],
            [
              0,
              12,
              119
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7264150943396226,
              "recall": 0.77,
              "f1": 0.7475728155339806,
              "support": 100
            },
            "medium": {
              "precision": 0.9142156862745098,
              "recall": 0.8923444976076556,
              "f1": 0.9031476997578692,
              "support": 418
            },
            "high": {
              "precision": 0.8814814814814815,
              "recall": 0.9083969465648855,
              "f1": 0.8947368421052632,
              "support": 131
            }
          },
          "rmse": 3.0429832339665794,
          "r2": 0.11143887030767141
        },
        {
          "candidate": "cnn_only_v5_1",
          "seed": 3407,
          "records": 649,
          "accuracy": 0.8828967642526965,
          "balanced_accuracy": 0.8626394925551213,
          "macro_f1": 0.855592904857246,
          "weighted_f1": 0.883732970106681,
          "macro_pr_auc": 0.9159689896855197,
          "nll": 0.2954359304961607,
          "confusion_matrix": [
            [
              78,
              22,
              0
            ],
            [
              29,
              376,
              13
            ],
            [
              0,
              12,
              119
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7289719626168224,
              "recall": 0.78,
              "f1": 0.7536231884057971,
              "support": 100
            },
            "medium": {
              "precision": 0.9170731707317074,
              "recall": 0.8995215311004785,
              "f1": 0.9082125603864735,
              "support": 418
            },
            "high": {
              "precision": 0.9015151515151515,
              "recall": 0.9083969465648855,
              "f1": 0.9049429657794676,
              "support": 131
            }
          },
          "rmse": 3.6841941243774903,
          "r2": -0.30248641766969797
        }
      ],
      "ablation_metrics": [
        {
          "candidate": "bilstm_only_v5_1_ensemble",
          "records": 649,
          "accuracy": 0.8258859784283513,
          "balanced_accuracy": 0.7985897950984331,
          "macro_f1": 0.784278331756525,
          "weighted_f1": 0.8261780035111856,
          "macro_pr_auc": 0.8648504032238492,
          "nll": 0.3952449079576304,
          "confusion_matrix": [
            [
              63,
              37,
              0
            ],
            [
              39,
              352,
              27
            ],
            [
              0,
              10,
              121
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.6176470588235294,
              "recall": 0.63,
              "f1": 0.6237623762376238,
              "support": 100
            },
            "medium": {
              "precision": 0.8822055137844611,
              "recall": 0.8421052631578947,
              "f1": 0.8616891064871481,
              "support": 418
            },
            "high": {
              "precision": 0.8175675675675675,
              "recall": 0.9236641221374046,
              "f1": 0.8673835125448028,
              "support": 131
            }
          },
          "rmse": 2.389419996102858,
          "r2": 0.45213547571963875
        },
        {
          "candidate": "cnn_bilstm_v5_1_ensemble",
          "records": 649,
          "accuracy": 0.889060092449923,
          "balanced_accuracy": 0.8675763663148155,
          "macro_f1": 0.8622587167738002,
          "weighted_f1": 0.8896472207203324,
          "macro_pr_auc": 0.914678708867879,
          "nll": 0.3078934731046347,
          "confusion_matrix": [
            [
              78,
              22,
              0
            ],
            [
              27,
              379,
              12
            ],
            [
              0,
              11,
              120
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7428571428571429,
              "recall": 0.78,
              "f1": 0.7609756097560976,
              "support": 100
            },
            "medium": {
              "precision": 0.9199029126213593,
              "recall": 0.9066985645933014,
              "f1": 0.9132530120481928,
              "support": 418
            },
            "high": {
              "precision": 0.9090909090909091,
              "recall": 0.916030534351145,
              "f1": 0.9125475285171103,
              "support": 131
            }
          },
          "rmse": 2.3496525355971873,
          "r2": 0.4702200964858734
        },
        {
          "candidate": "cnn_only_v5_1_ensemble",
          "records": 649,
          "accuracy": 0.8767334360554699,
          "balanced_accuracy": 0.8518420443892522,
          "macro_f1": 0.8468079089978452,
          "weighted_f1": 0.8772553774833354,
          "macro_pr_auc": 0.9214543817473171,
          "nll": 0.29590530593351183,
          "confusion_matrix": [
            [
              75,
              25,
              0
            ],
            [
              29,
              375,
              14
            ],
            [
              0,
              12,
              119
            ]
          ],
          "per_class": {
            "low": {
              "precision": 0.7211538461538461,
              "recall": 0.75,
              "f1": 0.7352941176470589,
              "support": 100
            },
            "medium": {
              "precision": 0.9101941747572816,
              "recall": 0.8971291866028708,
              "f1": 0.9036144578313253,
              "support": 418
            },
            "high": {
              "precision": 0.8947368421052632,
              "recall": 0.9083969465648855,
              "f1": 0.9015151515151515,
              "support": 131
            }
          },
          "rmse": 3.022715826377535,
          "r2": 0.12323575279189658
        }
      ],
      "seed_stability": {
        "mean": 0.8574525237765327,
        "std": 0.0057329135301778505,
        "min": 0.8490640266765621,
        "max": 0.8647926454007075
      },
      "future_accessed": false
    },
    "prediction_sha256": "14258e818e14c9cf5b5bd21077453f85869bc5605984b17b03dfd4b818d72670",
    "metric_sha256": "5ad0710721b0323a0e2e47371b983b9ad0bd6eb4b12a27e18a69b38e7c982a25",
    "configuration_sha256": "6c4c9f85c5b494e662d2dfe4e186841df332523d868411f602f266903a6c59d1",
    "checkpoint_count": 25,
    "checkpoint_sha256": {
      "outer_0_seed_1201.pt": "1c85f5e83134418bc382f6a77063ab313ac02fb5994f0a1ee9dddd9f4eb5ccfc",
      "outer_0_seed_2026.pt": "e885ad886154fa5a93359ba150213fc772d1af72136d34a22595b90989a74e77",
      "outer_0_seed_3407.pt": "2d5eb18bf444c7cd1b4f74206dfe054dc0f777378458c021a3d339195efbc18a",
      "outer_0_seed_42.pt": "3ccb2e2114fbf6b3cbcf625254d4e1fe5966f508b5385077db5e0e7bb533ef67",
      "outer_0_seed_7319.pt": "25462853b26eb7bd06677447a4b37308f6d289c96cd554716c161961f7bd486e",
      "outer_1_seed_1201.pt": "184b4647723756df23357e1e3fbb9794504a9b7bed8be5266ff198604df2464d",
      "outer_1_seed_2026.pt": "00379af5fbe66a52b974dd9fbe831c427d37fd74f6284e0596b1a2a85f52bbc6",
      "outer_1_seed_3407.pt": "9185754d60320cae9394ecd420d054c616b9feabeb603e450726a111276ee16a",
      "outer_1_seed_42.pt": "dcfa7a80f154ade7451d846a2fcb5b0511902728b6a8f3ae111980836726dbb2",
      "outer_1_seed_7319.pt": "5b2a24f0a88bb32f7c7b178814a609cb557ff76480bef6b7cacfd82279f0607b",
      "outer_2_seed_1201.pt": "023ce27d2200aaefbff1d20b1145842c8faf13ede7610e7551977bf0e3abdc17",
      "outer_2_seed_2026.pt": "071de1423ebab7e99820fba7712a45cf108e94dca3b2c74f68c54c9f1f4e4ac8",
      "outer_2_seed_3407.pt": "fc78e2eed383aae7554caf74247bb36325bc57694e6deb75988516510cfd81a4",
      "outer_2_seed_42.pt": "b9201d854098e70f838b277ec12b28053f810c7da53f912ad5a22c0247e19a84",
      "outer_2_seed_7319.pt": "d40ecc9019016ab2d23e8dedcd8530fa95579f856b529ee3d0c680f745c01e2f",
      "outer_3_seed_1201.pt": "5c0e7f34fe02057c6cef00a855bbc78fdeacdbf76a47a95ef8bf9b1a51ac01f1",
      "outer_3_seed_2026.pt": "a6132b282b822dc2dc0f94608efd9308ac5b8e320bb99ab0b50a75c25f60c95b",
      "outer_3_seed_3407.pt": "b8618b8ed56fc3041bf5230cf419990be1340962cfdadfa08d3faaa4f9953f30",
      "outer_3_seed_42.pt": "0caac3cdb0bf6121d8e44c440bb476620b2458f4298014973ba7de32d3e164c8",
      "outer_3_seed_7319.pt": "42a7ff5627e6833bbd600a0859745253c9c0a0abe727b4a67c3ecdb45e3be90f",
      "outer_4_seed_1201.pt": "7ac665fc5f4cd3965182af1b89c0621b6d678d1a42f7f2a3db92eb600782ac72",
      "outer_4_seed_2026.pt": "bfc79be6772613c86c74176cb4646b5445643b615b3ea860cff702a381606ddc",
      "outer_4_seed_3407.pt": "6c6246a61e7c4428225e2fc12f74bc4db251ea7b2a4f5fdc2f66e73c9b10081a",
      "outer_4_seed_42.pt": "49e27165cba1402771f056431dcae5dc3e3d6e282bb5262d4fccc5b7fd023df7",
      "outer_4_seed_7319.pt": "71bd330f19b187c3543f3f94d5a94414f40509f77c19d84a4290b3d9d834526d"
    },
    "protocol_snapshot": {
      "schema_version": "student_por_v5_1_protocol_v1",
      "protocol_status": "frozen_before_outer_evaluation",
      "dataset": "student-por",
      "source": {
        "path": "data/raw/student-por.csv",
        "sha256": "a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f",
        "delimiter": ";",
        "expected_rows": 649
      },
      "target": {
        "source_column": "G3",
        "classification": {
          "low": [
            0,
            9
          ],
          "medium": [
            10,
            14
          ],
          "high": [
            15,
            20
          ]
        },
        "regression": "raw_G3_0_to_20",
        "G3_as_input": "PROHIBITED"
      },
      "features": {
        "temporal_source": [
          "G1",
          "G2"
        ],
        "temporal_channels": [
          "normalized_grade",
          "stage_indicator",
          "signed_change_from_G1",
          "absolute_change_from_G1",
          "signed_distance_to_boundary_10",
          "signed_distance_to_boundary_15",
          "change_direction"
        ],
        "construction": "deterministic_from_G1_G2_only",
        "ml_receives_equivalent_flattened_temporal_features": true,
        "primary_safe_context": [
          "failures",
          "studytime",
          "schoolsup",
          "famsup",
          "paid",
          "activities",
          "internet",
          "higher",
          "traveltime",
          "freetime",
          "goout",
          "health"
        ],
        "primary_context_excludes": [
          "absences"
        ],
        "sensitivity_only_context": [
          "absences"
        ],
        "sensitive_primary_exclusions": [
          "sex",
          "age",
          "address",
          "famsize",
          "Pstatus",
          "Mjob",
          "Fjob"
        ],
        "preprocessing": "fit_current_training_partition_only",
        "unknown_category": "ignore_with_all_zero_onehot"
      },
      "splits": {
        "outer_manifest": "artifacts/v5/student_por/split_manifest.csv",
        "outer_manifest_sha256": "2ea07b2d17714fc9df3eec23150579090d245a1a45f33270b2864c88653d2de2",
        "outer_folds": 5,
        "inner_folds": 3,
        "inner_method": "StratifiedGroupKFold_with_conservative_quasi_identity",
        "split_seed": 1201,
        "cross_subject_results_are_external_validation": false
      },
      "architecture": {
        "temporal": {
          "input_projection": [
            16,
            24,
            32
          ],
          "parallel_kernels": [
            [
              1,
              2
            ]
          ],
          "channels_per_branch": [
            8,
            16,
            24,
            32,
            48
          ],
          "activation": [
            "GELU",
            "ReLU"
          ],
          "normalization": "LayerNorm",
          "residual_projection": "required",
          "dropout": [
            0.05,
            0.15,
            0.25,
            0.4
          ],
          "bilstm_layers": [
            1,
            2
          ],
          "bilstm_hidden": [
            16,
            24,
            32,
            48,
            64
          ]
        },
        "context": {
          "layers": [
            1,
            2
          ],
          "hidden": [
            16,
            24,
            32,
            48,
            64,
            96
          ],
          "normalization": "LayerNorm",
          "max_size_relative_to_temporal": "not_larger"
        },
        "fusion_candidates": [
          "concatenation",
          "gated",
          "film_residual"
        ],
        "objectives": [
          "classification_only",
          "classification_plus_huber_regression",
          "classification_plus_huber_regression_plus_ordinal"
        ],
        "regression_weight_candidates": [
          0.05,
          0.1,
          0.2
        ],
        "ordinal_weight_candidates": [
          0.025,
          0.05,
          0.1
        ],
        "parameter_limit": 1500000
      },
      "comparators": {
        "primary": "random_forest",
        "standard": [
          "logistic_regression",
          "decision_tree",
          "random_forest",
          "svm"
        ],
        "extended": [
          "hist_gradient_boosting"
        ],
        "fair_raw_information": [
          "G1",
          "G2",
          "primary_safe_context"
        ],
        "imbalance_tabular": [
          "none",
          "class_weight",
          "random_oversampling",
          "smote",
          "adasyn"
        ],
        "imbalance_deep": [
          "none",
          "class_weight",
          "random_sample_duplication",
          "focal"
        ]
      },
      "search": {
        "round_a_trials_per_component_group": 12,
        "round_a_component_groups": [
          "fusion",
          "objective",
          "imbalance"
        ],
        "round_a_seeds": [
          42,
          2026,
          3407
        ],
        "round_b_trials_min": 40,
        "round_b_trials_max": 80,
        "storage": "artifacts/v5_1/student_por/optuna.db",
        "sampler_seed": 7319,
        "plateau_non_improving_trials": 15
      },
      "evaluation": {
        "seeds": [
          42,
          1201,
          2026,
          3407,
          7319
        ],
        "primary_metric": "macro_f1",
        "primary_comparator": "random_forest",
        "baseline_v5_macro_f1": 0.8491516177055304,
        "directional_target_macro_f1": 0.86,
        "metrics": [
          "accuracy",
          "macro_f1",
          "weighted_f1",
          "per_class_precision_recall_f1",
          "macro_pr_auc",
          "rmse",
          "r2"
        ],
        "performance_target_is_gate": false
      },
      "artifacts": {
        "root": "artifacts/v5_1/student_por",
        "report_root": "reports/v5_1/student_por",
        "future_accessed": false
      }
    }
  }
}
```
