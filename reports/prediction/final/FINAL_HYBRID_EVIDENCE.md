# Final Hybrid Evidence

**Evidence scope: matched repeated inner-development cross-validation. Outer test remained sealed.**

Frozen model: unified A2 topology with dataset-specific Phase7D BCE training configurations. Phase7E AP was rejected.

## Final UCI comparison

| domain   | model               | stage   |   pr_auc_mean |   pr_auc_std |   roc_auc_mean |   roc_auc_std |   risk_recall_mean |   risk_recall_std |   risk_f1_mean |   risk_f1_std |
|:---------|:--------------------|:--------|--------------:|-------------:|---------------:|--------------:|-------------------:|------------------:|---------------:|--------------:|
| uci      | logistic_regression | S0      |      0.475388 |     0.030139 |       0.756826 |      0.028598 |           0.741387 |          0.127412 |       0.470332 |      0.012291 |
| uci      | random_forest       | S0      |      0.493887 |     0.044199 |       0.797187 |      0.022411 |           0.820826 |          0.082796 |       0.507171 |      0.037011 |
| uci      | xgboost             | S0      |      0.487203 |     0.055684 |       0.757263 |      0.017956 |           0.674971 |          0.116834 |       0.471454 |      0.030750 |
| uci      | catboost            | S0      |      0.489205 |     0.056023 |       0.771391 |      0.026078 |           0.750502 |          0.080910 |       0.494636 |      0.029669 |
| uci      | final_hybrid        | S0      |      0.491300 |     0.047493 |       0.758143 |      0.025764 |           0.658816 |          0.106545 |       0.465315 |      0.020891 |
| uci      | logistic_regression | S1      |      0.779417 |     0.043116 |       0.927650 |      0.009379 |           0.775773 |          0.076135 |       0.699948 |      0.017081 |
| uci      | random_forest       | S1      |      0.790970 |     0.052383 |       0.941163 |      0.015968 |           0.769469 |          0.140122 |       0.705211 |      0.025291 |
| uci      | xgboost             | S1      |      0.772115 |     0.049582 |       0.931423 |      0.016399 |           0.849641 |          0.107833 |       0.704956 |      0.025755 |
| uci      | catboost            | S1      |      0.780839 |     0.044989 |       0.937215 |      0.014933 |           0.757641 |          0.118648 |       0.707552 |      0.029649 |
| uci      | final_hybrid        | S1      |      0.806571 |     0.028279 |       0.940963 |      0.008623 |           0.799786 |          0.127395 |       0.701755 |      0.026240 |
| uci      | logistic_regression | S2      |      0.881220 |     0.035635 |       0.961288 |      0.010108 |           0.831555 |          0.065376 |       0.798846 |      0.025867 |
| uci      | random_forest       | S2      |      0.907566 |     0.022976 |       0.966769 |      0.008196 |           0.817662 |          0.093117 |       0.787341 |      0.047023 |
| uci      | xgboost             | S2      |      0.891514 |     0.022944 |       0.957857 |      0.009800 |           0.845174 |          0.059333 |       0.770088 |      0.023908 |
| uci      | catboost            | S2      |      0.901813 |     0.025533 |       0.963192 |      0.009363 |           0.818054 |          0.112402 |       0.770729 |      0.043572 |
| uci      | final_hybrid        | S2      |      0.894955 |     0.023708 |       0.966885 |      0.006848 |           0.825089 |          0.105455 |       0.781092 |      0.040969 |

## Final OULAD comparison

| domain   | model               | stage   |   pr_auc_mean |   pr_auc_std |   roc_auc_mean |   roc_auc_std |   risk_recall_mean |   risk_recall_std |   risk_f1_mean |   risk_f1_std |
|:---------|:--------------------|:--------|--------------:|-------------:|---------------:|--------------:|-------------------:|------------------:|---------------:|--------------:|
| oulad    | logistic_regression | 20pct   |      0.761439 |     0.004029 |       0.790999 |      0.002083 |           0.749458 |          0.016858 |       0.682827 |      0.010696 |
| oulad    | random_forest       | 20pct   |      0.752895 |     0.004470 |       0.779576 |      0.001050 |           0.747621 |          0.024714 |       0.672884 |      0.006656 |
| oulad    | xgboost             | 20pct   |      0.763601 |     0.004243 |       0.787072 |      0.002983 |           0.734878 |          0.040704 |       0.675438 |      0.006722 |
| oulad    | catboost            | 20pct   |      0.766755 |     0.004085 |       0.790574 |      0.001316 |           0.746781 |          0.036492 |       0.677947 |      0.003251 |
| oulad    | final_hybrid        | 20pct   |      0.765710 |     0.006024 |       0.791628 |      0.005030 |           0.761731 |          0.034117 |       0.683136 |      0.006570 |
| oulad    | logistic_regression | 35pct   |      0.800397 |     0.006460 |       0.828107 |      0.002716 |           0.730976 |          0.033305 |       0.698895 |      0.005370 |
| oulad    | random_forest       | 35pct   |      0.795800 |     0.004490 |       0.819443 |      0.002872 |           0.739468 |          0.049651 |       0.688074 |      0.004853 |
| oulad    | xgboost             | 35pct   |      0.807004 |     0.005341 |       0.828325 |      0.004961 |           0.703844 |          0.043297 |       0.693821 |      0.005031 |
| oulad    | catboost            | 35pct   |      0.808005 |     0.005480 |       0.830265 |      0.004001 |           0.702072 |          0.041703 |       0.695920 |      0.003749 |
| oulad    | final_hybrid        | 35pct   |      0.808871 |     0.005103 |       0.832571 |      0.003174 |           0.749278 |          0.044030 |       0.702266 |      0.006219 |
| oulad    | logistic_regression | 50pct   |      0.843479 |     0.006373 |       0.870910 |      0.004620 |           0.711563 |          0.031667 |       0.730899 |      0.007888 |
| oulad    | random_forest       | 50pct   |      0.841941 |     0.006478 |       0.866849 |      0.004815 |           0.717561 |          0.023579 |       0.724479 |      0.007978 |
| oulad    | xgboost             | 50pct   |      0.848766 |     0.007166 |       0.873045 |      0.005642 |           0.732075 |          0.032988 |       0.733514 |      0.009231 |
| oulad    | catboost            | 50pct   |      0.849860 |     0.006699 |       0.874827 |      0.004910 |           0.722264 |          0.034714 |       0.735001 |      0.007775 |
| oulad    | final_hybrid        | 50pct   |      0.849811 |     0.008431 |       0.874374 |      0.005635 |           0.724572 |          0.037709 |       0.734626 |      0.013568 |
| oulad    | logistic_regression | 75pct   |      0.885666 |     0.007651 |       0.910200 |      0.006478 |           0.704691 |          0.022230 |       0.778234 |      0.011920 |
| oulad    | random_forest       | 75pct   |      0.886248 |     0.006266 |       0.909205 |      0.003588 |           0.713271 |          0.039570 |       0.779503 |      0.009420 |
| oulad    | xgboost             | 75pct   |      0.891016 |     0.007744 |       0.913964 |      0.005156 |           0.727278 |          0.021910 |       0.784866 |      0.011561 |
| oulad    | catboost            | 75pct   |      0.891518 |     0.006379 |       0.914741 |      0.004391 |           0.730554 |          0.025058 |       0.788140 |      0.011813 |
| oulad    | final_hybrid        | 75pct   |      0.888627 |     0.007488 |       0.912328 |      0.004798 |           0.727707 |          0.039639 |       0.780814 |      0.012162 |

## Semantic branch ablation

| domain   | model              | stage   |   pr_auc_mean |   pr_auc_std |   risk_recall_mean |   risk_recall_std |   risk_f1_mean |   risk_f1_std |   generalization_gap_mean |   generalization_gap_std |   parameter_count |   final_pr_auc |   final_minus_ablation |
|:---------|:-------------------|:--------|--------------:|-------------:|-------------------:|------------------:|---------------:|--------------:|--------------------------:|-------------------------:|------------------:|---------------:|-----------------------:|
| oulad    | B0_static          | 20pct   |      0.573186 |     0.008527 |           0.890275 |          0.040645 |       0.610383 |      0.010493 |                  0.025525 |                 0.017717 |            495297 |       0.765710 |               0.192524 |
| oulad    | B0_static          | 35pct   |      0.549040 |     0.007870 |           0.875229 |          0.039839 |       0.589785 |      0.009991 |                  0.026785 |                 0.015180 |            495297 |       0.808871 |               0.259831 |
| oulad    | B0_static          | 50pct   |      0.527402 |     0.007775 |           0.846316 |          0.070485 |       0.566685 |      0.012655 |                  0.026820 |                 0.014161 |            495297 |       0.849811 |               0.322410 |
| oulad    | B0_static          | 75pct   |      0.487869 |     0.004746 |           0.796864 |          0.056967 |       0.534237 |      0.006434 |                  0.027246 |                 0.008548 |            495297 |       0.888627 |               0.400758 |
| oulad    | B1_cnn             | 20pct   |      0.682611 |     0.011665 |           0.799093 |          0.031623 |       0.653608 |      0.007913 |                  0.015569 |                 0.012306 |            495297 |       0.765710 |               0.083099 |
| oulad    | B1_cnn             | 35pct   |      0.769679 |     0.011734 |           0.723428 |          0.045480 |       0.672386 |      0.009925 |                  0.017459 |                 0.016405 |            495297 |       0.808871 |               0.039192 |
| oulad    | B1_cnn             | 50pct   |      0.822819 |     0.006891 |           0.701992 |          0.035274 |       0.710835 |      0.007740 |                  0.013517 |                 0.010238 |            495297 |       0.849811 |               0.026992 |
| oulad    | B1_cnn             | 75pct   |      0.863618 |     0.007247 |           0.712530 |          0.021245 |       0.754643 |      0.008368 |                  0.009770 |                 0.010373 |            495297 |       0.888627 |               0.025009 |
| oulad    | B2_bilstm          | 20pct   |      0.682738 |     0.008759 |           0.793893 |          0.043456 |       0.651200 |      0.004211 |                  0.022202 |                 0.013440 |            495297 |       0.765710 |               0.082972 |
| oulad    | B2_bilstm          | 35pct   |      0.778945 |     0.007763 |           0.743187 |          0.052744 |       0.675509 |      0.007668 |                  0.026313 |                 0.014380 |            495297 |       0.808871 |               0.029926 |
| oulad    | B2_bilstm          | 50pct   |      0.832003 |     0.007641 |           0.730869 |          0.025251 |       0.715882 |      0.009850 |                  0.018856 |                 0.013163 |            495297 |       0.849811 |               0.017809 |
| oulad    | B2_bilstm          | 75pct   |      0.877847 |     0.005209 |           0.696995 |          0.032400 |       0.771247 |      0.007198 |                  0.015378 |                 0.009252 |            495297 |       0.888627 |               0.010779 |
| oulad    | B3_cnn_bilstm      | 20pct   |      0.685999 |     0.009221 |           0.798422 |          0.029291 |       0.653034 |      0.005521 |                  0.022854 |                 0.013122 |            495297 |       0.765710 |               0.079711 |
| oulad    | B3_cnn_bilstm      | 35pct   |      0.779452 |     0.007869 |           0.749712 |          0.039782 |       0.678180 |      0.007081 |                  0.025575 |                 0.011948 |            495297 |       0.808871 |               0.029419 |
| oulad    | B3_cnn_bilstm      | 50pct   |      0.833309 |     0.006801 |           0.718082 |          0.044091 |       0.714880 |      0.008327 |                  0.019196 |                 0.010078 |            495297 |       0.849811 |               0.016503 |
| oulad    | B3_cnn_bilstm      | 75pct   |      0.879196 |     0.006098 |           0.706362 |          0.021032 |       0.773451 |      0.006191 |                  0.015735 |                 0.009166 |            495297 |       0.888627 |               0.009431 |
| oulad    | B4_static_temporal | 20pct   |      0.755482 |     0.007655 |           0.770140 |          0.019875 |       0.683466 |      0.007383 |                  0.030104 |                 0.011272 |            495297 |       0.765710 |               0.010228 |
| oulad    | B4_static_temporal | 35pct   |      0.802456 |     0.006576 |           0.765735 |          0.026408 |       0.700568 |      0.004875 |                  0.027729 |                 0.010739 |            495297 |       0.808871 |               0.006415 |
| oulad    | B4_static_temporal | 50pct   |      0.844275 |     0.007560 |           0.718965 |          0.045188 |       0.728377 |      0.009078 |                  0.021808 |                 0.013244 |            495297 |       0.849811 |               0.005536 |
| oulad    | B4_static_temporal | 75pct   |      0.886990 |     0.008350 |           0.718046 |          0.030308 |       0.778676 |      0.014275 |                  0.016202 |                 0.013260 |            495297 |       0.888627 |               0.001637 |
| oulad    | B5_final           | 20pct   |      0.765710 |     0.006024 |           0.761731 |          0.034117 |       0.683136 |      0.006570 |                  0.028933 |                 0.009527 |            495297 |       0.765710 |               0.000000 |
| oulad    | B5_final           | 35pct   |      0.808871 |     0.005103 |           0.749278 |          0.044030 |       0.702266 |      0.006219 |                  0.027146 |                 0.007882 |            495297 |       0.808871 |               0.000000 |
| oulad    | B5_final           | 50pct   |      0.849811 |     0.008431 |           0.724572 |          0.037709 |       0.734626 |      0.013568 |                  0.021300 |                 0.013596 |            495297 |       0.849811 |               0.000000 |
| oulad    | B5_final           | 75pct   |      0.888627 |     0.007488 |           0.727707 |          0.039639 |       0.780814 |      0.012162 |                  0.016826 |                 0.011455 |            495297 |       0.888627 |               0.000000 |
| uci      | B0_static          | S0      |      0.453790 |     0.063970 |           0.648199 |          0.249712 |       0.430291 |      0.095710 |                  0.245279 |                 0.146690 |            494337 |       0.491300 |               0.037510 |
| uci      | B0_static          | S1      |      0.453790 |     0.063970 |           0.648199 |          0.249712 |       0.430291 |      0.095710 |                  0.245279 |                 0.146690 |            494337 |       0.806571 |               0.352781 |
| uci      | B0_static          | S2      |      0.453790 |     0.063970 |           0.648199 |          0.249712 |       0.430291 |      0.095710 |                  0.245279 |                 0.146690 |            494337 |       0.894955 |               0.441165 |
| uci      | B1_cnn             | S0      |      0.214335 |     0.004897 |           1.000000 |          0.000000 |       0.351510 |      0.005468 |                  0.004858 |                 0.006011 |            494337 |       0.491300 |               0.276965 |
| uci      | B1_cnn             | S1      |      0.770516 |     0.047789 |           0.726248 |          0.139625 |       0.702731 |      0.022476 |                  0.013338 |                 0.057419 |            494337 |       0.806571 |               0.036055 |
| uci      | B1_cnn             | S2      |      0.904099 |     0.035924 |           0.850581 |          0.129299 |       0.764863 |      0.039269 |                  0.007163 |                 0.041325 |            494337 |       0.894955 |              -0.009144 |
| uci      | B2_bilstm          | S0      |      0.213923 |     0.003510 |           1.000000 |          0.000000 |       0.351510 |      0.005468 |                  0.005270 |                 0.004494 |            494337 |       0.491300 |               0.277377 |
| uci      | B2_bilstm          | S1      |      0.764377 |     0.042226 |           0.726248 |          0.139625 |       0.702731 |      0.022476 |                  0.014081 |                 0.051141 |            494337 |       0.806571 |               0.042194 |
| uci      | B2_bilstm          | S2      |      0.904128 |     0.032918 |           0.829015 |          0.132711 |       0.765315 |      0.037174 |                  0.004996 |                 0.041351 |            494337 |       0.894955 |              -0.009173 |
| uci      | B3_cnn_bilstm      | S0      |      0.213243 |     0.004017 |           1.000000 |          0.000000 |       0.351510 |      0.005468 |                  0.005950 |                 0.005133 |            494337 |       0.491300 |               0.278057 |
| uci      | B3_cnn_bilstm      | S1      |      0.773407 |     0.044173 |           0.726248 |          0.139625 |       0.702731 |      0.022476 |                  0.010500 |                 0.053168 |            494337 |       0.806571 |               0.033164 |
| uci      | B3_cnn_bilstm      | S2      |      0.904506 |     0.035261 |           0.854735 |          0.139145 |       0.760099 |      0.034950 |                  0.005817 |                 0.041034 |            494337 |       0.894955 |              -0.009551 |
| uci      | B4_static_temporal | S0      |      0.498952 |     0.042209 |           0.754370 |          0.131464 |       0.472869 |      0.025704 |                  0.212205 |                 0.039477 |            494337 |       0.491300 |              -0.007652 |
| uci      | B4_static_temporal | S1      |      0.793231 |     0.037868 |           0.831056 |          0.106639 |       0.699979 |      0.021626 |                  0.103187 |                 0.036912 |            494337 |       0.806571 |               0.013340 |
| uci      | B4_static_temporal | S2      |      0.896099 |     0.028171 |           0.838338 |          0.092626 |       0.784148 |      0.034203 |                  0.052511 |                 0.028196 |            494337 |       0.894955 |              -0.001144 |
| uci      | B5_final           | S0      |      0.491300 |     0.047493 |           0.658816 |          0.106545 |       0.465315 |      0.020891 |                  0.194397 |                 0.066814 |            494337 |       0.491300 |               0.000000 |
| uci      | B5_final           | S1      |      0.806571 |     0.028279 |           0.799786 |          0.127395 |       0.701755 |      0.026240 |                  0.072523 |                 0.027565 |            494337 |       0.806571 |               0.000000 |
| uci      | B5_final           | S2      |      0.894955 |     0.023708 |           0.825089 |          0.105455 |       0.781092 |      0.040969 |                  0.045228 |                 0.021999 |            494337 |       0.894955 |               0.000000 |

### Required PR-AUC contributions

| domain   | comparison              | stage   |   delta_pr_auc |
|:---------|:------------------------|:--------|---------------:|
| uci      | final_minus_static      | S0      |       0.037510 |
| uci      | final_minus_static      | S1      |       0.352781 |
| uci      | final_minus_static      | S2      |       0.441165 |
| uci      | final_minus_static      | macro   |       0.277152 |
| uci      | final_minus_temporal    | S0      |       0.278057 |
| uci      | final_minus_temporal    | S1      |       0.033164 |
| uci      | final_minus_temporal    | S2      |      -0.009551 |
| uci      | final_minus_temporal    | macro   |       0.100557 |
| uci      | cnn_bilstm_minus_cnn    | S0      |      -0.001092 |
| uci      | cnn_bilstm_minus_cnn    | S1      |       0.002891 |
| uci      | cnn_bilstm_minus_cnn    | S2      |       0.000408 |
| uci      | cnn_bilstm_minus_cnn    | macro   |       0.000735 |
| uci      | cnn_bilstm_minus_bilstm | S0      |      -0.000680 |
| uci      | cnn_bilstm_minus_bilstm | S1      |       0.009030 |
| uci      | cnn_bilstm_minus_bilstm | S2      |       0.000378 |
| uci      | cnn_bilstm_minus_bilstm | macro   |       0.002909 |
| uci      | aggregate_gain          | S0      |      -0.007652 |
| uci      | aggregate_gain          | S1      |       0.013340 |
| uci      | aggregate_gain          | S2      |      -0.001144 |
| uci      | aggregate_gain          | macro   |       0.001515 |
| oulad    | final_minus_static      | 20pct   |       0.192524 |
| oulad    | final_minus_static      | 35pct   |       0.259831 |
| oulad    | final_minus_static      | 50pct   |       0.322410 |
| oulad    | final_minus_static      | 75pct   |       0.400758 |
| oulad    | final_minus_static      | macro   |       0.293881 |
| oulad    | final_minus_temporal    | 20pct   |       0.079711 |
| oulad    | final_minus_temporal    | 35pct   |       0.029419 |
| oulad    | final_minus_temporal    | 50pct   |       0.016503 |
| oulad    | final_minus_temporal    | 75pct   |       0.009431 |
| oulad    | final_minus_temporal    | macro   |       0.033766 |
| oulad    | cnn_bilstm_minus_cnn    | 20pct   |       0.003388 |
| oulad    | cnn_bilstm_minus_cnn    | 35pct   |       0.009773 |
| oulad    | cnn_bilstm_minus_cnn    | 50pct   |       0.010489 |
| oulad    | cnn_bilstm_minus_cnn    | 75pct   |       0.015578 |
| oulad    | cnn_bilstm_minus_cnn    | macro   |       0.009807 |
| oulad    | cnn_bilstm_minus_bilstm | 20pct   |       0.003260 |
| oulad    | cnn_bilstm_minus_bilstm | 35pct   |       0.000507 |
| oulad    | cnn_bilstm_minus_bilstm | 50pct   |       0.001306 |
| oulad    | cnn_bilstm_minus_bilstm | 75pct   |       0.001348 |
| oulad    | cnn_bilstm_minus_bilstm | macro   |       0.001605 |
| oulad    | aggregate_gain          | 20pct   |       0.010228 |
| oulad    | aggregate_gain          | 35pct   |       0.006415 |
| oulad    | aggregate_gain          | 50pct   |       0.005536 |
| oulad    | aggregate_gain          | 75pct   |       0.001637 |
| oulad    | aggregate_gain          | macro   |       0.005954 |

## Paired hierarchical bootstrap versus frozen parity winner

| domain   | stage   | baseline_family                |   mean_delta |   ci_lower |   ci_upper |   probability_delta_gt_zero |   bootstrap_replicates | method                                         |
|:---------|:--------|:-------------------------------|-------------:|-----------:|-----------:|----------------------------:|-----------------------:|:-----------------------------------------------|
| uci      | S0      | random_forest                  |    -0.002587 |  -0.030165 |   0.025718 |                    0.399000 |                   2000 | paired_hierarchical_seed_fold_record_bootstrap |
| uci      | S1      | random_forest                  |     0.015601 |  -0.009815 |   0.037529 |                    0.862000 |                   2000 | paired_hierarchical_seed_fold_record_bootstrap |
| uci      | S2      | random_forest                  |    -0.012611 |  -0.029732 |   0.004282 |                    0.071500 |                   2000 | paired_hierarchical_seed_fold_record_bootstrap |
| uci      | macro   | stagewise_phase7_parity_winner |     0.000134 |  -0.014321 |   0.013193 |                    0.462000 |                   2000 | paired_hierarchical_seed_fold_record_bootstrap |
| oulad    | 20pct   | catboost                       |    -0.001045 |  -0.004285 |   0.002058 |                    0.258500 |                   2000 | paired_hierarchical_seed_fold_record_bootstrap |
| oulad    | 35pct   | catboost                       |     0.000866 |  -0.000965 |   0.002736 |                    0.825500 |                   2000 | paired_hierarchical_seed_fold_record_bootstrap |
| oulad    | 50pct   | catboost                       |    -0.000048 |  -0.002743 |   0.002394 |                    0.506500 |                   2000 | paired_hierarchical_seed_fold_record_bootstrap |
| oulad    | 75pct   | catboost                       |    -0.002891 |  -0.004521 |  -0.001365 |                    0.000000 |                   2000 | paired_hierarchical_seed_fold_record_bootstrap |
| oulad    | macro   | stagewise_phase7_parity_winner |    -0.000780 |  -0.001958 |   0.000420 |                    0.104000 |                   2000 | paired_hierarchical_seed_fold_record_bootstrap |

Intervals containing zero are reported as no clear statistical superiority.

## Final Hybrid robustness

| domain   | stage   |   pr_auc_mean |   pr_auc_std |   pr_auc_min |   pr_auc_max |   roc_auc_mean |   roc_auc_std |   roc_auc_min |   roc_auc_max |   risk_recall_mean |   risk_recall_std |   risk_recall_min |   risk_recall_max |   risk_f1_mean |   risk_f1_std |   risk_f1_min |   risk_f1_max |   train_validation_gap_mean |   train_validation_gap_std |   train_validation_gap_min |   train_validation_gap_max |   selected_threshold_mean |   selected_threshold_std |   selected_threshold_min |   selected_threshold_max |   best_epoch_mean |   best_epoch_std |   best_epoch_min |   best_epoch_max |
|:---------|:--------|--------------:|-------------:|-------------:|-------------:|---------------:|--------------:|--------------:|--------------:|-------------------:|------------------:|------------------:|------------------:|---------------:|--------------:|--------------:|--------------:|----------------------------:|---------------------------:|---------------------------:|---------------------------:|--------------------------:|-------------------------:|-------------------------:|-------------------------:|------------------:|-----------------:|-----------------:|-----------------:|
| oulad    | 20pct   |      0.765710 |     0.006024 |     0.758659 |     0.773809 |       0.791628 |      0.005030 |      0.784650 |      0.797510 |           0.761731 |          0.034117 |          0.714964 |          0.817059 |       0.683136 |      0.006570 |      0.674660 |      0.692777 |                    0.028933 |                   0.009527 |                   0.017239 |                   0.040896 |                  0.508889 |                 0.043429 |                 0.430000 |                 0.570000 |         72.222222 |        10.256434 |        56.000000 |        87.000000 |
| oulad    | 35pct   |      0.808871 |     0.005103 |     0.802745 |     0.816105 |       0.832571 |      0.003174 |      0.828713 |      0.836600 |           0.749278 |          0.044030 |          0.678246 |          0.823907 |       0.702266 |      0.006219 |      0.697544 |      0.714343 |                    0.027146 |                   0.007882 |                   0.016930 |                   0.036011 |                  0.498889 |                 0.056887 |                 0.390000 |                 0.590000 |         72.222222 |        10.256434 |        56.000000 |        87.000000 |
| oulad    | 50pct   |      0.849811 |     0.008431 |     0.843525 |     0.861750 |       0.874374 |      0.005635 |      0.868493 |      0.882217 |           0.724572 |          0.037709 |          0.666509 |          0.783756 |       0.734626 |      0.013568 |      0.720799 |      0.753011 |                    0.021300 |                   0.013596 |                   0.002013 |                   0.032565 |                  0.483333 |                 0.060622 |                 0.360000 |                 0.560000 |         72.222222 |        10.256434 |        56.000000 |        87.000000 |
| oulad    | 75pct   |      0.888627 |     0.007488 |     0.880634 |     0.898789 |       0.912328 |      0.004798 |      0.906541 |      0.918552 |           0.727707 |          0.039639 |          0.669105 |          0.793750 |       0.780814 |      0.012162 |      0.766602 |      0.799880 |                    0.016826 |                   0.011455 |                   0.001142 |                   0.027335 |                  0.546667 |                 0.094604 |                 0.410000 |                 0.690000 |         72.222222 |        10.256434 |        56.000000 |        87.000000 |
| uci      | S0      |      0.491300 |     0.047493 |     0.430392 |     0.560291 |       0.758143 |      0.025764 |      0.726090 |      0.793348 |           0.658816 |          0.106545 |          0.525424 |          0.864407 |       0.465315 |      0.020891 |      0.437768 |      0.492063 |                    0.194397 |                   0.066814 |                   0.085821 |                   0.286240 |                  0.388889 |                 0.120669 |                 0.160000 |                 0.550000 |         13.000000 |         1.732051 |         9.000000 |        15.000000 |
| uci      | S1      |      0.806571 |     0.028279 |     0.772034 |     0.842833 |       0.940963 |      0.008623 |      0.928692 |      0.951815 |           0.799786 |          0.127395 |          0.639344 |          0.964912 |       0.701755 |      0.026240 |      0.666667 |      0.741259 |                    0.072523 |                   0.027565 |                   0.032376 |                   0.118943 |                  0.625556 |                 0.159226 |                 0.430000 |                 0.820000 |         13.000000 |         1.732051 |         9.000000 |        15.000000 |
| uci      | S2      |      0.894955 |     0.023708 |     0.871411 |     0.930026 |       0.966885 |      0.006848 |      0.957730 |      0.975093 |           0.825089 |          0.105455 |          0.622951 |          0.947368 |       0.781092 |      0.040969 |      0.723810 |      0.834783 |                    0.045228 |                   0.021999 |                   0.013668 |                   0.075454 |                  0.613333 |                 0.179234 |                 0.260000 |                 0.830000 |         13.000000 |         1.732051 |         9.000000 |        15.000000 |

## OULAD Fail versus Withdrawn

| domain   | stage   | analysis     | subgroup   |   prediction_instances |   risk_instances |   false_negatives |   recall |
|:---------|:--------|:-------------|:-----------|-----------------------:|-----------------:|------------------:|---------:|
| oulad    | 20pct   | risk_outcome | Fail       |                  14157 |            14157 |              3237 | 0.771350 |
| oulad    | 20pct   | risk_outcome | Withdrawn  |                   8547 |             8547 |              2173 | 0.745759 |
| oulad    | 35pct   | risk_outcome | Fail       |                  14160 |            14160 |              3407 | 0.759393 |
| oulad    | 35pct   | risk_outcome | Withdrawn  |                   6375 |             6375 |              1742 | 0.726745 |
| oulad    | 50pct   | risk_outcome | Fail       |                  14163 |            14163 |              3735 | 0.736285 |
| oulad    | 50pct   | risk_outcome | Withdrawn  |                   4356 |             4356 |              1380 | 0.683196 |
| oulad    | 75pct   | risk_outcome | Fail       |                  14163 |            14163 |              3769 | 0.733884 |
| oulad    | 75pct   | risk_outcome | Withdrawn  |                   1488 |             1488 |               498 | 0.665323 |

## UCI S0 static-versus-final error overlap

| domain   | stage   | analysis                      | subgroup                   |   prediction_instances |   risk_instances |   false_negatives |   recall |
|:---------|:--------|:------------------------------|:---------------------------|-----------------------:|-----------------:|------------------:|---------:|
| uci      | S0      | static_vs_final_error_overlap | all_risk                   |                    531 |              531 |               181 | 0.659134 |
| uci      | S0      | static_vs_final_error_overlap | false_negative_both        |                    121 |              121 |               121 | 0.000000 |
| uci      | S0      | static_vs_final_error_overlap | false_negative_final_only  |                     60 |               60 |                60 | 0.000000 |
| uci      | S0      | static_vs_final_error_overlap | false_negative_static_only |                     67 |               67 |                 0 | 1.000000 |
| uci      | S0      | static_vs_final_error_overlap | correct_both               |                    283 |              283 |                 0 | 1.000000 |

## Supported thesis claims

- One frozen unified A2 topology operates on both UCI and OULAD
- The final Hybrid is competitive with fixed strong tree baselines
- Phase7 data redesign materially improved controlled OULAD development performance
- CNN and BiLSTM representations are non-identical
- Training HPO improved the frozen A2 architecture on average
- AP fine-tuning did not robustly improve the final result

## Claims not supported

- Hybrid universally outperforms CatBoost
- Hybrid has statistically significant superiority at every OULAD cutoff
- AP fine-tuning is beneficial
- A bigger model is better
- Historical Phase5 outer results are fresh confirmatory evidence
- Aggregate branch adds PR-AUC

## Limitations

- This is repeated inner-development evidence, not a fresh untouched outer-test confirmation.
- OULAD cohort and risk prevalence change with cutoff.
- Remaining baseline margins and bootstrap uncertainty limit superiority claims.
- UCI S0 contains no grade observations; its ceiling is dominated by static information and sample variability.
