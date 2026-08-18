# Canonical V3 validation

- status: **PASS**
- primary_model_count: **8**
- all_models_full_metrics: **True**
- uci_stages_complete: **True**
- oulad_stages_complete: **True**
- uci_fold_hashes: **{'student_mat': {'decision_tree': '9bcd2ca9a1d7c6a495a19d40049f874b993bcf192a9d8cce64160cd817715738', 'hist_gradient_boosting': '9bcd2ca9a1d7c6a495a19d40049f874b993bcf192a9d8cce64160cd817715738', 'hybrid': '9bcd2ca9a1d7c6a495a19d40049f874b993bcf192a9d8cce64160cd817715738', 'logistic_regression': '9bcd2ca9a1d7c6a495a19d40049f874b993bcf192a9d8cce64160cd817715738', 'mlp': '9bcd2ca9a1d7c6a495a19d40049f874b993bcf192a9d8cce64160cd817715738', 'random_forest': '9bcd2ca9a1d7c6a495a19d40049f874b993bcf192a9d8cce64160cd817715738', 'svm': '9bcd2ca9a1d7c6a495a19d40049f874b993bcf192a9d8cce64160cd817715738', 'xgboost': '9bcd2ca9a1d7c6a495a19d40049f874b993bcf192a9d8cce64160cd817715738'}, 'student_por': {'decision_tree': '4dfd81bea0a4752999c025e58dabd019b97e227f61f5dd8bbb53e63235d505ef', 'hist_gradient_boosting': '4dfd81bea0a4752999c025e58dabd019b97e227f61f5dd8bbb53e63235d505ef', 'hybrid': '4dfd81bea0a4752999c025e58dabd019b97e227f61f5dd8bbb53e63235d505ef', 'logistic_regression': '4dfd81bea0a4752999c025e58dabd019b97e227f61f5dd8bbb53e63235d505ef', 'mlp': '4dfd81bea0a4752999c025e58dabd019b97e227f61f5dd8bbb53e63235d505ef', 'random_forest': '4dfd81bea0a4752999c025e58dabd019b97e227f61f5dd8bbb53e63235d505ef', 'svm': '4dfd81bea0a4752999c025e58dabd019b97e227f61f5dd8bbb53e63235d505ef', 'xgboost': '4dfd81bea0a4752999c025e58dabd019b97e227f61f5dd8bbb53e63235d505ef'}}**
- oulad_final_fold_hashes: **{'decision_tree': '30d54fccc4f761e2039f5a535aaea93074b706de7190f705723761161ac8e52b', 'hist_gradient_boosting': '30d54fccc4f761e2039f5a535aaea93074b706de7190f705723761161ac8e52b', 'hybrid': '30d54fccc4f761e2039f5a535aaea93074b706de7190f705723761161ac8e52b', 'logistic_regression': '30d54fccc4f761e2039f5a535aaea93074b706de7190f705723761161ac8e52b', 'mlp': '30d54fccc4f761e2039f5a535aaea93074b706de7190f705723761161ac8e52b', 'random_forest': '30d54fccc4f761e2039f5a535aaea93074b706de7190f705723761161ac8e52b', 'svm': '30d54fccc4f761e2039f5a535aaea93074b706de7190f705723761161ac8e52b', 'xgboost': '30d54fccc4f761e2039f5a535aaea93074b706de7190f705723761161ac8e52b'}**
- unique_outer_fold_hash_per_dataset: **{'student_mat': 1, 'student_por': 1, 'oulad': 1}**
- ml_hybrid_information_authority_identical: **True**
- architecture_counts: **{'uci': 1, 'oulad': 1}**
- outer_labels_used_for_selection: **False**
- post_result_tuning: **False**

## Post-compute verification

- Full pytest suite: **206 passed, 23 skipped**
- Ruff: **PASS**
- compileall: **PASS**
- Final comparator validator: **PASS**
- Release verifier: **PASS**
- Frozen historical evidence checksums: **PASS**
- Headline metric replay from OOF predictions: **PASS**
- Required metrics finite for every primary model/task: **PASS**
