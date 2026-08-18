# OULAD SVM Runtime Protocol Amendment

The SVM family remains nonlinear RBF `sklearn.svm.SVC`. Internal `probability=True` calibration is superseded by `probability=False` and a one-variable Platt-style sigmoid fitted exclusively from pooled inner-OOF decision scores. Outer labels are used for neither calibration nor threshold selection. The archived partial checkpoints have `scientific_use=false` and are excluded from final evidence.

Folds, seeds, features, target, cohorts, stages, and the model-selection objective are unchanged.
