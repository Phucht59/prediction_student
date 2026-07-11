# Report correction patch

| Report location | Find | Proposed replacement | Reason | Source |
| --- | --- | --- | --- | --- |
| Methods/loss | Weighted CrossEntropyLoss | CrossEntropyLoss without class weighting | final class_weight_mode is none | selected_config.json |
| Imbalance | SMOTE and ADASYN evaluated | Final selection evaluated SMOTE and class weighting; ADASYN is supplementary post-hoc development-pool ablation | scope accuracy | supplementary protocol |
| Dataset table | paid risk feature | `paid` exists in raw data but is excluded from automated recommendation policy to avoid advice based on paid tutoring | policy fairness | recommendation.py |
| Evaluation | locked test only used once | Locked test was excluded from Optuna/tuning/final configuration selection; baseline/ablation tables are post-hoc after freeze | accurate history | protocol manifest |
| Results | point estimate only | Add bootstrap CI table (2,000, seed 42) | 79-record uncertainty | bootstrap artifact |
| Baselines | HGB architecture comparison | Feature/tuning protocols differ; report as system comparison, not controlled architecture comparison | comparability | baseline results |
| Ablation | CNN essential | Fixed, parameter-unequal ablations cannot establish causal component importance | interpretation | ABLATION_INTERPRETATION_NOTE.md |
| Methods | long time series | A very short ordered sequence of two assessments | data scope | selected config |
| Recommender | intelligent recommendation model | Rule-based advisory system | policy is deterministic | recommendation.py |
| Dataset | Vietnamese university students | Portuguese secondary-school mathematics benchmark | domain limitation | dataset manifest |
| Reproducibility | immutable checkpoint replay | Deterministic rerun is evidenced; exact immutable checkpoint linkage is not fully frozen | provenance limitation | `artifacts/supplementary/runtime_artifact_provenance/runtime_artifact_provenance.json` |
| Database | production-ready inference schema | Current schema supports experiment/evaluation; production inference needs a future extension | true label timing | production extension doc |
