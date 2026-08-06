import pandas as pd
from src.recommend_hybrid.explainable_v2.risk_policy_selection import metrics

def test_high_uncertainty_is_not_high():
    frame=pd.DataFrame({"risk_probability":[.9,.1],"hybrid_uncertainty":[.9,.1],"seed_disagreement":[pd.NA,pd.NA],"outcome":[1,0]})
    result=metrics(frame,.2,.5,.4,.1)
    assert result["high_coverage"] == 0.0

def test_single_class_metrics_are_finite():
    frame=pd.DataFrame({"risk_probability":[.1,.2],"hybrid_uncertainty":[.1,.1],"seed_disagreement":[pd.NA,pd.NA],"outcome":[0,0]})
    result=metrics(frame,.2,.8,.4,.1)
    assert result["roc_auc"] is None
    assert result["brier"] >= 0
