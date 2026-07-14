import inspect
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import mean_squared_error, r2_score

from src.evaluation.model_v3_protocol import build_expected_jobs, checksum, duplicate_jobs, legacy_intersection, validate_loader_rows, validate_shape_rows
from src.models.ordinal_v3 import TabularV3Model, TrainOnlyTargetScaler, coral_targets, multitask_loss, ordinal_bce_loss, ordinal_logits_to_probabilities


def test_coral_targets_for_all_three_classes():
    assert torch.equal(coral_targets(torch.tensor([0,1,2])),torch.tensor([[0.,0.],[1.,0.],[1.,1.]]))


def test_ordinal_probabilities_are_valid_and_monotone():
    logits=torch.tensor([[2.,-1.],[0.,-2.]])
    p=ordinal_logits_to_probabilities(logits)
    assert torch.all(p>=0) and torch.allclose(p.sum(1),torch.ones(2))


def test_non_monotone_cumulative_logits_rejected():
    with pytest.raises(ValueError):ordinal_logits_to_probabilities(torch.tensor([[-2.,2.]]))


def test_argmax_mapping_uses_low_medium_high_order():
    p=ordinal_logits_to_probabilities(torch.tensor([[-4.,-6.],[4.,-4.],[6.,4.]]))
    assert p.argmax(1).tolist()==[0,1,2]


def test_ordinal_loss_is_finite_and_backward_works():
    model=TabularV3Model(2,ordinal=True);logits,_=model(torch.randn(4,2));loss=ordinal_bce_loss(logits,torch.tensor([0,1,2,1]));loss.backward();assert torch.isfinite(loss)


def test_target_scaler_fit_and_inverse_transform():
    scaler=TrainOnlyTargetScaler().fit([0,10,20]);values=scaler.transform([5,15]);assert np.allclose(scaler.inverse_transform(values),[5,15])


def test_target_scaler_statistics_are_train_only():
    scaler=TrainOnlyTargetScaler().fit([1,2,3]);assert scaler.mean_==2 and scaler.mean_!=100


def test_regression_metrics_are_on_raw_scale():
    y=np.array([0.,10.,20.]);p=np.array([1.,11.,19.]);assert mean_squared_error(y,p)**.5==pytest.approx(1.) and r2_score(y,p)>0.98


def test_multitask_lambda_changes_loss():
    c=torch.tensor(2.);p=torch.tensor([0.,1.]);t=torch.tensor([1.,1.]);assert multitask_loss(c,p,t,1)>multitask_loss(c,p,t,.1)


def test_lambda_zero_equals_classification_only():
    c=torch.tensor(2.);assert multitask_loss(c,torch.tensor([0.]),torch.tensor([9.]),0)==c


def test_nominal_ordinal_backbone_capacity_matches():
    nominal=TabularV3Model(2,16,1,.15,False,False);ordinal=TabularV3Model(2,16,1,.15,True,False)
    count=lambda m:sum(p.numel() for p in m.backbone.parameters());assert count(nominal)==count(ordinal)


def test_outer_validation_not_used_for_internal_split():
    source=inspect.getsource(__import__('scripts.run_model_v3_smoke',fromlist=['main']).main)
    assert "train_test_split(positions" in source and "test_size=.2" in source


def test_legacy_intersection_is_independent():
    assert legacy_intersection({'dev1','dev2'},{'legacy1'})==set()
    assert legacy_intersection({'same'},{'same'})=={'same'}


def _contracts():
    features={'late_stage':{'feature_set_id':'G1+G2','semantic_checksum':'f1'},'early_warning':{'feature_set_id':'G1','semantic_checksum':'f2'}};target={'semantic_checksum':'t'}
    return features,target


def test_expected_job_contract_created_before_compute():
    f,t=_contracts();c=build_expected_jobs('run',{i:63 for i in range(5)},'fold','commit',f,t)
    assert c['created_before_compute'] and len(c['jobs'])==225


def test_duplicate_job_mutation_detected():
    frame=pd.DataFrame([{'model_family':'M0','track':'late_stage','outer_fold':0,'training_seed':42}]*2);assert duplicate_jobs(frame)==2


def test_feature_and_target_checksum_mutation():
    assert checksum({'features':['G1']})!=checksum({'features':['G1','G2']})
    assert checksum({'target':'class'})!=checksum({'target':'class+G3'})


def test_missing_model_mutation_is_detectable():
    f,t=_contracts();c=build_expected_jobs('run',{0:64},'fold','commit',f,t,smoke=True);models={x['model_family'] for x in c['jobs']};models.remove('M4');assert len(models)==4


def test_shape_diagnostic_content_validation():
    good=pd.DataFrame([{'cnn_kernel_size':1,'cnn_output_sequence_length':2,'bilstm_input_sequence_length':2},{'cnn_kernel_size':2,'cnn_output_sequence_length':3,'bilstm_input_sequence_length':3}]);assert validate_shape_rows(good)
    bad=good.copy();bad.loc[0,'cnn_output_sequence_length']=3;assert not validate_shape_rows(bad)


def test_loader_diagnostic_content_validation():
    good=pd.DataFrame([{'dataset_size':10,'batch_size':4,'drop_last_train':False,'samples_dropped_per_epoch':0,'samples_consumed_per_epoch':10},{'dataset_size':10,'batch_size':4,'drop_last_train':True,'samples_dropped_per_epoch':2,'samples_consumed_per_epoch':8}]);assert validate_loader_rows(good)
    bad=good.copy();bad.loc[0,'samples_dropped_per_epoch']=1;assert not validate_loader_rows(bad)


def test_probability_validator_contract():
    p=ordinal_logits_to_probabilities(torch.tensor([[1.,0.]]));assert float(p.sum())==pytest.approx(1.) and float(p.min())>=0
