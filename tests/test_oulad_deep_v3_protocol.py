from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.studies.oulad_v3.data import BASE_CHANNELS,DYNAMIC_CHANNELS,aggregate_dynamic_channels,build_dynamic_representation,build_inner_manifest,load_v3_data,manifest_indices,semantic_hash
from src.studies.oulad_v3.models import TemporalPoolingEncoder,prepare_inputs,set_deterministic_seed

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=json.loads((ROOT/"configs/oulad_deep_v3_protocol.yaml").read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@pytest.fixture(scope="module")
def data(): return load_v3_data(ROOT/"data/processed/study_c_oulad",PROTOCOL)

def test_v2_evidence_is_immutable():
    source=PROTOCOL["source"]; root=ROOT/source["v2_artifact_root"]
    assert digest(ROOT/source["v2_protocol"])==source["v2_protocol_sha256"]
    assert digest(root/"oof_predictions.parquet")==source["v2_oof_sha256"]
    assert digest(root/"selected_configs.json")==source["v2_selected_configs_sha256"]

def test_protocol_frozen_before_results_and_exploratory_label():
    assert PROTOCOL["protocol_status"]=="frozen_before_v3_outer_results"
    assert PROTOCOL["study_label"]=="exploratory post-V2 temporal representation study"

def test_f2_cohort_and_global_grouping(data):
    assert PROTOCOL["data"]["forecast_id"]=="F2_MIDDLE"
    assert len(data.development_indices)==15378
    for fold in range(3):
        train,val=data.outer_indices(fold); assert not set(data.groups[train])&set(data.groups[val])

def test_inner_outer_disjointness(data):
    for fold in range(3):
        manifest=build_inner_manifest(data.v2,fold,3407+fold); _,outer=data.outer_indices(fold)
        for inner in range(2):
            train,val=manifest_indices(data.v2,manifest,inner)
            assert not set(data.groups[train])&set(data.groups[val]); assert not set(data.groups[outer])&set(data.groups[train]); assert not set(data.groups[outer])&set(data.groups[val])

def test_dynamic_contract_width_hash_and_finite(data):
    assert len(BASE_CHANNELS)==16 and len(DYNAMIC_CHANNELS)==31 and data.dynamic_sequence.shape[2]==47
    assert semantic_hash(list(data.dynamic_channel_order))==PROTOCOL["dynamic_features"]["combined_channel_hash"]
    assert np.isfinite(data.dynamic_sequence).all()

def test_week_t_does_not_read_week_t_plus_one(data):
    base=data.base.sequence[:8].copy(); mask=data.base.padding_mask[:8].copy(); first,_=build_dynamic_representation(base,mask)
    changed=base.copy(); changed[:,10:,:]+=1234; second,_=build_dynamic_representation(changed,mask)
    np.testing.assert_allclose(first[:,:10],second[:,:10])

def test_first_delta_zero_and_rolling_past_only(data):
    dynamic=data.dynamic_sequence; offset=len(BASE_CHANNELS)
    for name in [n for n in DYNAMIC_CHANNELS if n.startswith("delta_")]: assert np.allclose(dynamic[:,0,offset+DYNAMIC_CHANNELS.index(name)],0)
    base=data.base.sequence[:2].copy(); mask=data.base.padding_mask[:2]; a,_=build_dynamic_representation(base,mask); base[:,1:]+=100; b,_=build_dynamic_representation(base,mask)
    np.testing.assert_allclose(a[:,0],b[:,0])

def test_dynamic_aggregate_parity(data):
    aggregate,names=aggregate_dynamic_channels(data.dynamic_sequence[:20],data.base.valid_lengths[:20])
    assert aggregate.shape==(20,279) and len(names)==279 and np.isfinite(aggregate).all()
    assert data.matched_vector.shape[1]==440

def test_d0_a1_mld_information_parity_is_declared():
    registry=PROTOCOL["candidate_registry"]["mandatory"]
    assert registry["V3-A1"]["inputs"]==registry["V3-MLD"]["inputs"]==["matched_vector_440","static"]
    assert PROTOCOL["dynamic_features"]["dynamic_aggregate_count"]==279

def _pooling(pooling):
    return TemporalPoolingEncoder(47,{"conv_channels":24,"kernel_size":3,"lstm_hidden":32,"lstm_layers":1,"dropout":.2,"pooling":pooling,"pooling_projection":32})

@pytest.mark.parametrize("pooling",["last_mean_max","masked_attention"])
def test_pooling_shapes_and_finite(pooling):
    model=_pooling(pooling); x=torch.randn(4,7,47); lengths=torch.tensor([7,5,3,1]); mask=torch.arange(7)[None,:]<lengths[:,None]
    output,attention=model(x,lengths,mask.float(),True); assert output.shape==(4,32) and torch.isfinite(output).all()
    if attention is not None: assert float(attention.masked_select(~mask).max())==0.0

def test_train_only_preprocessing_and_information_shapes(data):
    train,val=data.outer_indices(0); train=train[:100]; val=val[:20]
    d0=prepare_inputs(data,train,val,"V3-D0"); a1=prepare_inputs(data,train,val,"V3-A1")
    assert d0.sequence.shape[2]==47 and d0.aggregate.shape[1]==161 and a1.aggregate.shape[1]==440
    assert d0.preprocessors.sequence_mean is not None and a1.preprocessors.aggregate is not None

def test_threshold_is_inner_only_and_future_inaccessible():
    assert PROTOCOL["metrics"]["threshold_fit_scope"]=="pooled_inner_oof_only"
    assert PROTOCOL["future_policy"]["available_during_selection"] is False
    source=(ROOT/"scripts/run_oulad_deep_v3.py").read_text(); assert "future_predictions.parquet" not in source

def test_three_seed_and_ensemble_contract():
    assert PROTOCOL["seeds"]==[42,2026,3407] and PROTOCOL["ensemble"]["members"]==[42,2026,3407]
    source=(ROOT/"scripts/run_oulad_deep_v3.py").read_text(); assert '.agg(probability=("probability","mean")' in source

def test_seed_initialization_is_deterministic():
    set_deterministic_seed(42); a=torch.rand(4); set_deterministic_seed(42); torch.testing.assert_close(a,torch.rand(4))

def test_no_forbidden_conditional_models_or_future_execution():
    assert PROTOCOL["future_policy"]["default_execution"]=="NOT_EXECUTED"
    assert {"gating","multi_scale_CNN","Transformer","GNN"}<=set(PROTOCOL["conditional"]["forbidden"])

def test_parameter_guardrail_is_enforced():
    source=(ROOT/"src/studies/oulad_v3/training.py").read_text(); assert "count>=300000" in source

def test_adaptive_envelope_freezes_scientific_choices():
    forbidden=set(PROTOCOL["search"]["adaptive_envelope"]["forbidden"])
    assert {"target","cohort","cutoff","group","outer_folds","primary_metric","seed_registry","comparators","future_policy"}<=forbidden
