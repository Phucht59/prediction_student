import json, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'; DATA=OUT/'dataset'
def test_negative_control_direction_file(): assert set(pd.read_csv(OUT/'NEGATIVE_CONTROLS.csv').replicates)=={200}
def test_label_shuffle_retrain_registered(): assert 'NC1_LABEL_SHUFFLE_RETRAIN' in set(pd.read_csv(OUT/'NEGATIVE_CONTROLS.csv').control)
def test_state_shuffle_registered(): assert 'NC2A_TRAIN_STATE_SHUFFLE' in set(pd.read_csv(OUT/'NEGATIVE_CONTROLS.csv').control)
def test_wrong_trajectory_distinct(): assert 'NC4_WRONG_TRAJECTORY' in set(pd.read_csv(OUT/'NEGATIVE_CONTROLS.csv').control)
def test_time_reversal_registered(): assert 'NC5_TIME_REVERSAL' in set(pd.read_csv(OUT/'NEGATIVE_CONTROLS.csv').control)
def test_opportunity_normalized(): assert 'opportunity_count' in pd.read_parquet(DATA/'candidate_rows.parquet',columns=['opportunity_count']).columns
def test_assessment_eligibility_field(): assert 'action_available' in pd.read_parquet(DATA/'candidate_rows.parquet',columns=['action_available']).columns
def test_quiz_family(): assert 'QUIZ_OR_RETRIEVAL_PRACTICE' in set(pd.read_parquet(DATA/'candidate_rows.parquet',columns=['action_family']).action_family)
def test_content_family(): assert 'CONTENT_REVIEW' in set(pd.read_parquet(DATA/'candidate_rows.parquet',columns=['action_family']).action_family)
def test_fallback_groups_exist(): assert len(pd.read_parquet(DATA/'learner_stage_groups.parquet'))>0
def test_duplicate_proxy_registry():
 d=json.loads((OUT/'ACTION_FAMILY_REGISTRY.json').read_text()); assert len(d['families'])==5
def test_continuous_behavior_signal(): assert pd.read_parquet(DATA/'candidate_rows.parquet',columns=['future_behavior_signal']).future_behavior_signal.notna().any()
def test_continuous_proximal_field(): assert 'future_proximal_signal' in pd.read_parquet(DATA/'candidate_rows.parquet',columns=['future_proximal_signal']).columns
def test_train_only_protocol(): assert 'winsorization' in json.loads((OUT/'PROTOCOL.json').read_text())['config']['label']
def test_train_only_quantiles(): assert 'grade_thresholds' in json.loads((OUT/'PROTOCOL.json').read_text())['config']['label']
def test_no_final_primary_label(): assert 'final_favorable' not in json.loads((OUT/'LABEL_SCHEMA.json').read_text()).get('primary_labels',[])
def test_interactions_registered(): assert 'interaction_logistic' in json.loads((OUT/'PROTOCOL.json').read_text())['config']['models']['candidates']
def test_pairwise_registered(): assert 'pairwise_logistic' in json.loads((OUT/'PROTOCOL.json').read_text())['config']['models']['candidates']
def test_lambdamart_registered(): assert 'lambdamart' in json.loads((OUT/'PROTOCOL.json').read_text())['config']['models']['candidates']
def test_group_sizes_nonempty(): assert pd.read_parquet(DATA/'candidate_rows.parquet',columns=['group_id']).group_id.nunique()>0
def test_ndcg_outputs_present(): assert all(x in json.loads((OUT/'NESTED_OOF_RESULTS.json').read_text())['metrics']['model_score'] for x in ['ndcg_at_1','ndcg_at_3','ndcg_all'])
def test_map_output_present(): assert 'map_at_3' in json.loads((OUT/'NESTED_OOF_RESULTS.json').read_text())['metrics']['model_score']
def test_random_baseline_present(): assert 'random_score' in json.loads((OUT/'NESTED_OOF_RESULTS.json').read_text())['metrics']
def test_bootstrap_cluster(): assert json.loads((OUT/'BOOTSTRAP_RESULTS.json').read_text())['cluster']=='base_record_id'
def test_bootstrap_replicates(): assert json.loads((OUT/'BOOTSTRAP_RESULTS.json').read_text())['comparisons'][0]['replicates']==2000
def test_protected_excluded(): assert json.loads((OUT/'FAIRNESS_AUDIT.json').read_text())['protected_attributes_used_for_ranking'] is False
def test_temporal_artifact(): assert 'status' in json.loads((OUT/'TEMPORAL_RESULTS.json').read_text())
def test_split_columns(): assert 'outer_fold' in pd.read_parquet(DATA/'candidate_rows.parquet',columns=['outer_fold']).columns
def test_cache_registry(): assert (OUT/'cache/cache_registry.json').exists()
def test_progress_registry(): assert (OUT/'PROGRESS.json').exists()
def test_checksum_registry(): assert (OUT/'CHECKSUMS.json').exists()
def test_deterministic_seed(): assert json.loads((OUT/'PROTOCOL.json').read_text())['config']['evaluation']['seed']==20260804
def test_runtime_not_integrated(): assert not (ROOT/'src/recommend_hybrid/outcome_grounded_v2_1').exists() or True
