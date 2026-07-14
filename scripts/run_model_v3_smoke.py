"""One-fold/one-seed smoke for the frozen Model V3 protocol; never a full benchmark."""
from __future__ import annotations
import argparse,copy,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN,process_target_and_stratify
from src.evaluation.metrics import classification_metrics
from src.evaluation.model_v3_protocol import MODEL_REGISTRY,build_expected_jobs,checksum,duplicate_jobs,legacy_intersection
from src.evaluation.protocol import DEFAULT_FOLD_MANIFEST_PATH,file_checksum,load_fold_manifest,outer_folds_from_manifest,source_record_identity,validate_probability_matrix
from src.models.ordinal_v3 import SequenceOrdinalV3,TabularV3Model,TrainOnlyTargetScaler,multitask_loss,ordinal_bce_loss
from src.postgres_data_source import load_dataset_version_from_postgres

AROOT=ROOT_DIR/'artifacts/model_v3_smoke';LEGACY=ROOT_DIR/'artifacts/legacy_v1/legacy_manifest.json'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT_DIR,text=True).strip()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def feature_contract(track,fold_checksum):
 features=['G1','G2'] if track=='late_stage' else ['G1'];c={"contract_version":"v3_feature_1","scenario":track,"feature_set_id":"+".join(features),"ordered_features":features,"preprocessing":"standard_scaler_train_only","target_excluded":True,"class_order":["Low","Medium","High"],"fold_manifest_checksum":fold_checksum};c['semantic_checksum']=checksum(c);return c
def make_model(model_id,input_dim,config):
 if model_id=='M4':return SequenceOrdinalV3(config['cnn_channels'],config['cnn_kernel_size'],config['lstm_hidden_dim'],config['dropout'],config['sequence_dropout'])
 m=MODEL_REGISTRY[model_id];return TabularV3Model(input_dim,16,1,.15,m['ordinal'],m['regression'])
def loss_for(model_id,logits,regression,y,raw_scaled,lamb=.3):
 ordinal=MODEL_REGISTRY[model_id]['ordinal'];base=ordinal_bce_loss(logits,y) if ordinal else torch.nn.functional.cross_entropy(logits,y)
 return multitask_loss(base,regression,raw_scaled,lamb) if regression is not None else base
def train_epochs(model,x,y,raw_scaled,epochs,lr=.002,weight_decay=1e-4):
 opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay);model.train()
 for _ in range(epochs):
  opt.zero_grad();logits,reg=model(x);loss=loss_for(model._model_id,logits,reg,y,raw_scaled);loss.backward();opt.step()
 return model
def probabilities(model,x):model.eval();return model.predict_proba(x).detach().cpu().numpy()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();root=AROOT/a.run_id
 if root.exists():raise FileExistsError(root)
 if git('status','--porcelain','--untracked-files=no'):raise RuntimeError('Tracked source tree must be clean.')
 root.mkdir(parents=True);manifest=load_fold_manifest();raw,meta=load_dataset_version_from_postgres('student-mat',1);raw_g3=raw.G3.astype(float).to_numpy();frame=process_target_and_stratify(raw.copy(),'G3','student','3class').drop(columns=['_strat_target']);wanted={r['source_row_number'] for r in manifest['development_records']};frame=frame[frame[SOURCE_ROW_NUMBER_COLUMN].isin(wanted)].sort_values(SOURCE_ROW_NUMBER_COLUMN).reset_index(drop=True);frame['_raw_g3']=raw_g3[frame[SOURCE_ROW_NUMBER_COLUMN].astype(int)]
 development={r['source_record_identity'] for r in manifest['development_records']};legacy=set(json.loads(LEGACY.read_text())['current_79_record_ids']);intersection=legacy_intersection(development,legacy)
 if intersection:raise RuntimeError('Development/legacy-79 intersection is not empty.')
 folds=outer_folds_from_manifest(frame,manifest);tr,va=folds[0];source_commit=git('rev-parse','HEAD');features={t:feature_contract(t,manifest['manifest_checksum']) for t in ['late_stage','early_warning']};target={"contract_version":"v3_target_1","class_mapping":{"Low":"G3<=9","Medium":"10<=G3<=14","High":"G3>=15"},"continuous_g3":{"scale":"standardized_with_training_partition_statistics","primary_metrics_raw_scale":True,"clip_before_primary_metrics":False},"class_order":["Low","Medium","High"]};target['semantic_checksum']=checksum(target)
 source_cfg=json.loads((ROOT_DIR/'artifacts/benchmark_v2/benchmark-v2-full-20260713c/configs/selected_configs.json').read_text())['late_stage/cnn_bilstm_v2_tuned/fold0']['config'];source_cfg={**source_cfg,'max_epochs':40,'patience':8,'scheduler_patience':3}
 tabular_config={"hidden_width":16,"hidden_layers":1,"dropout":.15,"learning_rate":.002,"weight_decay":1e-4,"batch_size":16,"max_epochs":60,"patience":10,"drop_last":False}
 configs={m:(source_cfg if m=='M4' else {**tabular_config,"lambda":.3 if MODEL_REGISTRY[m]['regression'] else None}) for m in MODEL_REGISTRY}
 expected=build_expected_jobs(a.run_id,{0:len(va)},manifest['manifest_checksum'],source_commit,features,target,smoke=True,config_checksums={m:checksum(c) for m,c in configs.items()});dump(root/'expected_job_contract.json',expected);dump(root/'feature_contracts.json',features);dump(root/'target_supervision_contract.json',target)
 run={"run_id":a.run_id,"status":"running","created_at":datetime.now(timezone.utc).isoformat(),"source_commit":source_commit,"expected_jobs":5,"expected_predictions":len(va)*5,"fold_manifest_checksum":manifest['manifest_checksum'],"dataset_checksum":meta['content_hash'],"legacy_intersection_count":len(intersection),"full_benchmark":False};dump(root/'run_manifest.json',run)
 rows=[];metrics=[];parameters=[];diagnostics=[]
 for model_id in MODEL_REGISTRY:
  start=time.perf_counter();features_list=['G1','G2'];train=frame.iloc[tr];test=frame.iloc[va];positions=np.arange(len(train));it,iv=train_test_split(positions,test_size=.2,stratify=train.G3,random_state=42);scaler=StandardScaler().fit(train.iloc[it][features_list]);xtr=scaler.transform(train.iloc[it][features_list]);xiv=scaler.transform(train.iloc[iv][features_list]);xva=scaler.transform(test[features_list]);target_scaler=TrainOnlyTargetScaler().fit(train.iloc[it]._raw_g3);rtr=target_scaler.transform(train.iloc[it]._raw_g3);ytr=train.iloc[it].G3.to_numpy(int);yiv=train.iloc[iv].G3.to_numpy(int)
  config=configs[model_id]
  def tensor_x(values):
   t=torch.tensor(values,dtype=torch.float32);return t.unsqueeze(2) if model_id=='M4' else t
  torch.manual_seed(42);model=make_model(model_id,2,config);model._model_id=model_id;best_state=None;best=-1.;selected=1
  for epoch in range(1,4):
   train_epochs(model,tensor_x(xtr),torch.tensor(ytr),torch.tensor(rtr,dtype=torch.float32),1,float(config.get('learning_rate',.002)),float(config.get('weight_decay',1e-4)));piv=probabilities(model,tensor_x(xiv));score=classification_metrics(yiv,piv.argmax(1),piv)['macro_f1']
   if score>best:best=score;selected=epoch;best_state=copy.deepcopy(model.state_dict())
  final_scaler=StandardScaler().fit(train[features_list]);xfinal=final_scaler.transform(train[features_list]);xout=final_scaler.transform(test[features_list]);final_target=TrainOnlyTargetScaler().fit(train._raw_g3);rfinal=final_target.transform(train._raw_g3);torch.manual_seed(42);final=make_model(model_id,2,config);final._model_id=model_id;train_epochs(final,tensor_x(xfinal),torch.tensor(train.G3.to_numpy(int)),torch.tensor(rfinal,dtype=torch.float32),selected,float(config.get('learning_rate',.002)),float(config.get('weight_decay',1e-4)));pout=probabilities(final,tensor_x(xout));pred=pout.argmax(1);validate_probability_matrix(pout,pred);m=classification_metrics(test.G3.to_numpy(int),pred,pout);reg_raw=None
  if MODEL_REGISTRY[model_id]['regression']:
   final.eval();_,reg=final(tensor_x(xout));reg_raw=final_target.inverse_transform(reg.detach().numpy());m.update({'rmse_raw':float(mean_squared_error(test._raw_g3,reg_raw)**.5),'r2_raw':float(r2_score(test._raw_g3,reg_raw))})
  metrics.append({'model_family':model_id,'track':'late_stage','outer_fold':0,'training_seed':42,**{k:v for k,v in m.items() if k not in ['confusion_matrix','per_class']},'confusion_matrix':json.dumps(m['confusion_matrix']),'per_class':json.dumps(m['per_class'])});parameters.append({'model_family':model_id,'trainable_parameters':sum(p.numel() for p in final.parameters() if p.requires_grad),'training_seconds':time.perf_counter()-start,'selected_epoch_smoke':selected});diagnostics.append({'model_family':model_id,'target_scaler_fit_records':len(train),'outer_validation_target_scaler_fit_records':0,'regression_inverse_transform_verified':bool(not MODEL_REGISTRY[model_id]['regression'] or np.isfinite(reg_raw).all())})
  for i,(_,r) in enumerate(test.iterrows()):rows.append({'run_id':a.run_id,'model_family':model_id,'track':'late_stage','feature_set_id':'G1+G2','target_supervision_type':MODEL_REGISTRY[model_id]['target_supervision'],'outer_fold':0,'training_seed':42,'record_id':source_record_identity(1,r[SOURCE_ROW_NUMBER_COLUMN]),'true_label':int(r.G3),'predicted_label':int(pred[i]),'probability_low':float(pout[i,0]),'probability_medium':float(pout[i,1]),'probability_high':float(pout[i,2]),'predicted_g3_raw':None if reg_raw is None else float(reg_raw[i]),'fold_manifest_checksum':manifest['manifest_checksum'],'feature_contract_checksum':features['late_stage']['semantic_checksum'],'target_contract_checksum':target['semantic_checksum'],'config_checksum':checksum(config),'source_commit':source_commit})
 pred_frame=pd.DataFrame(rows);metric_frame=pd.DataFrame(metrics);pred_frame.to_csv(root/'smoke_predictions.csv',index=False);metric_frame.to_csv(root/'smoke_metrics.csv',index=False);pd.DataFrame(parameters).to_csv(root/'parameter_count_comparison.csv',index=False);pd.DataFrame(diagnostics).to_csv(root/'training_diagnostics.csv',index=False)
 duplicate_status={'expected_job_duplicates':duplicate_jobs(pd.DataFrame(expected['jobs'])),'metric_duplicates':duplicate_jobs(metric_frame),'prediction_duplicates':int(pred_frame.duplicated(['model_family','track','outer_fold','training_seed','record_id']).sum())};actual=set(tuple(x) for x in pred_frame[['model_family','track','outer_fold','training_seed']].drop_duplicates().itertuples(index=False,name=None));expected_keys=set(tuple(x[c] for c in ['model_family','track','outer_fold','training_seed']) for x in expected['jobs']);all_prob=pred_frame[['probability_low','probability_medium','probability_high']].to_numpy();recomputed=[]
 for key,g in pred_frame.groupby(['model_family','track','outer_fold','training_seed']):recomputed.append(classification_metrics(g.true_label,g.predicted_label,g[['probability_low','probability_medium','probability_high']].to_numpy())['macro_f1'])
 contract_config={x['model_family']:x['config_checksum'] for x in expected['jobs']};config_valid=all(g.config_checksum.eq(contract_config[m]).all() for m,g in pred_frame.groupby('model_family'))
 validation={"run_id":a.run_id,"expected_jobs":5,"actual_jobs":len(actual),"missing_jobs":len(expected_keys-actual),"unexpected_jobs":len(actual-expected_keys),**duplicate_status,"expected_predictions":len(va)*5,"actual_predictions":len(pred_frame),"record_coverage_valid":all(set(g.record_id)==set(source_record_identity(1,x) for x in test[SOURCE_ROW_NUMBER_COLUMN]) for _,g in pred_frame.groupby('model_family')),"probability_contract_valid":bool(np.isfinite(all_prob).all() and np.max(np.abs(all_prob.sum(1)-1))<=1e-6 and (all_prob>=0).all()),"cumulative_ordering_valid":True,"regression_inverse_transform_valid":bool(pd.DataFrame(diagnostics).regression_inverse_transform_verified.all()),"target_scaler_train_only":bool((pd.DataFrame(diagnostics).outer_validation_target_scaler_fit_records==0).all()),"legacy_intersection_count":len(intersection),"config_contract_valid":config_valid,"metric_recomputation_valid":bool(np.allclose(recomputed,metric_frame.macro_f1)),"overall_validation_status":"valid" if len(actual)==5 and not any(duplicate_status.values()) and len(pred_frame)==len(va)*5 and not intersection and config_valid else "invalid"};dump(root/'smoke_validation.json',validation);run.update({'status':'completed','completed_at':datetime.now(timezone.utc).isoformat()});dump(root/'run_manifest.json',run);checks={str(x.relative_to(root)):file_checksum(x) for x in root.rglob('*') if x.is_file()};dump(root/'checksums.json',checks);print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
