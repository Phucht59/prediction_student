"""V3.2 dual-track readiness smoke. One outer fold/seed; never a full benchmark."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN,process_target_and_stratify
from src.evaluation.metrics import classification_metrics
from src.evaluation.model_v3_protocol import checksum,map_g3_to_class,regression_metric_summary
from src.evaluation.model_v3_2 import build_b0_selection_contract,build_shared_inner_split_manifest,inner_split_seed,round_half_up_median,validate_inner_split_manifest,validate_selected_trials
from src.evaluation.protocol import file_checksum,load_fold_manifest,outer_folds_from_manifest,source_record_identity,validate_probability_matrix
from src.models.ordinal_v3 import TrainOnlyTargetScaler
from src.postgres_data_source import load_dataset_version_from_postgres
from scripts.run_model_v3_smoke import fit_torch,predict_torch

ROOT=ROOT_DIR/'artifacts/model_v3_smoke';LEGACY=ROOT_DIR/'artifacts/legacy_v1/legacy_manifest.json'
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=str),encoding='utf-8')
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT_DIR,text=True).strip()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();root=ROOT/a.run_id
 if root.exists():raise FileExistsError(root)
 if git('status','--porcelain','--untracked-files=no'):raise RuntimeError('Tracked tree is dirty.')
 root.mkdir(parents=True);source=git('rev-parse','HEAD');fm=load_fold_manifest();raw,meta=load_dataset_version_from_postgres('student-mat',1);frame=process_target_and_stratify(raw.copy(),'G3','student','3class').drop(columns=['_strat_target']);wanted={x['source_row_number'] for x in fm['development_records']};frame=frame[frame[SOURCE_ROW_NUMBER_COLUMN].isin(wanted)].sort_values(SOURCE_ROW_NUMBER_COLUMN).reset_index(drop=True);frame['_raw_g3']=raw.loc[frame[SOURCE_ROW_NUMBER_COLUMN].astype(int),'G3'].to_numpy(float);frame['record_id']=[source_record_identity(1,x) for x in frame[SOURCE_ROW_NUMBER_COLUMN]]
 dev=set(frame.record_id);legacy=set(json.loads(LEGACY.read_text())['current_79_record_ids']);
 if dev&legacy:raise RuntimeError('legacy intersection')
 folds=outer_folds_from_manifest(frame,fm);tr,va=folds[0];train,valid=frame.iloc[tr].reset_index(drop=True),frame.iloc[va].reset_index(drop=True)
 inner=build_shared_inner_split_manifest({0:train[['record_id','G3']].rename(columns={'G3':'true_label'})},fm['manifest_checksum']);dump(root/'shared_inner_split_manifest.json',inner)
 # Split indices come only from the shared, model-independent seed.
 from sklearn.model_selection import StratifiedKFold
 splits=list(StratifiedKFold(3,shuffle=True,random_state=inner_split_seed(0,fm['manifest_checksum'])).split(train,train.G3))
 features={'late_stage':['G1','G2'],'early_warning':['G1']};common={'hidden_width':16,'hidden_layers':1,'dropout':.15,'learning_rate':.002,'weight_decay':1e-4,'batch_size':16,'max_epochs':60,'patience':10,'drop_last':False};source_cfg=json.loads((ROOT_DIR/'artifacts/benchmark_v2/benchmark-v2-full-20260713c/configs/selected_configs.json').read_text())['late_stage/cnn_bilstm_v2_tuned/fold0']['config'];m4={**source_cfg,'max_epochs':40,'patience':8,'scheduler_patience':3};m4_checksum=checksum(m4)
 neural_studies=[];b0_studies=[];trials=[];selected=[];epoch_rows=[]
 for track,cols in features.items():
  for fam in ['M0','M1','M2','M3']:
   study={'study_id':f'{a.run_id}:{fam}:{track}:outer0','model_family':fam,'track':track,'outer_fold':0,'trial_budget':1,'inner_split_manifest_checksum':inner['semantic_checksum']};neural_studies.append(study);config={**common,**({'lambda':.3} if fam in ['M2','M3'] else {})};scores=[]
   for inner_fold,(it,iv) in enumerate(splits):
    scaler=StandardScaler().fit(train.loc[it,cols]);ts=TrainOnlyTargetScaler().fit(train.loc[it,'_raw_g3']);model=fit_torch(fam,config,scaler.transform(train.loc[it,cols]),train.loc[it,'G3'].to_numpy(int),ts.transform(train.loc[it,'_raw_g3']),seed=42+inner_fold,epochs=2);prob,_,_=predict_torch(model,fam,scaler.transform(train.loc[iv,cols]));metric=classification_metrics(train.loc[iv,'G3'],prob.argmax(1),prob);trials.append({'study_id':study['study_id'],'trial_id':0,'inner_fold':inner_fold,'status':'completed','config_payload':json.dumps(config),'config_checksum':checksum(config),'macro_f1':metric['macro_f1'],'ordinal_mae':metric['ordinal_mae'],'best_epoch':2});scores.append(2)
   selected.append({'study_id':study['study_id'],'selected_trial_id':0,'config_payload':json.dumps(config),'config_checksum':checksum(config),'inner_split_manifest_checksum':inner['semantic_checksum']});epoch_rows.append({'study_id':study['study_id'],'best_epochs':scores,'refit_epochs':round_half_up_median(scores,60),'rounding_rule':'round_half_up_median'})
  study={'study_id':f'{a.run_id}:B0:{track}:outer0','model_family':'B0','track':track,'outer_fold':0,'trial_budget':4};b0_studies.append(study)
  for tid,alpha in enumerate([.01,.1,1.,10.]):
   for inner_fold,(it,iv) in enumerate(splits):
    scaler=StandardScaler().fit(train.loc[it,cols]);model=Ridge(alpha=alpha).fit(scaler.transform(train.loc[it,cols]),train.loc[it,'_raw_g3']);pred=model.predict(scaler.transform(train.loc[iv,cols]));m=regression_metric_summary(train.loc[iv,'_raw_g3'],pred);trials.append({'study_id':study['study_id'],'trial_id':tid,'inner_fold':inner_fold,'status':'completed','config_payload':json.dumps({'alpha':alpha}),'config_checksum':checksum({'alpha':alpha}),'rmse_raw':m['rmse_raw'],'mae_raw':m['mae_raw']})
  best=sorted([x for x in trials if x['study_id']==study['study_id'] and x['inner_fold']==0],key=lambda x:(np.mean([z['rmse_raw'] for z in trials if z['study_id']==study['study_id'] and z['trial_id']==x['trial_id']]),np.mean([z['mae_raw'] for z in trials if z['study_id']==study['study_id'] and z['trial_id']==x['trial_id']]),json.loads(x['config_payload'])['alpha']))[0];selected.append({'study_id':study['study_id'],'selected_trial_id':best['trial_id'],'config_payload':best['config_payload'],'config_checksum':best['config_checksum'],'inner_split_manifest_checksum':inner['semantic_checksum']})
 trial_df=pd.DataFrame(trials);selected_df=pd.DataFrame(selected);trial_df.to_csv(root/'selection_trials.csv',index=False);selected_df.to_csv(root/'selected_configs.csv',index=False);dump(root/'final_refit_epochs.json',epoch_rows)
 # Validate evidence before outer refits.
 sel_errors=validate_selected_trials(neural_studies,trial_df[trial_df.study_id.str.contains(':M')],selected_df[selected_df.study_id.str.contains(':M')],inner['semantic_checksum'],{})
 b0_errors=validate_selected_trials(b0_studies,trial_df[trial_df.study_id.str.contains(':B0')],selected_df[selected_df.study_id.str.contains(':B0')],inner['semantic_checksum'],{})
 rows=[];metrics=[];expected=0
 for track,cols in features.items():
  families=['M0','M1','M2','M3','B0']+(['M4'] if track=='late_stage' else [])
  for fam in families:
   pick=selected_df[selected_df.study_id==f'{a.run_id}:{fam}:{track}:outer0'] if fam!='M4' else pd.DataFrame()
   config=m4 if fam=='M4' else json.loads(pick.iloc[0].config_payload);epochs=round_half_up_median([2,2,2],40) if fam=='M4' else (2 if fam!='B0' else None);scaler=StandardScaler().fit(train[cols]);xtr=scaler.transform(train[cols]);xv=scaler.transform(valid[cols]);raw_pred=None
   if fam=='B0':model=Ridge(alpha=config['alpha']).fit(xtr,train._raw_g3);raw_pred=model.predict(xv);pred=map_g3_to_class(raw_pred);prob=np.eye(3)[pred];seed=0
   else:
    ts=TrainOnlyTargetScaler().fit(train._raw_g3);model=fit_torch(fam,config,xtr,train.G3.to_numpy(int),ts.transform(train._raw_g3),seed=42,epochs=epochs);prob,cum,scaled=predict_torch(model,fam,xv);pred=prob.argmax(1);raw_pred=None if scaled is None else ts.inverse_transform(scaled);seed=42
   validate_probability_matrix(prob,pred);metric=classification_metrics(valid.G3,pred,prob)
   if raw_pred is not None:metric.update(regression_metric_summary(valid._raw_g3,raw_pred))
   metrics.append({'model_family':fam,'track':track,'outer_fold':0,'training_seed':seed,**{k:v for k,v in metric.items() if k not in ['confusion_matrix','per_class']}})
   for i,r in valid.iterrows():rows.append({'model_family':fam,'track':track,'outer_fold':0,'training_seed':seed,'record_id':r.record_id,'true_label':int(r.G3),'raw_g3':float(r._raw_g3),'predicted_label':int(pred[i]),'probability_low':float(prob[i,0]),'probability_medium':float(prob[i,1]),'probability_high':float(prob[i,2]),'predicted_g3_raw':None if raw_pred is None else float(raw_pred[i]),'fixed_m4_config_checksum':m4_checksum if fam=='M4' else None})
   expected+=len(valid)
 pred_df=pd.DataFrame(rows);pred_df.to_csv(root/'dual_track_smoke_predictions.csv',index=False);pd.DataFrame(metrics).to_csv(root/'dual_track_smoke_metrics.csv',index=False)
 validation={'run_id':a.run_id,'expected_jobs':11,'actual_jobs':len(metrics),'expected_predictions':expected,'actual_predictions':len(pred_df),'shared_inner_split_checksum':inner['semantic_checksum'],'selection_validation':sel_errors,'b0_selection_validation':b0_errors,'record_coverage_valid':all(len(g)==len(valid) for _,g in pred_df.groupby(['model_family','track'])),'probability_contract_valid':bool(np.max(np.abs(pred_df[['probability_low','probability_medium','probability_high']].to_numpy().sum(1)-1))<=1e-6),'legacy_intersection_count':0,'m4_fixed_config_bound':bool(pred_df[pred_df.model_family=='M4'].fixed_m4_config_checksum.eq(m4_checksum).all()),'overall_validation_status':'valid'}
 if validation['actual_jobs']!=11 or validation['actual_predictions']!=expected or any(sel_errors.values()) or any(b0_errors.values()):validation['overall_validation_status']='invalid'
 dump(root/'dual_track_smoke_validation.json',validation);dump(root/'run_manifest.json',{'run_id':a.run_id,'source_commit':source,'status':'completed','full_benchmark':False,'scientific_eligibility':'smoke_only','dataset_checksum':meta['content_hash']});checks={str(x.relative_to(root)):file_checksum(x) for x in root.rglob('*') if x.is_file()};dump(root/'checksums.json',checks);print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
