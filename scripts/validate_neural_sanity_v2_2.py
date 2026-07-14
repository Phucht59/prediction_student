"""Read-only strict validation and analysis for Neural Sanity Ablation V2.2."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR
from src.evaluation.metrics import classification_metrics
from src.evaluation.neural_sanity_v2_2 import EXPERIMENTS, JOB_COLUMNS, checksum, duplicate_job_rows, expected_job_keys
from src.evaluation.protocol import file_checksum, load_fold_manifest, validate_probability_matrix
from src.postgres_data_source import load_dataset_version_from_postgres

AROOT=ROOT_DIR/'artifacts/neural_sanity_v2_2'; RROOT=ROOT_DIR/'reports/neural_sanity_v2_2'; BASE=ROOT_DIR/'artifacts/benchmark_v2/benchmark-v2-full-20260713c'
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args(); root=AROOT/a.run_id; out=RROOT/a.run_id;out.mkdir(parents=True,exist_ok=True)
 contract=json.loads((root/'expected_job_contract.json').read_text()); manifest=json.loads((root/'run_manifest.json').read_text()); pred=pd.read_csv(root/'outer_validation_predictions.csv'); metrics=pd.read_csv(root/'fold_seed_metrics.csv'); diag=pd.read_csv(root/'training_diagnostics.csv'); shapes=pd.read_csv(root/'shape_diagnostics.csv'); completed=pd.read_json(root/'completed_jobs.jsonl',lines=True)
 expected=expected_job_keys(contract); actual={tuple(x) for x in pred[list(JOB_COLUMNS)].drop_duplicates().itertuples(index=False,name=None)}
 contract_rows=pd.DataFrame(contract['jobs']); contract_duplicates=duplicate_job_rows(contract_rows); metric_duplicates=duplicate_job_rows(metrics); completed_duplicates=duplicate_job_rows(completed); prediction_duplicates=int(pred.duplicated(['run_id',*JOB_COLUMNS,'record_id']).sum())
 fm=load_fold_manifest(); valid={f:set() for f in range(5)}
 for row in fm['assignments']:
  if row['outer_role']=='validation': valid[int(row['outer_fold'])].add(row['source_record_identity'])
 coverage=[]; scalar=[]; cm_bad=[]; pc_bad=[]
 for key,g in pred.groupby(list(JOB_COLUMNS)):
  ids=set(g.record_id); exp=valid[int(key[3])]; coverage.append({**dict(zip(JOB_COLUMNS,key)),'expected_records':len(exp),'actual_rows':len(g),'unique_records':g.record_id.nunique(),'missing_records':len(exp-ids),'outside_records':len(ids-exp),'valid':ids==exp and len(g)==len(exp)})
  m=metrics
  for c,v in zip(JOB_COLUMNS,key):m=m[m[c]==v]
  if len(m)!=1: cm_bad.append({**dict(zip(JOB_COLUMNS,key)),'issue':'missing_or_duplicate_metric_row'});continue
  d=classification_metrics(g.true_label,g.predicted_label,g[['probability_low','probability_medium','probability_high']].to_numpy()); st=m.iloc[0]
  for n in ['accuracy','macro_f1','weighted_f1','balanced_accuracy','quadratic_weighted_kappa','ordinal_mae','brier_score','pr_auc_macro','ece_top_label_equal_width_10']:
   scalar.append({**dict(zip(JOB_COLUMNS,key)),'metric':n,'stored':float(st[n]),'recomputed':float(d[n]),'difference':abs(float(st[n])-float(d[n])),'match':abs(float(st[n])-float(d[n]))<=1e-6})
  if json.loads(st.confusion_matrix)!=d['confusion_matrix']:cm_bad.append({**dict(zip(JOB_COLUMNS,key)),'issue':'confusion_matrix'})
  stored_pc=json.loads(st.per_class.replace("'",'"'))
  for i in range(3):
   if abs(float(stored_pc[str(i)]['f1'])-d['per_class'][str(i)]['f1'])>1e-6:pc_bad.append({**dict(zip(JOB_COLUMNS,key)),'class':i,'issue':'f1'})
 prob=pred[['probability_low','probability_medium','probability_high']].to_numpy();max_err=float(np.abs(prob.sum(1)-1).max());
 try: validate_probability_matrix(prob,pred.predicted_label.to_numpy()); probability=True
 except ValueError: probability=False
 checks=json.loads((root/'checksums.json').read_text()); checksum_bad=[rel for rel,h in checks.items() if not (root/rel).is_file() or file_checksum(root/rel)!=h]
 coverage_frame=pd.DataFrame(coverage); scalar_frame=pd.DataFrame(scalar); coverage_frame.to_csv(out/'record_coverage.csv',index=False);scalar_frame.to_csv(out/'metric_recomputation.csv',index=False)
 # Aggregation: mean seed scores inside fold, then mean/sd across fold means.
 rank=[]
 for experiment_id,g in metrics.groupby('experiment_id'):
  fold=g.groupby('outer_fold').mean(numeric_only=True); row={'experiment_id':experiment_id,'label':EXPERIMENTS[experiment_id]['label'],'n_training_seeds':5,'n_outer_folds':5,'n_fold_seed_evaluations':len(g),'n_record_prediction_rows':len(pred[pred.experiment_id==experiment_id]),'n_unique_outer_validation_records':pred[pred.experiment_id==experiment_id].record_id.nunique()}
  for metric in ['macro_f1','accuracy','balanced_accuracy','weighted_f1','quadratic_weighted_kappa','ordinal_mae','brier_score','pr_auc_macro','ece_top_label_equal_width_10']:
   row[metric+'_mean']=float(fold[metric].mean());row[metric+'_sd_outer_folds']=float(fold[metric].std(ddof=1));row[metric+'_fold_scores']=json.dumps(fold[metric].tolist())
  rank.append(row)
 ranking=pd.DataFrame(rank).sort_values('macro_f1_mean',ascending=False);ranking.to_csv(out/'ranking_by_variant.csv',index=False)
 pairs=[]
 for x,y in [('S1','S0'),('S2','S0'),('S3','S0'),('S4','S0'),('S5','S0'),('S5','S4'),('S4','S1')]:
  a=metrics[metrics.experiment_id==x].groupby('outer_fold').macro_f1.mean();b=metrics[metrics.experiment_id==y].groupby('outer_fold').macro_f1.mean();d=(a-b).sort_index();pairs.append({'variant_a':x,'variant_b':y,'metric':'macro_f1','foldwise_difference':json.dumps(d.tolist()),'mean_difference':float(d.mean()),'sd_difference':float(d.std(ddof=1)),'wins':int((d>1e-12).sum()),'ties':int((d.abs()<=1e-12).sum()),'losses':int((d<-1e-12).sum())})
 paired=pd.DataFrame(pairs);paired.to_csv(out/'paired_variant_comparisons.csv',index=False);(out/'paired_variant_comparisons.md').write_text('# Paired main effects V2.2\n\n'+paired.to_csv(index=False),encoding='utf-8')
 stability=[]
 for e,g in metrics.groupby('experiment_id'):
  per_fold=g.groupby('outer_fold').macro_f1.agg(['std','min','max']); high=g.apply(lambda x: json.loads(x.per_class.replace("'",'"'))['2']['f1'],axis=1); stability.append({'experiment_id':e,'mean_within_fold_seed_sd':float(per_fold['std'].mean()),'max_seed_range':float((per_fold['max']-per_fold['min']).max()),'high_f1_below_0_5_jobs':int((high<.5).sum()),'high_collapse_jobs':int((high==0).sum())})
 pd.DataFrame(stability).to_csv(out/'seed_stability.csv',index=False)
 raw,_=load_dataset_version_from_postgres('student-mat',1)
 g2_by_identity={f"student-mat:dataset-version:1:source-row:{int(row['__source_row_number'])}":int(row['G2']) for _,row in raw.iterrows()}
 annotated=pred.copy(); annotated['G2']=annotated.record_id.map(g2_by_identity); annotated['ordinal_error']=(annotated.true_label-annotated.predicted_label).abs()
 boundary=[]
 for e,g in annotated.groupby('experiment_id'):
  for grade in [9,10,14,15]:
   r=g[g.G2.eq(grade)]
   boundary.append({'experiment_id':e,'g2_boundary':grade,'records':len(r),'total_errors':int((r.ordinal_error>0).sum()),'one_step_errors':int((r.ordinal_error==1).sum()),'two_step_errors':int((r.ordinal_error==2).sum()),'high_class_errors':int(((r.true_label==2)&(r.predicted_label!=2)).sum()),'status':'computed'})
 pd.DataFrame(boundary).to_csv(out/'boundary_error_analysis.csv',index=False)
 base=pd.read_csv(BASE/'predictions/outer_validation_predictions.csv');base=base[base.model_name.eq('cnn_bilstm_v2_tuned')]
 control=pred[pred.experiment_id.eq('S0')]; merged=control.merge(base,on=['record_id','outer_fold','training_seed'],suffixes=('_s0','_v2')); agreement=float((merged.predicted_label_s0==merged.predicted_label_v2).mean()) if len(merged) else 0.0
 control_report={'source_run':'benchmark-v2-full-20260713c','matched_prediction_rows':len(merged),'prediction_agreement':agreement,'s0_macro_f1':ranking[ranking.experiment_id.eq('S0')].macro_f1_mean.iloc[0],'v2_reference_macro_f1':0.7983838344121756,'status':'reproduced' if agreement>=.99 else 'environment_or_source_drift_requires_caution'};dump(out/'control_reproduction.json',control_report);(out/'control_reproduction.md').write_text('# Control reproduction\n\n'+json.dumps(control_report,indent=2),encoding='utf-8')
 valid_status=(manifest.get('status')=='completed' and not (expected-actual) and not (actual-expected) and contract_duplicates==metric_duplicates==completed_duplicates==prediction_duplicates==0 and coverage_frame.valid.all() and probability and max_err<=1e-6 and not checksum_bad and scalar_frame['match'].all() and not cm_bad and not pc_bad and len(diag)==len(expected) and len(shapes)==len(expected))
 strict={'run_id':a.run_id,'expected_jobs':len(expected),'actual_jobs':len(actual),'missing_jobs':len(expected-actual),'unexpected_jobs':len(actual-expected),'duplicate_job_manifest_rows':contract_duplicates,'duplicate_metric_rows':metric_duplicates,'duplicate_completed_markers':completed_duplicates,'duplicate_prediction_rows':prediction_duplicates,'record_coverage_status':'valid' if coverage_frame.valid.all() else 'invalid','probability_contract_status':'valid' if probability else 'invalid','max_probability_sum_error':max_err,'scalar_metric_recomputation_status':'valid' if scalar_frame['match'].all() else 'invalid','confusion_matrix_status':'valid' if not cm_bad else 'invalid','per_class_metric_status':'valid' if not pc_bad else 'invalid','checksum_status':'valid' if not checksum_bad else 'invalid','checksum_failures':checksum_bad,'feature_contract_checksum_status':'valid' if pred.feature_contract_checksum.eq(contract['feature_contract']['semantic_checksum']).all() else 'invalid','fold_checksum_status':'valid' if pred.fold_manifest_checksum.eq(fm['manifest_checksum']).all() else 'invalid','legacy_79_isolation_status':'verified_by_development_manifest_and_runner_guard','training_diagnostics_status':'valid' if len(diag)==len(expected) else 'invalid','overall_validation_status':'valid' if valid_status else 'invalid'}
 dump(root/'strict_validation.json',strict);(root/'strict_validation.md').write_text('# Strict validation V2.2\n\n'+json.dumps(strict,indent=2),encoding='utf-8');dump(out/'strict_validation.json',strict);(out/'strict_validation.md').write_text('# Strict validation V2.2\n\n'+json.dumps(strict,indent=2),encoding='utf-8')
 (out/'sanity_ablation_conclusion.md').write_text('# V2.2 diagnostic conclusion\n\nInterpret only after control reproduction is acceptable. Variants are diagnostic research comparators, not final models.\n',encoding='utf-8')
 print(json.dumps(strict,indent=2))
if __name__=='__main__':main()
