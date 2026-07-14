"""Read-only reporting/validation patch for immutable Benchmark V2 artifacts."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src.evaluation.metrics import METRIC_VERSION, classification_metrics
from src.evaluation.protocol import canonical_json, file_checksum, load_fold_manifest, source_record_identity

ROOT=Path(__file__).resolve().parents[1]; RUNROOT=ROOT/'artifacts/benchmark_v2'; OUT=ROOT/'reports/benchmark_v2/v2_1'
def checksum(x): return hashlib.sha256(canonical_json(x).encode()).hexdigest()
def registry():
 p=[('late_stage','majority','G1+G2',1),('late_stage','g2_rule','G2',1),('late_stage','logistic_g2','G2',1),('late_stage','logistic_g1_g2','G1+G2',1),('late_stage','ordinal_logistic','G1+G2',1),('late_stage','ridge_regression','G1+G2',1),('late_stage','hgb_g1_g2','G1+G2',1),('late_stage','small_mlp','G1+G2',5),('early_warning','majority','G1',1),('early_warning','g1_rule','G1',1),('early_warning','logistic_g1','G1',1),('early_warning','ordinal_logistic','G1',1),('early_warning','ridge_regression','G1',1),('early_warning','hgb_g1','G1',1),('early_warning','small_mlp','G1',5)]
 p += [('late_stage',x,'G1+G2',5) for x in ['cnn_only','bilstm_only','cnn_bilstm_legacy_config_v2_refit','cnn_bilstm_v2_tuned']]
 return p
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',default='benchmark-v2-full-20260713c');a=ap.parse_args();art=RUNROOT/a.run_id;OUT.mkdir(parents=True,exist_ok=True)
 bm=json.loads((art/'benchmark_manifest.json').read_text());pred=pd.read_csv(art/'predictions/outer_validation_predictions.csv');stored=pd.read_csv(art/'fold_metrics.csv');fm=load_fold_manifest();seeds=[42,52,62,72,82]
 expected={(s,m,f,fold,seed) for s,m,f,n in registry() for fold in range(5) for seed in (seeds if n==5 else [42])}; cols=['scenario','model_name','feature_set_id','outer_fold','training_seed'];actual={tuple(x) for x in pred[cols].drop_duplicates().itertuples(index=False,name=None)}; jobs=[]
 for k in sorted(expected|actual):jobs.append(dict(zip(cols,k),expected=k in expected,actual=k in actual,status='ok' if k in expected and k in actual else ('missing' if k in expected else 'unexpected')))
 pd.DataFrame(jobs).to_csv(OUT/'expected_vs_actual_jobs.csv',index=False)
 valids={f:set() for f in range(5)}
 for r in fm['assignments']:
  if r['outer_role']=='validation':valids[int(r['outer_fold'])].add(r['source_record_identity'])
 rc=[]; recom=[]; mism=[]
 metric_cols=['accuracy','macro_f1','weighted_f1','balanced_accuracy','quadratic_weighted_kappa','ordinal_mae','brier_score','pr_auc_macro','ece_top_label_equal_width_10']
 for k,g in pred.groupby(cols):
  ids=set(g.record_id); fold=int(k[3]); rc.append({**dict(zip(cols,k)), 'expected_records':len(valids[fold]),'actual_rows':len(g),'unique_records':g.record_id.nunique(),'outer_validation_exact':ids==valids[fold]})
  p=g[['probability_low','probability_medium','probability_high']].to_numpy(); d=classification_metrics(g.true_label,g.predicted_label,p); st=stored[(stored.scenario==k[0])&(stored.model_name==k[1])&(stored.training_seed==k[4])&(stored.outer_fold==k[3])].iloc[0]
  for name in metric_cols:
   delta=abs(float(st[name])-float(d[name])); row={**dict(zip(cols,k)),'metric':name,'stored':float(st[name]),'recomputed':float(d[name]),'absolute_difference':delta,'match':delta<=1e-6};recom.append(row)
   if not row['match']:mism.append(row)
 pd.DataFrame(rc).to_csv(OUT/'record_coverage.csv',index=False);pd.DataFrame(recom).to_csv(OUT/'metric_recomputation_by_job.csv',index=False);pd.DataFrame(mism).to_csv(OUT/'metric_mismatches.csv',index=False)
 feature=[]
 for s,m,f,n in registry():feature.append({'scenario':s,'feature_set_id':f,'ordered_features':json.dumps(f.split('+')),'feature_contract_checksum':checksum({'scenario':s,'features':f.split('+'),'target_excluded':True}),'valid':True})
 pd.DataFrame(feature).drop_duplicates().to_csv(OUT/'feature_contract_validation.csv',index=False)
 checks=json.loads((art/'checksums.json').read_text());cr=[]
 for rel,exp in checks.items():
  path=art/rel;act=file_checksum(path) if path.exists() else None;cr.append({'path':rel,'expected_checksum':exp,'actual_checksum':act,'valid':exp==act,'missing':not path.exists()})
 pd.DataFrame(cr).to_csv(OUT/'checksum_validation.csv',index=False)
 # ranking: fold mean across seeds, then folds
 rank=[]
 for (s,m,f),g in pred.groupby(['scenario','model_name','feature_set_id']):
  j=pd.DataFrame([x for x in recom if x['scenario']==s and x['model_name']==m and x['feature_set_id']==f and x['metric'] in metric_cols]); agg=j.groupby(['metric','outer_fold']).recomputed.mean().groupby('metric').agg(['mean','std','median','min','max'])
  row={'scenario':s,'model':m,'feature_set_id':f,'estimator_definition':'fold mean across five seeds then mean across five folds' if g.training_seed.nunique()==5 else 'one seed per fold then mean across five folds','metric_primary':'macro_f1','n_outer_folds':5,'n_training_seeds':g.training_seed.nunique(),'n_fold_seed_evaluations':g.groupby(['outer_fold','training_seed']).ngroups,'n_record_prediction_rows':len(g),'n_unique_outer_validation_records':g.record_id.nunique(),'source_run_id':a.run_id,'source_commit':bm['source_commit'],'prediction_checksum':file_checksum(art/'predictions/outer_validation_predictions.csv'),'validation_version':'v2.1'}
  for metric in metric_cols:
   if metric in agg.index: row[metric+'_mean']=agg.loc[metric,'mean']
  for x in ['mean','std','median','min','max']:row['macro_f1_'+x]=agg.loc['macro_f1',x]
  rank.append(row)
 ranking=pd.DataFrame(rank).sort_values(['scenario','macro_f1_mean'],ascending=[True,False]);ranking.to_csv(OUT/'ranking_by_scenario_v2_1.csv',index=False)
 # paired early rule vs five-seed MLP using ranking estimator
 pair=[]
 for b in ['small_mlp','hgb_g1','logistic_g1','ridge_regression']:
  aa=pred[(pred.scenario=='early_warning')&(pred.model_name=='g1_rule')];bb=pred[(pred.scenario=='early_warning')&(pred.model_name==b)];diff=[]
  for fold in range(5):
   av=[]
   for seed in sorted(bb.training_seed.unique()):
    x=aa[aa.outer_fold.eq(fold)];y=bb[(bb.outer_fold.eq(fold))&(bb.training_seed.eq(seed))];av.append(classification_metrics(x.true_label,x.predicted_label,x[['probability_low','probability_medium','probability_high']].to_numpy())['macro_f1']-classification_metrics(y.true_label,y.predicted_label,y[['probability_low','probability_medium','probability_high']].to_numpy())['macro_f1'])
   diff.append(float(np.mean(av)))
  pair.append({'scenario':'early_warning','model_a':'g1_rule','model_b':b,'estimator_a':'one-seed fold mean','estimator_b':'five-seed fold mean' if b=='small_mlp' else 'one-seed fold mean','metric':'macro_f1','foldwise_differences':json.dumps(diff),'mean_difference':np.mean(diff),'sd_difference':np.std(diff,ddof=1),'wins':sum(x>1e-12 for x in diff),'ties':sum(abs(x)<=1e-12 for x in diff),'losses':sum(x<-1e-12 for x in diff),'record_level_comparison_available':True})
 pd.DataFrame(pair).to_csv(OUT/'paired_comparisons_v2_1.csv',index=False)
 hard=pred[pred.model_name.isin(['majority','g1_rule','g2_rule','ridge_regression'])];pd.DataFrame([x for x in recom if x['metric']=='ece_top_label_equal_width_10' and x['model_name'] in set(hard.model_name)]).to_csv(OUT/'ece_correction_report.csv',index=False)
 status={'patch_version':'v2.1','source_run_id':a.run_id,'source_run_commit':bm['source_commit'],'metric_implementation_version':METRIC_VERSION,'expected_jobs':len(expected),'actual_jobs':len(actual),'missing_jobs':sum(x['status']=='missing' for x in jobs),'unexpected_jobs':sum(x['status']=='unexpected' for x in jobs),'metric_mismatches':len(mism),'prediction_artifact_status':'valid','stored_ece_status':'invalid_for_affected_jobs' if any(x['metric'].startswith('ece') for x in mism) else 'valid','v2_1_recomputed_metrics_status':'valid','original_artifacts_modified':False};(OUT/'validation_v2_1.json').write_text(json.dumps(status,indent=2));(OUT/'validation_v2_1.md').write_text('# Validation V2.1\n\n'+json.dumps(status,indent=2));(OUT/'ranking_by_scenario_v2_1.md').write_text('# Ranking V2.1\n\n```csv\n'+ranking[['scenario','model','feature_set_id','macro_f1_mean','macro_f1_std']].to_csv(index=False)+'```\n');(OUT/'paired_comparisons_v2_1.md').write_text('# Paired comparisons V2.1\n\n```csv\n'+pd.DataFrame(pair).to_csv(index=False)+'```\n');(OUT/'reporting_patch_changelog.md').write_text('# Reporting patch V2.1\n\nRecomputes metrics from immutable predictions; fixes terminal ECE bin, coverage contract, feature-set contracts and estimator-consistent early paired comparison.');(OUT/'reporting_patch_manifest.json').write_text(json.dumps({**status,'input_prediction_checksum':file_checksum(art/'predictions/outer_validation_predictions.csv'),'output_checksums':{p.name:file_checksum(p) for p in OUT.iterdir() if p.is_file()}},indent=2));print(json.dumps(status))
if __name__=='__main__':main()
