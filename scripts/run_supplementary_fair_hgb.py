"""Predeclared, development-only G1/G2 HGB control experiment."""
from __future__ import annotations
import csv, hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATASETS, ROOT_DIR
from src.data_pipeline import process_target_and_stratify
from src.model_selection import make_folds
from src.postgres_data_source import load_dataset_version_from_postgres, reconstruct_splits_from_run
OUT=ROOT_DIR/'artifacts/supplementary/fair_g1_g2_baselines'; RUN='5a0b5041-5216-4a48-9e46-b0c16ab14866'
GRID={"learning_rate":[.03,.05,.1],"max_iter":[50,100,200],"max_leaf_nodes":[7,15,31],"l2_regularization":[0.,.1,1.]}
def fit_score(xtr,ytr,xva,yva,p):
 m=HistGradientBoostingClassifier(random_state=42,**p);m.fit(xtr,ytr);return f1_score(yva,m.predict(xva),average='macro',zero_division=0)
def main():
 OUT.mkdir(parents=True,exist_ok=False);spec=DATASETS['student-mat'];raw,_=load_dataset_version_from_postgres('student-mat',1)
 frame=process_target_and_stratify(raw.copy(),spec.target_col,spec.kind,'3class').dropna(subset=['_strat_target']).drop(columns=['_strat_target']); train,_=reconstruct_splits_from_run(frame,RUN)
 x=train[['G1','G2']].to_numpy();y=train[spec.target_col].to_numpy(int); outer=make_folds(train,spec.target_col,n_splits=5,seed=42); trials=[]; folds=[]
 combos=[dict(zip(GRID,k)) for k in itertools.product(*GRID.values())]
 for oi,(ot,ov) in enumerate(outer):
  inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=42+oi); best=None
  for ti,p in enumerate(combos):
   vals=[fit_score(x[ot][a],y[ot][a],x[ot][b],y[ot][b],p) for a,b in inner.split(x[ot],y[ot])]; score=float(np.mean(vals));trials.append({'outer_fold':oi,'trial':ti,**p,'inner_macro_f1_mean':score})
   if best is None or score>best[0]:best=(score,p)
  score=fit_score(x[ot],y[ot],x[ov],y[ov],best[1]);folds.append({'outer_fold':oi,**best[1],'best_inner_macro_f1':best[0],'outer_macro_f1':score,'n_train':len(ot),'n_validation':len(ov)})
 def write(name,rows):
  with (OUT/name).open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write('inner_trials.csv',trials);write('outer_fold_metrics.csv',folds)
 summary={'label':'supplementary post-hoc analysis','outer_macro_f1_mean':float(np.mean([r['outer_macro_f1'] for r in folds])),'outer_macro_f1_std':float(np.std([r['outer_macro_f1'] for r in folds],ddof=1)),'outer_folds':5,'inner_folds':3,'seed':42,'objective':'Macro-F1','locked_test_used':False}
 protocol={'label':'supplementary post-hoc analysis','data':'frozen train pool only','feature_set':['G1','G2'],'search_space':GRID,'search_predeclared':True,'locked_test_used':False,'final_model_changed':False}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 (OUT/'README.md').write_text('# Fair G1/G2 HGB control\n\nThis is a supplementary post-hoc control experiment using the same G1/G2 information, 5×3 development-only nested CV and a predeclared grid. It does not replace the final model, does not use the locked test, and must not be merged with the HGB full-feature locked score 0.9463.\n',encoding='utf-8')
 checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name != 'artifact_checksums.json'};(OUT/'artifact_checksums.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
if __name__=='__main__':main()
