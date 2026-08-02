"""Materialize the evidence-policy release index from validated canonical artefacts."""
from __future__ import annotations
import hashlib, json, shutil, sys
from pathlib import Path
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.recommend_hybrid.validate_phase5 import generate_artifacts

OUT=ROOT/'artifacts/final/recommendation'; SRC=ROOT/'artifacts/recommend_hybrid/final'
def write(name, payload): (OUT/name).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    generate_artifacts(); OUT.mkdir(parents=True,exist_ok=True); (OUT/'redacted_examples').mkdir(exist_ok=True)
    metrics=json.loads((SRC/'FINAL_METRICS.json').read_text()); bootstrap=json.loads((SRC/'BOOTSTRAP_CONFIDENCE_INTERVALS.json').read_text())
    cfg=yaml.safe_load((ROOT/'configs/final/recommendation.yaml').read_text())
    write('final_recommendation_registry.json',{**cfg,'release_authority':'RECOMMEND_HYBRID_PHASE5_FINAL_PASS','neural_ranker_artifacts_excluded':True,'manifest':'artifacts/recommend_hybrid/final/FINAL_RELEASE_MANIFEST.json'})
    shutil.copy2(ROOT/'configs/recommend_hybrid/planning.yaml',OUT/'final_protocol_snapshot.yaml'); shutil.copy2(ROOT/'configs/recommend_hybrid/actions.yaml',OUT/'final_action_catalog.yaml'); shutil.copy2(ROOT/'artifacts/recommend_hybrid/scientific_labeling/source_registry.yaml',OUT/'final_source_registry.yaml')
    write('technical_evaluation.json',metrics); write('full_recommendation_metrics.json',metrics['overall']); write('bootstrap_ci.json',bootstrap); write('action_distribution.json',metrics['overall']['action_frequency']); write('dataset_stage_matrix.json',metrics['by_dataset_stage']); write('deterministic_replay.json',metrics['reproducibility']); write('claim_audit.json',{'causal_effectiveness_claimed':False,'expert_validated':False,'neural_ranker_released':False,'technical_validity_only':True})
    # Existing canonical plans are retained as the full operational artefact; redact identifiers in examples only.
    rows=[]
    for line in (OUT/'recommendation_plans.jsonl').read_text(encoding='utf-8').splitlines() if (OUT/'recommendation_plans.jsonl').exists() else []:
        r=json.loads(line); rows.append({'plan_id':r.get('plan_id'),'dataset':r.get('dataset'),'status':r.get('status'),'actions':json.dumps(r.get('actions',[]),sort_keys=True)})
    if rows: pd.DataFrame(rows).to_parquet(OUT/'recommendations.parquet',index=False)
    for path in sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.sha256'):
        pass
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.sha256')
    (OUT/'checksums.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files),encoding='utf-8')
    print('FINAL_EVIDENCE_RECOMMENDATIONS_BUILT')
if __name__=='__main__': main()
