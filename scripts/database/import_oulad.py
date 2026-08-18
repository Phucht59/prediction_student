from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from src.hybrid.data.oulad import load_oulad_static_tables, build_compact_vle_daily, load_assessment_events, compute_weekly_features_at_cutoff, OULAD_CATEGORICAL_CONTEXT, OULAD_NUMERIC_CONTEXT
from src.database.repository import upsert_student,upsert_course,upsert_enrollment,upsert_dataset,upsert_dataset_version,upsert_feature_snapshot,upsert_temporal_observations
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/database'; OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    raw=ROOT/'data/raw'; vle=raw/'studentVle.csv'; m={'dataset_key':'OULAD','version':'checked-in-static','source_path':str(raw),'started_at':datetime.now(timezone.utc).isoformat()}
    missing_temporal=vle.read_text(encoding='utf-8',errors='ignore').startswith('version https://git-lfs.github.com/spec')
    courses,reg,base=load_oulad_static_tables(raw); did=upsert_dataset('OULAD','Open University Learning Analytics Dataset'); vid=upsert_dataset_version(did,'checked-in',sha(vle),'OULAD static and temporal source files'); students=set(); course_ids={}; enrollments=0; enrollment_map={}
    for _,r in base.iterrows():
        sid=upsert_student(f'OULAD:{r.id_student}'); students.add(str(sid)); k=(r.code_module,r.code_presentation); cid=course_ids.setdefault(k,upsert_course(r.code_module,r.code_presentation,r.code_module)); eid=upsert_enrollment(sid,cid); enrollment_map[(r.code_module,r.code_presentation,int(r.id_student))]=eid; enrollments+=1
    snapshots=temporal_count=0
    if not missing_temporal:
        vle_daily=build_compact_vle_daily(raw, ROOT/'artifacts/hybrid/phase1/runtime'); assessments=load_assessment_events(raw)
        for frac,stage in ((.20,'20'),(.35,'35'),(.50,'50'),(.75,'75'),(1.0,'FINAL')):
            eligible,view,_=compute_weekly_features_at_cutoff(base,vle_daily,assessments,frac,include_pre_end_withdrawals=(stage=='FINAL'))
            for i,r in eligible.iterrows():
                eid=enrollment_map[(r.code_module,r.code_presentation,int(r.id_student))]
                static={k:(r[k].item() if hasattr(r[k],'item') else r[k]) for k in OULAD_CATEGORICAL_CONTEXT+OULAD_NUMERIC_CONTEXT if k in r.index}
                snap=upsert_feature_snapshot(eid,vid,stage,float(frac),static_features=static,temporal_summary={'channels':view.metadata.get('channels',[]),'length':int(view.lengths[i])}); snapshots+=1
                obs=({'enrollment_id':eid,'time_index':j,'features':{name:float(view.temporal[i,j,k]) for k,name in enumerate(view.metadata.get('channels',[]))}} for j in range(int(view.lengths[i])) for k in [0] )
                temporal_count+=upsert_temporal_observations(obs,chunk_size=1000)
    m.update({'source_sha256':sha(vle),'student_count':len(students),'course_count':len(course_ids),'enrollment_count':enrollments,'snapshot_count':snapshots,'temporal_observation_count':temporal_count,'status':'COMPLETED','completed_at':datetime.now(timezone.utc).isoformat()})
    if missing_temporal: m['blocker']='studentVle.csv is a Git-LFS pointer; temporal weekly features remain pending'
    (OUT/'OULAD_IMPORT.json').write_text(json.dumps(m,indent=2),encoding='utf-8'); print('OULAD static',enrollments)
if __name__=='__main__': main()
