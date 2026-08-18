from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from src.hybrid.data.uci import build_uci_combined, UCI_FORBIDDEN_PREDICTORS, UCI_CATEGORICAL_CONTEXT, UCI_NUMERIC_CONTEXT
from src.database.repository import upsert_student,upsert_course,upsert_enrollment,upsert_dataset,upsert_dataset_version,upsert_feature_snapshot,upsert_temporal_observations
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/database'; OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def import_one(key,path,subject,df,version_id):
    cid=upsert_course(subject,subject,subject.title()); students=set(); enrollments=set(); ns=nt=0
    for _,r in df.iterrows():
        sid=upsert_student(f'UCI:{r.global_student_group}'); students.add(str(sid)); eid=upsert_enrollment(sid,cid); enrollments.add(str(eid))
        static={k:(r[k].item() if hasattr(r[k],'item') else r[k]) for k in UCI_CATEGORICAL_CONTEXT+UCI_NUMERIC_CONTEXT if k in r.index and k not in UCI_FORBIDDEN_PREDICTORS}
        upsert_feature_snapshot(eid,version_id,'S0',0,static_features=static); ns+=1
        for stage, vals in (('S1',[(0,float(r.G1))]),('S2',[(0,float(r.G1)),(1,float(r.G2))])):
            upsert_feature_snapshot(eid,version_id,stage,len(vals),static_features=static,temporal_summary={'channels':['grade'],'length':len(vals)}); ns+=1
            nt+=upsert_temporal_observations({'enrollment_id':eid,'time_index':i,'features':{'grade':v/20.0}} for i,v in vals)
    return {'dataset_key':key,'version':'checked-in','source_path':str(path),'source_sha256':sha(path),'student_count':len(students),'course_count':1,'enrollment_count':len(enrollments),'snapshot_count':ns,'temporal_observation_count':nt}
def main():
    mat=ROOT/'data/raw/student-mat.csv'; por=ROOT/'data/raw/student-por.csv'; combined,_=build_uci_combined(mat,por)
    for key,path,subject in (('UCI_MAT',mat,'math'),('UCI_POR',por,'portuguese')):
        did=upsert_dataset(key,f'UCI Student {subject.title()}'); vid=upsert_dataset_version(did,'checked-in',sha(path),'G3 is never written to predictor JSON')
        m=import_one(key,path,subject,combined[combined.subject==subject],vid); m.update({'started_at':datetime.now(timezone.utc).isoformat(),'completed_at':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED'}); (OUT/f'{key}_IMPORT.json').write_text(json.dumps(m,indent=2),encoding='utf-8'); print(key,m['enrollment_count'],m['snapshot_count'])
if __name__=='__main__': main()
