"""Idempotent Core-SQL persistence functions; no Hybrid inference logic lives here."""
from __future__ import annotations
import json
import math
from typing import Any, Iterable, Mapping
from sqlalchemy import text
from src.database.connection import engine, transaction

def _json(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)): return v
    if isinstance(v, (dict, list)):
        def clean(x):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
            if isinstance(x, dict): return {k:clean(y) for k,y in x.items()}
            if isinstance(x, list): return [clean(y) for y in x]
            return x
        return json.dumps(clean(v), default=str, allow_nan=False)
    return json.loads(json.dumps(v, default=str))
def _one(sql: str, p: Mapping[str, Any]):
    with transaction() as c: return c.execute(text(sql), p).scalar_one()

def upsert_student(external_student_id, full_name=None, email=None, metadata=None):
    return _one("""INSERT INTO catalog.student(external_student_id,full_name,email) VALUES (:k,:n,:e)
    ON CONFLICT(external_student_id) DO UPDATE SET full_name=COALESCE(EXCLUDED.full_name,catalog.student.full_name),email=COALESCE(EXCLUDED.email,catalog.student.email),updated_at=NOW() RETURNING student_id""", {"k":external_student_id,"n":full_name,"e":email})
create_student = upsert_student
def upsert_course(course_code, presentation, name=None, metadata=None):
    return _one("""INSERT INTO catalog.course(course_code,course_name,presentation) VALUES (:c,:n,:p)
    ON CONFLICT(course_code,presentation) DO UPDATE SET course_name=COALESCE(EXCLUDED.course_name,catalog.course.course_name) RETURNING course_id""", {"c":course_code,"p":presentation,"n":name or course_code})
def upsert_enrollment(student_id, course_id, external_enrollment_id=None, metadata=None):
    return _one("""INSERT INTO catalog.enrollment(student_id,course_id,enrolled_at,status) VALUES (:s,:c,NOW(),'ACTIVE')
    ON CONFLICT(student_id,course_id) DO UPDATE SET status='ACTIVE' RETURNING enrollment_id""", {"s":student_id,"c":course_id})
def upsert_dataset(dataset_key, dataset_name):
    return _one("INSERT INTO data.dataset(dataset_key,dataset_name) VALUES (:k,:n) ON CONFLICT(dataset_key) DO UPDATE SET dataset_name=EXCLUDED.dataset_name RETURNING dataset_id", {"k":dataset_key,"n":dataset_name})
def upsert_dataset_version(dataset_id, version, source_hash=None, description=None):
    return _one("""INSERT INTO data.dataset_version(dataset_id,version,source_hash,description) VALUES (:d,:v,:h,:x) ON CONFLICT(dataset_id,version) DO UPDATE SET source_hash=COALESCE(EXCLUDED.source_hash,data.dataset_version.source_hash),description=COALESCE(EXCLUDED.description,data.dataset_version.description) RETURNING dataset_version_id""", {"d":dataset_id,"v":version,"h":source_hash,"x":description})

def upsert_temporal_observations(rows: Iterable[Mapping[str, Any]], chunk_size=2000):
    q=text("""INSERT INTO data.temporal_observation(enrollment_id,time_index,observation_date,features) VALUES (:e,:t,:d,:f)
    ON CONFLICT(enrollment_id,time_index) DO UPDATE SET observation_date=EXCLUDED.observation_date,features=EXCLUDED.features""")
    total=0; batch=[]
    for r in rows:
        batch.append({"e":r["enrollment_id"],"t":int(r["time_index"]),"d":r.get("observation_date"),"f":_json(r.get("features",{}))})
        if len(batch)>=chunk_size:
            with transaction() as c: c.execute(q,batch)
            total+=len(batch); batch=[]
    if batch:
        with transaction() as c: c.execute(q,batch)
        total+=len(batch)
    return total
def upsert_feature_snapshot(enrollment_id,dataset_version_id,stage,cutoff_value=None,static_features=None,aggregate_features=None,temporal_summary=None):
    return _one("""INSERT INTO data.feature_snapshot(enrollment_id,dataset_version_id,stage,cutoff_value,static_features,aggregate_features,temporal_summary) VALUES (:e,:d,:s,:c,:sf,:af,:ts)
    ON CONFLICT(enrollment_id,dataset_version_id,stage) DO UPDATE SET cutoff_value=EXCLUDED.cutoff_value,static_features=EXCLUDED.static_features,aggregate_features=EXCLUDED.aggregate_features,temporal_summary=EXCLUDED.temporal_summary RETURNING snapshot_id""", {"e":enrollment_id,"d":dataset_version_id,"s":stage,"c":str(cutoff_value) if cutoff_value is not None else None,"sf":_json(static_features or {}),"af":_json(aggregate_features or {}),"ts":_json(temporal_summary or {})})

def register_model(model_key="hybrid",display_name="Hybrid",model_type="CNN + BiLSTM Hybrid",metadata=None):
    with transaction() as c:
        existing=c.execute(text("SELECT model_id FROM prediction.model WHERE model_key=:k"), {"k":model_key}).scalar_one_or_none()
        if existing:
            c.execute(text("UPDATE prediction.model SET display_name=:n,model_type=:t WHERE model_id=:i"), {"i":existing,"n":display_name,"t":model_type}); return existing
        return c.execute(text("INSERT INTO prediction.model(model_key,display_name,version,model_type) VALUES (:k,:n,'final',:t) RETURNING model_id"), {"k":model_key,"n":display_name,"t":model_type}).scalar_one()
def get_active_model(model_key="hybrid"):
    with engine.connect() as c: return c.execute(text("SELECT model_id FROM prediction.model WHERE model_key=:k AND is_active"), {"k":model_key}).scalar_one_or_none()
def set_active_model(model_id):
    with transaction() as c:
        c.execute(text("UPDATE prediction.model SET is_active=false WHERE model_id<>:i"), {"i":model_id}); c.execute(text("UPDATE prediction.model SET is_active=true WHERE model_id=:i"), {"i":model_id})
def create_model_run(model_id,status="PENDING",metadata=None,dataset="unknown",task="risk_prediction"):
    if status not in {"PENDING","RUNNING","COMPLETED","FAILED"}: raise ValueError("invalid model run status")
    return _one("INSERT INTO prediction.model_run(model_id,dataset,task,status,metadata) VALUES (:m,:d,:t,:s,:x) RETURNING run_id", {"m":model_id,"d":dataset,"t":task,"s":status,"x":_json(metadata or {})})
def _finish(run_id,status,metadata=None):
    if status not in {"COMPLETED","FAILED"}: raise ValueError("invalid terminal status")
    with transaction() as c: c.execute(text("UPDATE prediction.model_run SET status=:s,metadata=COALESCE(:x,metadata),completed_at=NOW() WHERE run_id=:i"), {"i":run_id,"s":status,"x":_json(metadata)})
def complete_model_run(run_id,metadata=None): _finish(run_id,"COMPLETED",metadata)
def fail_model_run(run_id,metadata=None): _finish(run_id,"FAILED",metadata)
def upsert_prediction(enrollment_id,run_id,snapshot_id,stage,risk_probability,threshold,uncertainty=None,metadata=None):
    if not 0<=risk_probability<=1 or not 0<=threshold<=1 or (uncertainty is not None and uncertainty<0): raise ValueError("probability and threshold must be in [0,1]; uncertainty must be non-negative")
    return _one("""INSERT INTO prediction.prediction(enrollment_id,run_id,snapshot_id,stage,risk_probability,predicted_risk,threshold,uncertainty,metadata) VALUES (:e,:r,:s,:g,:p,:y,:t,:u,:m)
    ON CONFLICT(enrollment_id,run_id,stage) DO UPDATE SET snapshot_id=EXCLUDED.snapshot_id,risk_probability=EXCLUDED.risk_probability,predicted_risk=EXCLUDED.predicted_risk,threshold=EXCLUDED.threshold,uncertainty=EXCLUDED.uncertainty,metadata=EXCLUDED.metadata RETURNING prediction_id""", {"e":enrollment_id,"r":run_id,"s":snapshot_id,"g":stage,"p":risk_probability,"y":risk_probability>=threshold,"t":threshold,"u":uncertainty,"m":_json(metadata or {})})
def get_latest_prediction(enrollment_id):
    with engine.connect() as c: return c.execute(text("SELECT * FROM prediction.prediction WHERE enrollment_id=:e ORDER BY created_at DESC LIMIT 1"), {"e":enrollment_id}).mappings().one_or_none()
def get_predictions_for_enrollment(enrollment_id):
    with engine.connect() as c: return c.execute(text("SELECT * FROM prediction.prediction WHERE enrollment_id=:e ORDER BY created_at"), {"e":enrollment_id}).mappings().all()
def get_predictions_for_run(run_id):
    with engine.connect() as c: return c.execute(text("SELECT * FROM prediction.prediction WHERE run_id=:r ORDER BY created_at"), {"r":run_id}).mappings().all()
def create_recommendation(enrollment_id,prediction_id=None,status="PENDING",metadata=None):
    return _one("INSERT INTO recommendation.recommendation(prediction_id,risk_band,route_status,metadata) VALUES (:p,:b,:s,:m) RETURNING recommendation_id", {"p":prediction_id,"b":status,"s":status,"m":_json(metadata or {})})
def add_recommendation_item(recommendation_id,action_id,rank=1,rationale=None,metadata=None):
    return _one("INSERT INTO recommendation.recommendation_item(recommendation_id,action_id,rank_position,score,explanation) VALUES (:r,:a,:n,NULL,:x) ON CONFLICT(recommendation_id,action_id) DO UPDATE SET rank_position=EXCLUDED.rank_position,explanation=EXCLUDED.explanation RETURNING recommendation_item_id", {"r":recommendation_id,"a":action_id,"n":rank,"x":rationale})
def get_recommendation(recommendation_id):
    with engine.connect() as c: return c.execute(text("SELECT * FROM recommendation.recommendation WHERE recommendation_id=:i"), {"i":recommendation_id}).mappings().one_or_none()
def write_system_event(event_type,entity_type=None,entity_id=None,payload=None):
    return _one("INSERT INTO audit.system_event(event_type,entity_type,entity_id,payload) VALUES (:t,:e,:i,:p) RETURNING event_id", {"t":event_type,"e":entity_type,"i":str(entity_id) if entity_id is not None else None,"p":_json(payload or {})})
