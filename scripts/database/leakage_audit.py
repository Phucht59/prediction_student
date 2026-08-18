import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sqlalchemy import text
from src.database.connection import engine
ROOT=Path(__file__).resolve().parents[2]; forbidden={'G3','target','final_result','date_unregistration','score'}; violations=[]
with engine.connect() as c:
    rows=c.execute(text('SELECT snapshot_id,static_features,aggregate_features,temporal_summary FROM data.feature_snapshot'))
    for r in rows:
        payload={}; payload.update(r.static_features or {}); payload.update(r.aggregate_features or {}); payload.update(r.temporal_summary or {})
        bad=sorted(k for k in payload if k.lower() in {x.lower() for x in forbidden})
        if bad: violations.append({'snapshot_id':str(r.snapshot_id),'columns':bad})
with engine.connect() as c: checked=c.execute(text('SELECT count(*) FROM data.feature_snapshot')).scalar()
report={'status':'PASS' if not violations else 'FAIL','checked_snapshots':checked,'violations':violations,'forbidden_predictor_fields':sorted(forbidden)}
(ROOT/'artifacts/database/LEAKAGE_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(report['status'],checked,len(violations)); raise SystemExit(0 if not violations else 1)
