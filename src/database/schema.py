from __future__ import annotations
from sqlalchemy import text
from src.database.connection import engine

EXPECTED = {
    "catalog": {"student": {"student_id","external_student_id"}, "course": {"course_id","course_code","course_name","presentation"}, "enrollment": {"enrollment_id","student_id","course_id"}},
    "data": {"dataset": {"dataset_id","dataset_key"}, "dataset_version": {"dataset_version_id","dataset_id","version"}, "temporal_observation": {"observation_id","enrollment_id","time_index","features"}, "feature_snapshot": {"snapshot_id","enrollment_id","dataset_version_id","stage"}},
    "prediction": {"model": {"model_id","model_key","display_name"}, "model_run": {"run_id","model_id","status"}, "prediction": {"prediction_id","enrollment_id","run_id","stage","risk_probability","predicted_risk","threshold"}},
    "recommendation": {"action": {"action_id","action_key"}, "recommendation": {"recommendation_id","prediction_id"}, "recommendation_item": {"recommendation_item_id","recommendation_id","action_id"}},
    "audit": {"system_event": {"event_id","event_type","payload"}},
}

def check_database_schema() -> dict:
    report = {"schemas": {}, "missing": [], "foreign_keys": [], "unique_constraints": [], "ok": True}
    with engine.connect() as c:
        for schema, tables in EXPECTED.items():
            exists = bool(c.execute(text("SELECT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name=:s)"), {"s":schema}).scalar())
            report["schemas"][schema] = exists
            for table, columns in tables.items():
                actual = set(c.execute(text("SELECT column_name FROM information_schema.columns WHERE table_schema=:s AND table_name=:t"), {"s":schema,"t":table}).scalars())
                if not exists: report["missing"].append(f"schema:{schema}")
                if not actual: report["missing"].append(f"table:{schema}.{table}")
                for col in columns - actual: report["missing"].append(f"column:{schema}.{table}.{col}")
        required_uniques = [("catalog","student",("external_student_id",)),("catalog","course",("course_code","presentation")),("catalog","enrollment",("student_id","course_id")),("data","dataset",("dataset_key",)),("data","dataset_version",("dataset_id","version")),("data","temporal_observation",("enrollment_id","time_index")),("data","feature_snapshot",("enrollment_id","dataset_version_id","stage"))]
        for schema, table, cols in required_uniques:
            found = bool(c.execute(text("""SELECT 1 FROM pg_constraint pc JOIN pg_class cl ON cl.oid=pc.conrelid JOIN pg_namespace pn ON pn.oid=cl.relnamespace WHERE pn.nspname=:s AND cl.relname=:t AND pc.contype IN ('u','p') AND (SELECT array_agg(att.attname ORDER BY k.ord) FROM unnest(pc.conkey) WITH ORDINALITY k(attnum,ord) JOIN pg_attribute att ON att.attrelid=cl.oid AND att.attnum=k.attnum)=:cols"""), {"s":schema,"t":table,"cols":list(cols)}).scalar())
            if not found: report["unique_constraints"].append(f"{schema}.{table}{cols}")
        fk_count = c.execute(text("SELECT count(*) FROM information_schema.table_constraints WHERE constraint_type='FOREIGN KEY' AND table_schema IN ('catalog','data','prediction','recommendation')")).scalar()
        if not fk_count: report["foreign_keys"].append("no foreign keys found")
    report["ok"] = not report["missing"] and not report["foreign_keys"] and not report["unique_constraints"]
    return report
