from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor

ROOT=Path(__file__).resolve().parents[1]
REDACTED="postgresql://<redacted>@localhost:5432/student_predict"

def write_json(path,payload): Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True,default=str),encoding="utf-8")
def connect(dsn): return psycopg2.connect(dsn)
def fetch(cur,sql,params=None): cur.execute(sql,params); return [dict(row) for row in cur.fetchall()]

def connection_profile(dsn,key):
    try:
        with connect(dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("select current_database() database,current_user,session_user,current_schema() schema,current_setting('server_version') server_version,pg_is_in_recovery() in_recovery")
                base=dict(cur.fetchone()); cur.execute("select rolname,rolsuper,rolcreatedb,rolcreaterole,rolcanlogin from pg_roles where rolname=current_user"); base.update(dict(cur.fetchone()))
                base.update({"environment_key":key,"status":"available","classification":"migration_admin_connection_only" if base["rolsuper"] or base["rolcreatedb"] or base["rolcreaterole"] else "least_privileged_application_connection","valid_as_application_permission_evidence":not(base["rolsuper"] or base["rolcreatedb"] or base["rolcreaterole"])})
                return base
    except Exception as error:
        return {"environment_key":key,"status":"unavailable","classification":"invalid_or_unavailable_connection","error_type":type(error).__name__,"error":"connection failed; credentials redacted"}

def audit(dsn):
    with connect(dsn) as conn:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            schemas=fetch(cur,"select schema_name from information_schema.schemata where schema_name not in ('pg_catalog','information_schema') and schema_name not like 'pg_toast%' order by 1")
            tables=fetch(cur,"""select n.nspname schema_name,c.relname table_name,c.relkind,pg_get_userbyid(c.relowner) owner,pg_total_relation_size(c.oid) total_bytes,obj_description(c.oid) comment from pg_class c join pg_namespace n on n.oid=c.relnamespace where n.nspname not in ('pg_catalog','information_schema') and n.nspname not like 'pg_toast%' and c.relkind in ('r','p','v','m') order by 1,2""")
            columns=fetch(cur,"""select table_schema,table_name,column_name,ordinal_position,data_type,is_nullable,column_default from information_schema.columns where table_schema not in ('pg_catalog','information_schema') order by table_schema,table_name,ordinal_position""")
            constraints=fetch(cur,"""select n.nspname schema_name,c.relname table_name,con.conname constraint_name,con.contype constraint_type,pg_get_constraintdef(con.oid,true) definition,con.convalidated validated from pg_constraint con join pg_class c on c.oid=con.conrelid join pg_namespace n on n.oid=c.relnamespace where n.nspname not in ('pg_catalog','information_schema') order by 1,2,3""")
            indexes=fetch(cur,"""select schemaname schema_name,tablename table_name,indexname index_name,indexdef index_definition from pg_indexes where schemaname not in ('pg_catalog','information_schema') order by 1,2,3""")
            triggers=fetch(cur,"""select event_object_schema schema_name,event_object_table table_name,trigger_name,event_manipulation,action_statement from information_schema.triggers where trigger_schema not in ('pg_catalog','information_schema') order by 1,2,3""")
            sequences=fetch(cur,"select sequence_schema,sequence_name,data_type,start_value,minimum_value,maximum_value,increment from information_schema.sequences where sequence_schema not in ('pg_catalog','information_schema') order by 1,2")
            grants=fetch(cur,"""select table_schema,table_name,grantee,privilege_type from information_schema.role_table_grants where table_schema not in ('pg_catalog','information_schema') order by 1,2,3,4""")
            roles=fetch(cur,"select rolname,rolsuper,rolcreatedb,rolcreaterole,rolcanlogin,rolinherit from pg_roles where rolname in ('postgres','student_predict_app','student_predict_app_local') order by rolname")
            counts=[]
            for table in tables:
                if table["relkind"] not in ("r","p"): continue
                cur.execute(f'SELECT count(*) count FROM "{table["schema_name"]}"."{table["table_name"]}"'); counts.append({"schema_name":table["schema_name"],"table_name":table["table_name"],"row_count":cur.fetchone()["count"],"total_bytes":table["total_bytes"]})
            integrity={}
            names={row["table_name"] for row in tables if row["schema_name"]=="public"}
            if {"ml_experiment_runs","ml_run_record_splits","ml_predictions"}<=names:
                cur.execute("select count(*) count from ml_run_record_splits s left join ml_experiment_runs r on r.run_id=s.run_id where r.run_id is null"); integrity["orphan_splits"]=cur.fetchone()["count"]
                cur.execute("select count(*) count from ml_predictions p left join ml_run_record_splits s on s.run_id=p.run_id and s.record_id=p.record_id and s.split_name=p.split_name where s.run_id is null"); integrity["orphan_predictions"]=cur.fetchone()["count"]
                cur.execute("select count(*) count from (select run_id,record_id,split_name,count(*) from ml_predictions group by 1,2,3 having count(*)>1) q"); integrity["duplicate_prediction_keys"]=cur.fetchone()["count"]
                cur.execute("select count(*) count from ml_experiment_runs where status not in ('running','completed','failed')"); integrity["invalid_run_statuses"]=cur.fetchone()["count"]
            return {"schemas":schemas,"tables":tables,"columns":columns,"triggers":triggers,"sequences":sequences,"grants":grants,"integrity":integrity},counts,constraints,indexes,roles

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=["before","after"],required=True); parser.add_argument("--artifact-root",required=True); parser.add_argument("--report-root",required=True); args=parser.parse_args()
    root=Path(args.artifact_root); report=Path(args.report_root); root.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True)
    admin=os.environ.get("POSTGRES_TEST_DSN"); app=os.environ.get("POSTGRES_TEST_APP_DSN");
    if not admin: raise RuntimeError("POSTGRES_TEST_DSN is required")
    profiles=[connection_profile(admin,"POSTGRES_TEST_DSN")]
    if app: profiles.append(connection_profile(app,"POSTGRES_TEST_APP_DSN"))
    runtime=os.environ.get("POSTGRES_RUNTIME_APP_DSN")
    if runtime: profiles.append(connection_profile(runtime,"POSTGRES_RUNTIME_APP_DSN"))
    write_json(root/"postgres_connectivity_audit.json",{"database":"student_predict","redacted_dsn":REDACTED,"connections":profiles,"credential_redaction":"PASS","superuser_application_evidence_forbidden":True})
    schema,counts,constraints,indexes,roles=audit(admin); write_json(root/f"postgres_schema_{args.phase}.json",schema)
    import pandas as pd
    pd.DataFrame(counts).to_csv(root/f"postgres_counts_{args.phase}.csv",index=False); pd.DataFrame(constraints).to_csv(root/f"postgres_constraints_{args.phase}.csv",index=False); pd.DataFrame(indexes).to_csv(root/f"postgres_indexes_{args.phase}.csv",index=False); write_json(root/f"postgres_roles_{args.phase}.json",roles)
    if args.phase=="before":
        lines=["# PostgreSQL Audit Before", "",f"- Database: `student_predict`",f"- Non-system relations: {len(schema['tables'])}",f"- Integrity: `{json.dumps(schema['integrity'],sort_keys=True)}`","","No write was performed by this audit."]
        (report/"POSTGRES_AUDIT_BEFORE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"phase":args.phase,"relations":len(schema["tables"]),"integrity":schema["integrity"],"connections":[{"key":x["environment_key"],"status":x["status"],"classification":x["classification"]} for x in profiles]},indent=2))
if __name__=="__main__": main()
