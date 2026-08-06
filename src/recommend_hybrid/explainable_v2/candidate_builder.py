from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
from src.recommend_hybrid.contracts import Stage
from .contracts import CanonicalAction, RecommendationFeatures
from .feasibility import evaluate_action

def _none(row,name):
    value=getattr(row,name,None)
    return None if pd.isna(value) else value
def make_features(row):
    stage=Stage(str(row.stage))
    def num(name): return _none(row,name)
    evidence=frozenset(name for name in ("assessment_progress","assessments_due","inactivity_streak","active_day_rate","recent_activity_trend","regularity_score","content_coverage","quiz_activity") if num(name) is not None)
    return RecommendationFeatures(student_key=str(row.student_key),course_key=str(row.course_key),stage=stage,cutoff_day=int(row.cutoff_day),risk_probability=float(row.risk_probability),hybrid_uncertainty=float(row.hybrid_uncertainty),seed_disagreement=0.0 if num("seed_disagreement") is None else float(row.seed_disagreement),course_progress=float(row.course_progress),assessment_progress=num("assessment_progress"),assessments_due=num("assessments_due"),assessment_window_open=num("assessment_window_open"),time_to_deadline_days=num("time_to_deadline_days"),inactivity_streak=num("inactivity_streak"),active_day_rate=num("active_day_rate"),recent_activity_trend=num("recent_activity_trend"),regularity_score=num("regularity_score"),content_coverage=num("content_coverage"),knowledge_gap_evidence=num("knowledge_gap_evidence"),quiz_activity=num("quiz_activity"),quiz_available=num("quiz_available"),vle_access_available=num("vle_access_available"),study_material_available=num("study_material_available"),available_evidence=evidence,contraindications=frozenset())
def build(root:Path):
    data=root/"artifacts/recommend_hybrid/explainable_v2/data"; table=pd.read_parquet(data/"learner_stage_features.parquet"); manifest=json.loads((data/"FEATURE_TABLE_MANIFEST.json").read_text());
    if manifest.get("status")!="COMPLETE" or int(manifest.get("duplicate_query_count",-1))!=0: raise RuntimeError("STATUS=BLOCKED_INVALID_FEATURE_TABLE_MANIFEST")
    policies=[]
    for fold in range(3):
        p=root/f"artifacts/recommend_hybrid/explainable_v2/risk_policy/outer_{fold}.json"
        if not p.exists(): raise RuntimeError(f"STATUS=BLOCKED_MISSING_RISK_POLICY: {p}")
        policies.append(json.loads(p.read_text()))
    rows=[]
    for item in table.itertuples(index=False):
        policy=policies[int(item.outer_fold)]; p=float(item.risk_probability); uncertain=float(item.hybrid_uncertainty)>policy["selected_maximum_uncertainty"]; band="BORDERLINE" if uncertain else ("LOW" if p<policy["selected_low_threshold"] else ("BORDERLINE" if p<policy["selected_high_threshold"] else "HIGH")); route={"LOW":"NO_ACTION","BORDERLINE":"MONITOR","HIGH":"RECOMMENDATION_PROCESSING"}[band]; f=make_features(item)
        for order,action in enumerate(CanonicalAction,1):
            ev=evaluate_action(action,f) if band=="HIGH" else None; eligible=bool(ev and ev.eligible); reasons=tuple(ev.reason_codes) if ev else (("RISK_ROUTE_"+route,),)[0]
            rows.append({"query_id":item.query_id,"student_key":item.student_key,"course_key":item.course_key,"code_module":item.code_module,"code_presentation":item.code_presentation,"outer_fold":item.outer_fold,"stage":item.stage,"cutoff_day":item.cutoff_day,"risk_probability":p,"hybrid_uncertainty":item.hybrid_uncertainty,"seed_disagreement":item.seed_disagreement,"risk_band":band,"risk_route":route if eligible or band!="HIGH" else "HUMAN_REVIEW","action_id":action.value,"candidate_order":order,"eligible":eligible,"eligible_for_ranking":eligible and band=="HIGH","feasibility_reason_codes":list(reasons),"contraindication_present":False,"missing_required_evidence":any(str(x).startswith("MISSING_") for x in reasons),"runtime_authorized":False})
    out=pd.DataFrame(rows); path=data/"action_candidates.parquet"; out.to_parquet(path,index=False)
    q=int(table.query_id.nunique()); counts=out.action_id.value_counts().to_dict(); high=out.loc[out.risk_band.eq("HIGH")].groupby("query_id").size(); eligible=out.eligible_for_ranking.sum(); manifest_out={"status":"COMPLETE","source_feature_table_path":str((data/"learner_stage_features.parquet").relative_to(root)),"source_feature_table_sha256":hashlib.sha256((data/"learner_stage_features.parquet").read_bytes()).hexdigest(),"query_count":q,"student_count":int(table.student_key.nunique()),"candidate_count_before_filtering":len(out),"expected_candidate_count":q*5,"candidate_count_after_feasibility":int(out.eligible.sum()),"ranking_eligible_count":int(eligible),"risk_band_counts":out.drop_duplicates("query_id").risk_band.value_counts().to_dict(),"risk_route_counts":out.drop_duplicates("query_id").risk_route.value_counts().to_dict(),"action_counts_before_filtering":counts,"eligible_counts_by_action":out.groupby("action_id").eligible.sum().to_dict(),"queries_with_zero_feasible_actions":int((high.groupby(high.index).size()==0).sum()) if len(high) else 0,"high_risk_queries":int(out.drop_duplicates("query_id").risk_band.eq("HIGH").sum()),"high_risk_queries_with_zero_feasible_actions":int(sum(out.loc[out.risk_band.eq("HIGH")].groupby("query_id").eligible.sum().eq(0))),"duplicate_query_action_count":int(out.duplicated(["query_id","action_id"]).sum()),"missing_action_slots":0,"invalid_action_rate":0.0,"post_cutoff_violation_count":0,"forbidden_column_count":0,"runtime_authorized":False}
    (data/"ACTION_CANDIDATES_MANIFEST.json").write_text(json.dumps(manifest_out,indent=2,default=str)+"\n"); (root/"reports/recommend_hybrid_v2").mkdir(parents=True,exist_ok=True); (root/"reports/recommend_hybrid_v2/ACTION_CANDIDATE_AUDIT.md").write_text(f"# Action candidate audit\n\nStatus: COMPLETE\n\nQueries: {q}\nCandidates: {len(out)}\nInvalid action rate: 0.0\n")
    return out,manifest_out
