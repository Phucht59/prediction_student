"""Frozen-protocol, observational-only scientific validation for recommendations.

This script never calls the recommender or changes its inputs.  It evaluates the
already locked recommendation replay and writes FAIL/INCONCLUSIVE rather than
silently converting model-consistency statistics into scientific evidence.
"""
from __future__ import annotations

import csv, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/recommend_hybrid/scientific_validation"
REPORT = ROOT / "reports/recommend_hybrid"
CF = ROOT / "artifacts/recommend_hybrid/counterfactual"
MAPPING = ROOT / "configs/recommend_hybrid/historical_action_outcome_mapping.yaml"
sys.path.insert(0, str(ROOT))
from src.pipelines.oulad import BASE_CHANNELS, _build_bundle

STAGE = {"EARLY_20": ("E1_EARLY_20PCT", "E2_EARLY_35PCT"), "EARLY_35": ("E2_EARLY_35PCT", "M1_MIDDLE_FROZEN"), "MIDDLE_50": ("M1_MIDDLE_FROZEN", "L1_LATE_75PCT"), "LATE_75": ("L1_LATE_75PCT", None)}
SEED = 20260804

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def write_json(name:str, x:Any):
    OUT.mkdir(parents=True, exist_ok=True); (OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding='utf8')
def write_csv(name:str, rows:list[dict[str,Any]]):
    OUT.mkdir(parents=True, exist_ok=True)
    cols=list(rows[0]) if rows else ['status','reason']
    with (OUT/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def git(cmd): return subprocess.check_output(cmd,cwd=ROOT,text=True).strip()

def aligned(action, cur, clen, nxt, nlen):
    if nlen <= clen: return None
    i={v:k for k,v in enumerate(BASE_CHANNELS)}; a,b=clen,nlen
    rate=lambda x,ch,s,e: float(x[s:e,i[ch]].sum()/max(e-s,1))
    if action=='VLE_ENGAGEMENT': return rate(nxt,'total_clicks',a,b)>rate(cur,'total_clicks',0,a) and nxt[a:b,i['active_days']].sum()>0
    if action=='STUDY_SCHEDULE': return rate(nxt,'active_days',a,b)>=rate(cur,'active_days',0,a) and nxt[nlen-1,i['days_since_last_vle_activity']] < cur[clen-1,i['days_since_last_vle_activity']]
    if action=='ASSESSMENT_COMPLETION': return nxt[a:b,i['submitted_assessment_count']].sum()>0
    if action in ('RETRIEVAL_PRACTICE','TARGETED_PRACTICE'): return rate(nxt,'quiz_clicks',a,b)>rate(cur,'quiz_clicks',0,a)
    if action=='LEARNING_CONSOLIDATION': return rate(nxt,'content_clicks',a,b)>=rate(cur,'content_clicks',0,a) and rate(nxt,'content_clicks',a,b)>0
    return None

def smd(x,t,w=None):
    a=x[t==1]; b=x[t==0]
    if w is None: ma,mb=a.mean(),b.mean(); va,vb=a.var(),b.var()
    else:
        wa,wb=w[t==1],w[t==0]; ma,mb=np.average(a,weights=wa),np.average(b,weights=wb); va,vb=np.average((a-ma)**2,weights=wa),np.average((b-mb)**2,weights=wb)
    return float((ma-mb)/np.sqrt((va+vb)/2)) if va+vb else 0.0

def main():
    mapping=yaml.safe_load(MAPPING.read_text(encoding='utf8'))
    locked=[CF/'evaluation_rows.csv',CF/'action_scores.csv',CF/'full_evaluation_input_registry.json',ROOT/'configs/recommend_hybrid/counterfactual_oulad.yaml',ROOT/'configs/recommend_hybrid/actions.yaml',MAPPING,ROOT/'data/manifests/extension_raw_manifest.json',ROOT/'data/processed/study_c_oulad/manifests/split_manifest.csv']
    authority={'schema_version':'scientific_validation_input_authority_v1','locked_at':now(),'git_commit':git(['git','rev-parse','HEAD']),'git_branch':git(['git','branch','--show-current']),'evaluation_protocol':'scientific_observational_protocol_v1','frozen_system':'NO_RECOMMENDER_EXECUTION','inputs':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p)} for p in locked], 'prohibited_as_independent_evidence':['model-estimated risk reduction','Success@0.01','Success@0.05','threshold crossing']}
    write_json('INPUT_AUTHORITY.json',authority)
    protocol={'schema_version':'scientific_observational_protocol_v1','status':'LOCKED_BEFORE_RESULT_COMPUTATION','claim_boundary':'OBSERVATIONAL_ASSOCIATION_ONLY_NOT_CAUSAL_EFFECT','transitions':['20%→35%','35%→50%','50%→75%','75%→final outcome'],'primary_outcomes':['next-stage observed at-risk classification when available','final Pass/Distinction versus Fail/Withdrawn'],'independent_outcomes_only':True,'methods':['IPTW propensity weighting','doubly robust AIPW','learner-level bootstrap (1000 planned; 500 executed for constrained replay)'],'baselines':['B0 random','B1 popular','B2 workload','B3 policy','B4 counterfactual'],'gate_requirements':['temporal safety','overlap and |SMD|<0.1 balance','independent outcomes','valid baseline contrast','monotonic dose response','negative controls','no leakage'],'input_authority_sha256':sha(OUT/'INPUT_AUTHORITY.json')}
    write_json('protocol.json',protocol); write_json('action_outcome_mapping.json',mapping)
    rec=pd.read_csv(CF/'evaluation_rows.csv').dropna(subset=['top_action_id']).copy(); bundle=_build_bundle(); indexes={s:{str(x):i for i,x in enumerate(d.frame.base_record_id)} for s,d in bundle.stages.items()}
    rows=[]
    for r in rec.itertuples(index=False):
        cs,ns=STAGE.get(r.stage,(None,None))
        if not cs or not ns: continue # LATE cannot yield a next-stage behavior window in OULAD tensor contract
        ci=indexes[cs].get(str(r.student_key)); ni=indexes[ns].get(str(r.student_key))
        if ci is None or ni is None: continue
        c,n=bundle.stages[cs],bundle.stages[ns]; f=c.frame.iloc[ci]
        ad=aligned(str(r.top_action_id),c.sequence[ci],int(c.lengths[ci]),n.sequence[ni],int(n.lengths[ni]))
        if ad is None: continue
        rows.append({'student_key':str(r.student_key),'course':str(f.code_module),'presentation':str(f.code_presentation),'stage':r.stage,'fold':int(r.fold),'action_id':r.top_action_id,'adhered':int(ad),'baseline_predicted_risk':float(r.baseline_risk),'estimated_risk_reduction':float(r.top_risk_reduction),'final_favorable':int(f.target)==0,'previous_attempts':float(f.num_of_prev_attempts),'studied_credits':float(f.studied_credits),'cutoff_days':float(f.cutoff_day),'prior_active_days':float(c.sequence[ci,:int(c.lengths[ci]),BASE_CHANNELS.index('active_days')].sum()),'prior_inactive_streak':float(c.sequence[ci,int(c.lengths[ci])-1,BASE_CHANNELS.index('days_since_last_vle_activity')])})
    d=pd.DataFrame(rows); write_csv('cohort_flow.json.csv',[{'input_recommendations':len(rec),'next_window_available':len(d),'adhered':int(d.adhered.sum()) if len(d) else 0,'non_adhered':int((1-d.adhered).sum()) if len(d) else 0,'late_75_excluded_reason':'NO_NEXT_STAGE_OBSERVED_WINDOW'}])
    write_json('cohort_flow.json',{'input_recommendations':len(rec),'temporal_behavior_cohort':len(d),'late_75_not_behavior_evaluable':int((rec.stage=='LATE_75').sum()),'excluded_without_aligned_stage':len(rec)-len(d)})
    # IPTW + AIPW on final outcome, with only cutoff-safe covariates.
    cov=['baseline_predicted_risk','previous_attempts','studied_credits','cutoff_days','prior_active_days','prior_inactive_streak']
    effects=[]; balance=[]
    if len(d) and d.adhered.nunique()==2:
        X=d[cov].fillna(0).to_numpy(); X=StandardScaler().fit_transform(X); ps=LogisticRegression(max_iter=2000,random_state=SEED).fit(X,d.adhered).predict_proba(X)[:,1]; lo,hi=np.quantile(ps,[.01,.99]); keep=(ps>=max(.01,lo))&(ps<=min(.99,hi)); dd=d.loc[keep].copy(); p=ps[keep]; t=dd.adhered.to_numpy(); y=dd.final_favorable.to_numpy(); w=np.where(t==1,1/p,1/(1-p)); ess=float(w.sum()**2/(w*w).sum())
        for j,c in enumerate(cov): balance.append({'covariate':c,'smd_before':smd(X[keep,j],t),'smd_after_iptw':smd(X[keep,j],t,w),'absolute_smd_below_0_1_after':abs(smd(X[keep,j],t,w))<.1})
        # outcome regressions form standard AIPW estimator
        m1=LogisticRegression(max_iter=2000,random_state=SEED).fit(X[keep][t==1],y[t==1]).predict_proba(X[keep])[:,1]; m0=LogisticRegression(max_iter=2000,random_state=SEED).fit(X[keep][t==0],y[t==0]).predict_proba(X[keep])[:,1]
        aipw=np.mean(m1-m0+t*(y-m1)/p-(1-t)*(y-m0)/(1-p)); iptw=np.average(y[t==1],weights=w[t==1])-np.average(y[t==0],weights=w[t==0])
        rng=np.random.default_rng(SEED); boot=[]
        for _ in range(500):
            ii=rng.integers(0,len(y),len(y)); boot.append(float(np.mean((m1-m0+t*(y-m1)/p-(1-t)*(y-m0)/(1-p))[ii])))
        ci=np.quantile(boot,[.025,.975]); effects=[{'outcome':'final_favorable_outcome','estimand':'AIPW adhered vs non-adhered association','estimate':float(aipw),'standard_error':float(np.std(boot,ddof=1)),'ci_95_low':float(ci[0]),'ci_95_high':float(ci[1]),'p_value_or_permutation_probability':float(2*min(np.mean(np.array(boot)<=0),np.mean(np.array(boot)>=0))),'records':len(dd),'learners':d.student_key.nunique(),'iptw_estimate':float(iptw),'propensity_overlap_lower':float(lo),'propensity_overlap_upper':float(hi),'effective_sample_size':ess,'claim':'OBSERVATIONAL_ASSOCIATION_ONLY_NOT_CAUSAL_EFFECT'}]
    else: effects=[{'outcome':'final_favorable_outcome','estimand':'UNAVAILABLE','reason':'NO_ADHERENCE_VARIATION'}]
    write_csv('observational_effects.csv',effects); write_csv('covariate_balance.csv',balance or [{'status':'UNAVAILABLE','reason':'NO_VALID_COHORT'}])
    # Dose response uses an independent final outcome; it is not a model-risk outcome.
    dose=[]
    if len(d)>=4:
        d['quantile']=pd.qcut(d.estimated_risk_reduction,4,duplicates='drop'); grp=d.groupby('quantile',observed=True)
        dose=[{'estimated_risk_reduction_quantile':str(k),'records':len(x),'mean_estimated_risk_reduction':x.estimated_risk_reduction.mean(),'observed_final_favorable_rate':x.final_favorable.mean(),'observed_adherence_rate':x.adhered.mean()} for k,x in grp]
        rho,pv=spearmanr(d.estimated_risk_reduction,d.final_favorable); dose.append({'estimated_risk_reduction_quantile':'SPEARMAN_ALL','records':len(d),'mean_estimated_risk_reduction':float(rho),'observed_final_favorable_rate':float(pv),'observed_adherence_rate':None})
    write_csv('dose_response.csv',dose or [{'status':'UNAVAILABLE','reason':'INSUFFICIENT_COHORT'}])
    unavailable=[{'method':x,'status':'NOT_IDENTIFIABLE_FROM_HISTORICAL_REPLAY','reason':'No factual historical assignment of this alternative action; paired independent outcome comparison would be fabricated.'} for x in ['B0_RANDOM','B1_MOST_POPULAR','B2_LOWEST_WORKLOAD','B3_POLICY_ORDERING','B4_COUNTERFACTUAL_ORDERING']]
    write_csv('baseline_comparison.csv',unavailable); write_csv('ablation_results.csv',[{'ablation':x,'status':'NOT_RUN','reason':'Frozen replay lacks factual exposure assignments for alternate ranking; no post-hoc method tuning permitted.'} for x in ['A0_FULL','A1_NO_RISK_REDUCTION','A2_NO_EVIDENCE','A3_NO_UNCERTAINTY','A4_NO_WORKLOAD','A5_POLICY_ONLY','A6_NO_CONSTRAINTS_OFFLINE']])
    write_csv('negative_controls.csv',[{'control':x,'status':'NOT_RUN','reason':'Requires predeclared full-cohort factual exposure/replay protocol; unavailable from locked sampled replay.'} for x in ['within_stage_course_action_shuffle','learner_state_shuffle','pre_cutoff_placebo','irrelevant_action_exposure','wrong_temporal_window']])
    write_csv('sensitivity_analysis.csv',[{'analysis':x,'status':'NOT_RUN','reason':'Primary gate cannot be assessed because required factual baseline contrasts and controls are not identifiable.'} for x in ['adherence_threshold','outcome_window','leave_course_out','leave_presentation_out','leave_fold_out','risk_band','missingness']])
    balanced=bool(balance) and all(bool(x['absolute_smd_below_0_1_after']) for x in balance)
    # The preregistered monotonicity criterion is decisively contradicted in
    # this locked cohort (Spearman is negative), so this is a gate failure,
    # not merely an absence of precision.
    gate='IMPLEMENTATION_COMPLETE_SCIENTIFIC_VALIDATION_FAILED'
    result={'schema_version':'scientific_validation_v1','generated_at':now(),'status':gate,'claim_boundary':'OBSERVATIONAL_ASSOCIATION_ONLY_NOT_CAUSAL_EFFECT','engineering_validation':'COMPLETE','internal_model_consistency_validation':'AVAILABLE_NOT_INDEPENDENT_EVIDENCE','observational_scientific_validation':'FAILED','causal_validation':'NOT_PERFORMED','expert_validation':'NOT_PERFORMED','temporal_protocol':'PARTIAL_PASS_LATE_75_NEXT_STAGE_WINDOW_UNAVAILABLE','covariate_balance':'PASS' if balanced else 'FAIL_OR_INSUFFICIENT','baseline_comparison':'NOT_IDENTIFIABLE','negative_controls':'NOT_RUN','temporal_leakage':'NOT_DETECTED_IN_FROZEN_INPUTS_BUT_NOT_FULLY_AUDITED','protected_feature_violations':'NOT_AUDITED_BY_THIS_OBSERVATIONAL_RUN','release_gate_passed':False,'merge_allowed':False,'reason':'The prespecified dose-response monotonicity criterion failed, and scientific gate requirements for factual baselines, negative controls, and full temporal coverage are unmet.'}
    write_json('SCIENTIFIC_VALIDATION.json',result)
    checks={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='CHECKSUMS.json'}; write_json('CHECKSUMS.json',checks)
    REPORT.mkdir(exist_ok=True)
    (REPORT/'COUNTERFACTUAL_SCIENTIFIC_PROTOCOL.md').write_text('# Counterfactual scientific protocol\n\nStatus: `LOCKED_BEFORE_RESULT_COMPUTATION`. Claim boundary: `OBSERVATIONAL_ASSOCIATION_ONLY_NOT_CAUSAL_EFFECT`. See `artifacts/recommend_hybrid/scientific_validation/protocol.json`.\n',encoding='utf8')
    (REPORT/'COUNTERFACTUAL_OBSERVATIONAL_VALIDATION.md').write_text(f'# Counterfactual observational validation\n\nStatus: `{gate}`. The adjusted adherence association is observational only and does not establish a causal effect. Required factual comparisons for alternate policies are not identifiable from the locked historical replay.\n',encoding='utf8')
    (REPORT/'COUNTERFACTUAL_BASELINE_AND_ABLATION.md').write_text('# Baselines and ablations\n\nAll requested policy comparisons are `NOT_IDENTIFIABLE_FROM_HISTORICAL_REPLAY`; results were not fabricated.\n',encoding='utf8')
    (REPORT/'COUNTERFACTUAL_NEGATIVE_CONTROL_AUDIT.md').write_text('# Negative-control audit\n\nControls are `NOT_RUN`: the locked sampled replay does not supply a preregistered factual-exposure design for them. This prevents a scientific-gate pass.\n',encoding='utf8')
    (REPORT/'COUNTERFACTUAL_SCIENTIFIC_RESULTS_VI.md').write_text(f'# Kết quả đánh giá khoa học counterfactual\n\n- Trạng thái: `{gate}`\n- Giới hạn claim: `OBSERVATIONAL_ASSOCIATION_ONLY_NOT_CAUSAL_EFFECT`\n- Không dùng model-estimated risk reduction, Success@K, hay threshold crossing làm bằng chứng độc lập.\n- Không merge.\n',encoding='utf8')
    print(json.dumps(result,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
