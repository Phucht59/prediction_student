"""Verify frozen Hybrid checkpoint and intervention-stage mapping authority."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
SEEDS=(42,1201,2026,3407,7319)
STAGES=("EARLY_20","EARLY_35","MIDDLE_50","LATE_75")
ARCH="df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e"
PARAMS=160492
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    mp=ROOT/'artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json'
    m=json.loads(mp.read_text(encoding='utf-8'))
    rows=[r for r in m['checkpoints'] if r.get('usage')=='INTERVENTION_STAGE_SHARED']
    all_rows=list(m['checkpoints'])
    missing=[]; lfs=[]; sha_bad=[]; arch_bad=[]; param_bad=[]; loads=[]; paths={}
    for r in all_rows:
        p=ROOT/r['provenance']['source_checkpoint_path']; paths[str(p)]=p
        if not p.exists(): missing.append(str(p)); continue
        with p.open('rb') as f: head=f.read(200)
        if head.startswith(b'version https://git-lfs.github.com/spec'): lfs.append(str(p)); continue
        if sha(p)!=r['sha256']: sha_bad.append(str(p))
        try:
            payload=torch.load(p,map_location='cpu',weights_only=False)
            if payload.get('architecture_hash')!=ARCH: arch_bad.append(str(p))
            if int(payload.get('parameter_count',-1))!=PARAMS: param_bad.append(str(p))
        except Exception as exc: loads.append({'path':str(p),'error':type(exc).__name__})
    mappings={(s,int(r['outer_fold']),int(r['seed'])) for s in STAGES for r in rows if int(r['seed']) in SEEDS}
    audit={'manifest_status':m.get('status'),'checkpoint_set_status':m.get('checkpoint_set_status'),'physical_checkpoint_count':len(paths),'stage_fold_seed_mapping_count':len(mappings),'verified_mapping_count':len(mappings)-len(missing)-len(lfs)-len(sha_bad)-len(arch_bad)-len(param_bad)-len(loads),'missing_paths':missing,'lfs_pointer_paths':lfs,'sha_mismatches':sha_bad,'architecture_mismatches':arch_bad,'parameter_count_mismatches':param_bad,'load_failures':loads,'authority_status':'PASS' if not any((missing,lfs,sha_bad,arch_bad,param_bad,loads)) and len(paths)==30 and len(mappings)==60 else 'BLOCKED'}
    out=ROOT/'artifacts/recommend_hybrid/explainable_v2/run_state/HYBRID_OOF_AUTHORITY_AUDIT.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(audit,indent=2)); return 0 if audit['authority_status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
