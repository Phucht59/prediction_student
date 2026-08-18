from pathlib import Path
import json, subprocess, sys, time
ROOT=Path(__file__).resolve().parent; R=ROOT/'runtime';R.mkdir(exist_ok=True); seeds=(42,1201,2026,3407,7319); jobs=[(f,s) for f in range(3) for s in seeds]
def write(**x):
 p=R/'OULAD_NONE_STATUS.json'; old=json.loads(p.read_text()) if p.is_file() else {'total_expected':15,'completed':0,'failed':0};old.update(x,last_update_at=time.time());p.write_text(json.dumps(old,indent=2))
write(status='RUNNING');(R/'OULAD_NONE_RUNNING').write_text('RUNNING')
try:
 for i,(f,s) in enumerate(jobs):
  d=ROOT/'oulad_none_runs'/f'oulad__fixed_none__fold{f}__seed{s}';m=d/'run_manifest.json'
  if m.is_file() and json.loads(m.read_text()).get('status')=='COMPLETE':write(completed=i+1);continue
  write(current_run=f'oulad__fixed_none__fold{f}__seed{s}',completed=i)
  subprocess.run([sys.executable,str(ROOT/'run_oulad_none.py'),'--fold',str(f),'--seed',str(s)],check=True)
  write(completed=i+1,current_run=None)
 write(status='COMPLETE');(R/'OULAD_NONE_COMPLETE').write_text('COMPLETE')
except Exception as e:
 write(status='FAILED',error=repr(e));(R/'OULAD_NONE_FAILED').write_text(repr(e));raise
finally:(R/'OULAD_NONE_RUNNING').unlink(missing_ok=True)
