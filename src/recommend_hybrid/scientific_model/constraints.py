import numpy as np
def constrain(prob, frame):
 out=prob.copy()
 unsupported=~frame.apply(lambda r:r.dataset in r.action_datasets and r.stage in r.action_stages,axis=1).to_numpy()
 gap=frame.action_id.isin(["ASSESSMENT_COMPLETION","ATTENDANCE_IMPROVEMENT"]).to_numpy(); review=frame.human_review_required.to_numpy(bool)
 out[unsupported]=[1.,0.,0.]
 for mask in (gap,review):
  excess=out[mask,2].copy(); out[mask,1]+=excess; out[mask,2]=0
 return out,unsupported,gap,review
