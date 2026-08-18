from __future__ import annotations
import numpy as np
from src.hybrid.contracts import HybridDataView
from src.database.service import load_student_timeline

def load_hybrid_view(enrollment_ids, channels: list[str], targets=None) -> HybridDataView:
    rows = [load_student_timeline(e) for e in enrollment_ids]
    max_t = max((len(x) for x in rows), default=0); temporal=np.zeros((len(rows),max_t,len(channels)),dtype=np.float32); mask=np.zeros((len(rows),max_t),dtype=bool)
    for i, obs in enumerate(rows):
        for j, row in enumerate(obs):
            vals=row["features"] or {}; mask[i,j]=True
            for k,name in enumerate(channels): temporal[i,j,k]=float(vals.get(name,0.0))
    view=HybridDataView(record_id=np.asarray([str(x) for x in enrollment_ids]),group_id=np.asarray([str(x) for x in enrollment_ids]),target=np.asarray(targets if targets is not None else np.zeros(len(rows),dtype=np.int64)),temporal=temporal,mask=mask,lengths=mask.sum(axis=1).astype(np.int64),metadata={"source":"POSTGRES"})
    view.validate(); return view
