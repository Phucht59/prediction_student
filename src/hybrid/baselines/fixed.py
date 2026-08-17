"""Fixed strong Phase 2 baselines; no parameter search or outer-test access."""
from __future__ import annotations
import gc, os, tempfile, time
from pathlib import Path
import numpy as np
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(tempfile.gettempdir()) / "torchinductor_hybrid_phase2"))
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

WORKERS=max(1,min(8,(os.cpu_count() or 2)-1))

def make_fixed_estimator(family, y_train, seed=42):
    if family=='logistic_regression': return LogisticRegression(solver='liblinear',C=1.,penalty='l2',class_weight='balanced',max_iter=2000,random_state=seed)
    if family=='svm': return LinearSVC(C=1.,class_weight='balanced',random_state=seed,max_iter=5000)
    if family=='random_forest': return RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',bootstrap=True,n_jobs=WORKERS,random_state=seed)
    if family=='xgboost':
        pos=max(1,int(np.sum(y_train))); neg=len(y_train)-pos
        return XGBClassifier(objective='binary:logistic',eval_metric='logloss',device='cuda',tree_method='hist',n_estimators=300,max_depth=6,learning_rate=.05,min_child_weight=1.,subsample=.8,colsample_bytree=.8,reg_alpha=0.,reg_lambda=1.,gamma=0.,scale_pos_weight=neg/pos,random_state=seed,n_jobs=WORKERS)
    if family=='catboost': return CatBoostClassifier(iterations=300,depth=6,learning_rate=.05,loss_function='Logloss',auto_class_weights='Balanced',task_type='GPU',devices='0',border_count=128,random_seed=seed,verbose=False,allow_writing_files=False)
    raise ValueError(family)

class TorchMLP(torch.nn.Module):
    def __init__(self, width):
        super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(width,128),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Dropout(.15),torch.nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(1)

def split_mlp_fit_early_stop(train_frame, seed=42):
    """Group-safe early-stop holdout drawn solely from the current inner-train."""
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    y=train_frame.target.to_numpy(); groups=train_frame.group_id.astype(str).to_numpy()
    for fit_idx, stop_idx in splitter.split(train_frame,y,groups):
        if len(np.unique(y[fit_idx]))==2 and len(np.unique(y[stop_idx]))==2:
            if set(groups[fit_idx]) & set(groups[stop_idx]): raise RuntimeError('MLP early-stop group leakage')
            return fit_idx,stop_idx
    raise RuntimeError('Unable to create two-class group-safe MLP early-stop holdout')

def fit_score_torch_mlp(X_train,y_train,X_stop,y_stop,X_valid,seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); device=torch.device('cuda:0'); torch.cuda.empty_cache()
    model=TorchMLP(X_train.shape[1]).to(device); pos=max(1,int(np.sum(y_train))); weight=torch.tensor([(len(y_train)-pos)/pos],device=device)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=weight); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); scaler=torch.amp.GradScaler('cuda')
    xt=torch.as_tensor(X_train,dtype=torch.float32); yt=torch.as_tensor(y_train,dtype=torch.float32); xs=torch.as_tensor(X_stop,dtype=torch.float32,device=device); xv=torch.as_tensor(X_valid,dtype=torch.float32,device=device)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt,yt),batch_size=512,shuffle=True,generator=torch.Generator().manual_seed(seed)); best=-1.; best_state=None; stale=0
    for _ in range(100):
        model.train()
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16): loss=loss_fn(model(xb),yb)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        model.eval()
        with torch.no_grad(): stop_score=torch.sigmoid(model(xs)).float().cpu().numpy()
        ap=average_precision_score(y_stop,stop_score)
        if ap>best: best=ap; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=10: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): score=torch.sigmoid(model(xv)).float().cpu().numpy()
    del model,xt,yt,xs,xv,loader; gc.collect(); torch.cuda.empty_cache(); return score
