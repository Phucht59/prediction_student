import torch
import torch.optim as optim
import numpy as np
import copy
from .losses import V26HybridLoss
from .evaluate import compute_metrics, evaluate_model

def compute_class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    weights = np.zeros(n_classes, dtype=np.float32)
    total = len(y)
    for i in range(n_classes):
        if counts[i] > 0:
            weights[i] = total / (n_classes * counts[i])
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights / weight_sum * n_classes
    return torch.tensor(weights, dtype=torch.float32)

def train_v26_model(
    model, train_loader, val_loader, config, 
    class_weights=None, use_ordinal=True, use_focal=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    
    for epoch in range(config.max_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        # Two-stage check
        is_pretrain = epoch < config.pretrain_epochs
        current_use_focal = False if is_pretrain else use_focal
        current_lambda = 0.0 if is_pretrain else (config.ordinal_lambda if use_ordinal else 0.0)
        
        criterion = V26HybridLoss(
            class_weights=class_weights if current_use_focal else None,
            focal_gamma=config.focal_gamma,
            ordinal_lambda=current_lambda,
            label_smoothing=config.label_smoothing,
            use_focal=current_use_focal
        )
        
        for seq_x, num_x, cat_x, y_cls, y_ord in train_loader:
            seq_x, num_x, cat_x = seq_x.to(device), num_x.to(device), cat_x.to(device)
            y_cls, y_ord = y_cls.to(device), y_ord.to(device)
            
            optimizer.zero_grad()
            logits, ord_score = model(seq_x, num_x, cat_x)
            loss, _, _ = criterion(logits, ord_score, y_cls, y_ord)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * seq_x.size(0)
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(y_cls.cpu().numpy())
            
        train_loss /= len(train_loader.dataset)
        train_metrics = compute_metrics(train_targets, train_preds)
        
        # Val
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for seq_x, num_x, cat_x, y_cls, y_ord in val_loader:
                seq_x, num_x, cat_x = seq_x.to(device), num_x.to(device), cat_x.to(device)
                y_cls, y_ord = y_cls.to(device), y_ord.to(device)
                logits, ord_score = model(seq_x, num_x, cat_x)
                loss, _, _ = criterion(logits, ord_score, y_cls, y_ord)
                val_loss += loss.item() * seq_x.size(0)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(y_cls.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_metrics = compute_metrics(val_targets, val_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_f1'].append(val_metrics['f1_macro'])
        
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config.patience and epoch >= config.pretrain_epochs:
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
