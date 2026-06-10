import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from .losses import HybridLoss
from .metrics import compute_metrics

def compute_class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    # Inverse frequency weighting
    weights = np.zeros(n_classes, dtype=np.float32)
    total = len(y)
    for i in range(n_classes):
        if counts[i] > 0:
            weights[i] = total / (n_classes * counts[i])
        else:
            weights[i] = 0.0
            
    # Normalize weights so they sum to n_classes
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights / weight_sum * n_classes
        
    return torch.tensor(weights, dtype=torch.float32)

def train_hybrid_model(
    model,
    train_loader,
    val_loader,
    config,
    class_weights=None,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    model = model.to(device)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        
    criterion = HybridLoss(
        class_weights=class_weights,
        focal_gamma=config.focal_gamma,
        ordinal_lambda=config.ordinal_lambda,
        label_smoothing=config.label_smoothing
    )
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Cosine Annealing scheduler helps with deep models
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    
    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }
    
    for epoch in range(config.max_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for seq_x, num_x, cat_x, y_cls, y_ord in train_loader:
            seq_x, num_x, cat_x = seq_x.to(device), num_x.to(device), cat_x.to(device)
            y_cls, y_ord = y_cls.to(device), y_ord.to(device)
            
            optimizer.zero_grad()
            logits, ord_score = model(seq_x, num_x, cat_x)
            
            loss, _, _ = criterion(logits, ord_score, y_cls, y_ord)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping for stability
            optimizer.step()
            
            train_loss += loss.item() * seq_x.size(0)
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(y_cls.cpu().numpy())
            
        scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        train_metrics = compute_metrics(train_targets, train_preds)
        
        # Validation
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
            
        if epochs_no_improve >= config.patience:
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, history

def evaluate_model(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets_cls = []
    all_preds_ord = []
    all_targets_ord = []
    all_probs = []
    
    with torch.no_grad():
        for seq_x, num_x, cat_x, y_cls, y_ord in dataloader:
            seq_x, num_x, cat_x = seq_x.to(device), num_x.to(device), cat_x.to(device)
            
            logits, ord_score = model(seq_x, num_x, cat_x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets_cls.extend(y_cls.numpy())
            all_preds_ord.extend(ord_score.cpu().numpy())
            all_targets_ord.extend(y_ord.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    metrics = compute_metrics(all_targets_cls, all_preds, all_targets_ord, all_preds_ord)
    
    return metrics, all_targets_cls, all_preds, all_probs
