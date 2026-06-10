import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class V26HybridLoss(nn.Module):
    def __init__(self, class_weights=None, focal_gamma=2.0, ordinal_lambda=0.1, label_smoothing=0.05, use_focal=True):
        super().__init__()
        self.ordinal_lambda = ordinal_lambda
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.use_focal = use_focal
        
        if self.use_focal:
            self.focal = FocalLoss(weight=class_weights, gamma=focal_gamma)
            
    def forward(self, logits, ordinal_score, y_class, y_ordinal):
        if self.use_focal:
            if self.label_smoothing > 0:
                ce_loss = F.cross_entropy(logits, y_class, weight=self.class_weights, label_smoothing=self.label_smoothing, reduction='none')
                pt = torch.exp(-F.cross_entropy(logits, y_class, weight=self.class_weights, reduction='none'))
                cls_loss = (((1 - pt) ** self.focal.gamma) * ce_loss).mean()
            else:
                cls_loss = self.focal(logits, y_class)
        else:
            cls_loss = F.cross_entropy(logits, y_class, weight=self.class_weights, label_smoothing=self.label_smoothing)
            
        if self.ordinal_lambda > 0:
            ord_loss = F.smooth_l1_loss(ordinal_score, y_ordinal)
            return cls_loss + self.ordinal_lambda * ord_loss, cls_loss, ord_loss
        else:
            return cls_loss, cls_loss, torch.tensor(0.0, device=logits.device)
