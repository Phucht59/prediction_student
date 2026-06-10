import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class HybridLoss(nn.Module):
    def __init__(self, class_weights=None, focal_gamma=2.0, ordinal_lambda=0.1, label_smoothing=0.0):
        super().__init__()
        self.ordinal_lambda = ordinal_lambda
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        
        if focal_gamma > 0:
            self.classification_loss = FocalLoss(weight=class_weights, gamma=focal_gamma)
            self.use_focal = True
        else:
            self.classification_loss = nn.CrossEntropyLoss(
                weight=class_weights, 
                label_smoothing=label_smoothing
            )
            self.use_focal = False
            
    def forward(self, logits, ordinal_score, y_class, y_ordinal):
        if self.use_focal and self.label_smoothing > 0:
            # PyTorch's cross_entropy supports label smoothing, but our custom FocalLoss might not easily.
            # We can use PyTorch cross_entropy with label smoothing to get CE loss, then apply focal weight.
            ce_loss = F.cross_entropy(logits, y_class, weight=self.class_weights, label_smoothing=self.label_smoothing, reduction='none')
            pt = torch.exp(-F.cross_entropy(logits, y_class, weight=self.class_weights, reduction='none'))
            cls_loss = (((1 - pt) ** self.classification_loss.gamma) * ce_loss).mean()
        else:
            cls_loss = self.classification_loss(logits, y_class)
            
        ord_loss = F.smooth_l1_loss(ordinal_score, y_ordinal)
        
        return cls_loss + self.ordinal_lambda * ord_loss, cls_loss, ord_loss
