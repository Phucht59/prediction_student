import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for classification, addressing class imbalance by down-weighting easy examples."""

    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    """Class Balanced Focal Loss, weighting the Focal Loss based on the effective number of samples."""

    def __init__(self, class_counts, beta=0.99, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # Calculate class-balanced weights based on effective sample size
        counts = torch.tensor(class_counts, dtype=torch.float32)
        effective_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta + 1e-8)
        weights = 1.0 / effective_num
        # Normalize weights so they sum to the number of classes
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer('weights', weights)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class OrdinalLoss(nn.Module):
    """Ordinal Classification Loss, mapping targets to binary thresholds and optimizing with BCE."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, inputs, targets):
        # inputs shape: (batch_size, num_classes - 1)
        # targets shape: (batch_size) containing class labels in range [0, num_classes - 1]
        num_classes_minus_1 = inputs.shape[1]
        batch_size = targets.shape[0]
        device = inputs.device
        
        # Construct binary target matrix where binary_targets[i, j] is 1 if target > j, else 0
        binary_targets = torch.zeros((batch_size, num_classes_minus_1), device=device)
        for j in range(num_classes_minus_1):
            binary_targets[:, j] = (targets > j).float()
            
        return self.bce(inputs, binary_targets)


class JointHybridLoss(nn.Module):
    """Joint Hybrid Loss combining classification loss, ordinal classification loss, and regression loss."""

    def __init__(self, class_loss_fn, ordinal_loss_fn, regression_loss_fn=None, w_class=1.0, w_ord=1.0, w_reg=1.0):
        super().__init__()
        self.class_loss_fn = class_loss_fn
        self.ordinal_loss_fn = ordinal_loss_fn
        self.regression_loss_fn = regression_loss_fn or nn.MSELoss()
        self.w_class = w_class
        self.w_ord = w_ord
        self.w_reg = w_reg

    def forward(self, outputs, target_class, target_reg):
        """
        outputs: tuple of (class_logits, ordinal_logits, reg_logits)
        target_class: classification class labels
        target_reg: continuous regression targets
        """
        class_logits, ordinal_logits, reg_logits = outputs
        
        loss_class = self.class_loss_fn(class_logits, target_class)
        loss_ord = self.ordinal_loss_fn(ordinal_logits, target_class)
        loss_reg = self.regression_loss_fn(reg_logits, target_reg)
        
        total_loss = (self.w_class * loss_class +
                      self.w_ord * loss_ord +
                      self.w_reg * loss_reg)
        return total_loss
