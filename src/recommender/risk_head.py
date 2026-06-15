import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RiskDiagnosisHead(nn.Module):
    """
    3-layer MLP predicting 6 academic risks.
    Input: student features (normalized) concatenated with class probabilities.
    """
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, output_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RiskDiagnosisModel:
    def __init__(self, model: RiskDiagnosisHead, feature_mean: np.ndarray, feature_scale: np.ndarray):
        self.model = model
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        
    def predict_logits(self, features: np.ndarray, class_probs: np.ndarray, device: str = "cpu") -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        X = np.hstack([normalized, class_probs])
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32).to(device))
            
    def predict_proba(self, features: np.ndarray, class_probs: np.ndarray, device: str = "cpu") -> np.ndarray:
        logits = self.predict_logits(features, class_probs, device)
        probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

def train_risk_head(
    features: np.ndarray,
    class_probs: np.ndarray,
    targets: np.ndarray,
    epochs: int = 300,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    device: str = "cpu"
) -> RiskDiagnosisModel:
    # 1. Normalize features
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    normalized_features = (features - feature_mean) / feature_scale
    
    # 2. Concatenate normalized features and class probabilities
    X = np.hstack([normalized_features, class_probs])
    
    # Convert to Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
    
    # 3. Calculate pos_weight for BCEWithLogitsLoss
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    pos_weight = np.clip(negatives / np.clip(positives, 1.0, None), 0.5, 10.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    
    # 4. Instantiate Model
    input_dim = X.shape[1]
    model = RiskDiagnosisHead(input_dim=input_dim, output_dim=targets.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # 5. Train loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()
        
    return RiskDiagnosisModel(model, feature_mean, feature_scale)
