import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data_pipeline import StudentDataset
from src.models_v27 import AttentionPooling1D, GatedFusion, StudentHybridV27
from src.losses_v27 import FocalLoss, ClassBalancedFocalLoss, OrdinalLoss, JointHybridLoss


def test_student_dataset_returns_six_elements_with_g3_raw_preserved():
    # Test StudentDataset with student kind and G3_raw present
    df_student = pd.DataFrame({
        "G1": [10.0, 12.0],
        "G2": [11.0, 13.0],
        "absences": [2.0, 4.0],
        "studytime": [2.0, 3.0],
        "target": [1, 2],
        "G3_raw": [11.5, 14.0]
    })
    
    dataset = StudentDataset(
        df=df_student,
        kind="student",
        target_col="target",
        numerical_cols=["absences", "studytime", "G3_raw"],
        categorical_cols=[]
    )
    
    assert len(dataset) == 2
    # Check that item unpacking yields 6 elements
    seq_x, num_x, cat_x, label, idx, reg_val = dataset[0]
    assert isinstance(seq_x, torch.Tensor)
    assert isinstance(num_x, torch.Tensor)
    assert isinstance(cat_x, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert isinstance(idx, int)
    assert isinstance(reg_val, torch.Tensor)
    
    # Check values
    assert reg_val.item() == 11.5
    assert label.item() == 1
    
    # Test with xapi kind (which should default reg_label to 0.0)
    df_xapi = pd.DataFrame({
        "raisedhands": [50.0, 60.0],
        "VisITedResources": [70.0, 80.0],
        "target": [0, 1]
    })
    dataset_xapi = StudentDataset(
        df=df_xapi,
        kind="xapi",
        target_col="target",
        numerical_cols=["raisedhands", "VisITedResources"],
        categorical_cols=[]
    )
    _, _, _, _, _, reg_val_xapi = dataset_xapi[0]
    assert reg_val_xapi.item() == 0.0


def test_attention_pooling_computes_valid_shapes():
    batch_size = 4
    seq_len = 5
    hidden_dim = 16
    
    pool = AttentionPooling1D(hidden_dim)
    seq_input = torch.randn(batch_size, seq_len, hidden_dim)
    
    pooled, weights = pool(seq_input)
    assert pooled.shape == (batch_size, hidden_dim)
    assert weights.shape == (batch_size, seq_len, 1)
    
    # Check attention weights sum to 1 over the sequence dimension
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size, 1), atol=1e-5)


def test_gated_fusion_computes_dynamic_blended_output():
    batch_size = 4
    seq_dim = 16
    ctx_dim = 8
    out_dim = 12
    
    fusion = GatedFusion(seq_dim=seq_dim, ctx_dim=ctx_dim, out_dim=out_dim)
    seq_vec = torch.randn(batch_size, seq_dim)
    ctx_vec = torch.randn(batch_size, ctx_dim)
    
    fused = fusion(seq_vec, ctx_vec)
    assert fused.shape == (batch_size, out_dim)


def test_student_hybrid_v27_forward_outputs_three_heads():
    batch_size = 8
    num_classes = 3
    num_numerical = 4
    cat_cardinalities = [2, 3]
    
    model = StudentHybridV27(
        num_classes=num_classes,
        seq_in_channels=1,
        num_numerical=num_numerical,
        cat_cardinalities=cat_cardinalities,
        cnn_channels=16,
        lstm_hidden_dim=32,
        context_hidden_dim=24,
        fusion_hidden_dim=16
    )
    
    seq_x = torch.randn(batch_size, 2, 1) # seq_len=2, channels=1
    num_x = torch.randn(batch_size, num_numerical)
    cat_x = torch.randint(0, 2, (batch_size, len(cat_cardinalities)))
    
    class_logits, ordinal_logits, reg_logits = model(seq_x, num_x, cat_x)
    
    assert class_logits.shape == (batch_size, num_classes)
    assert ordinal_logits.shape == (batch_size, num_classes - 1)
    assert reg_logits.shape == (batch_size,)


def test_losses_v27_focal_and_ordinal():
    batch_size = 4
    num_classes = 3
    
    class_logits = torch.randn(batch_size, num_classes)
    ordinal_logits = torch.randn(batch_size, num_classes - 1)
    reg_logits = torch.randn(batch_size)
    
    targets_class = torch.randint(0, num_classes, (batch_size,))
    targets_reg = torch.randn(batch_size)
    
    # 1. Focal Loss
    fl = FocalLoss(gamma=2.0)
    loss_fl = fl(class_logits, targets_class)
    assert loss_fl.item() >= 0.0
    
    # 2. Class Balanced Focal Loss
    cbfl = ClassBalancedFocalLoss(class_counts=[100, 50, 20], beta=0.99, gamma=2.0)
    loss_cbfl = cbfl(class_logits, targets_class)
    assert loss_cbfl.item() >= 0.0
    
    # 3. Ordinal Loss
    ol = OrdinalLoss()
    loss_ol = ol(ordinal_logits, targets_class)
    assert loss_ol.item() >= 0.0
    
    # 4. Joint Hybrid Loss
    joint_loss = JointHybridLoss(
        class_loss_fn=cbfl,
        ordinal_loss_fn=ol,
        regression_loss_fn=nn.MSELoss(),
        w_class=1.0,
        w_ord=1.0,
        w_reg=1.0
    )
    loss_joint = joint_loss((class_logits, ordinal_logits, reg_logits), targets_class, targets_reg)
    assert loss_joint.item() >= 0.0
