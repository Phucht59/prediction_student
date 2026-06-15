import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from typing import Any

def evaluate_risk_diagnosis(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """
    Calculate F1, Precision, Recall, and Hamming Loss for multi-label Risk Diagnosis.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred))
    }

def evaluate_ranking(
    recommendations_list: list[list[dict[str, Any]]],
    actual_risks_list: list[list[str]],
    catalog_df: pd.DataFrame,
    k: int = 3
) -> dict[str, float]:
    """
    Calculate Precision@K, Recall@K, NDCG@K, and Catalog Coverage.
    """
    catalog_items = catalog_df.to_dict("records")
    total_catalog_size = len(catalog_df)
    
    p_at_k = []
    r_at_k = []
    ndcg_at_k = []
    recommended_unique = set()
    
    for recs, actual_risks in zip(recommendations_list, actual_risks_list):
        actual_risks_set = set(actual_risks)
        
        # Determine relevant items for this student based on actual risks
        relevant_items = set()
        for item in catalog_items:
            target_risks_str = str(item.get("target_risks", ""))
            item_risks = [r.strip() for r in target_risks_str.split(",") if r.strip()]
            if any(r in actual_risks_set for r in item_risks):
                relevant_items.add(item["item_id"])
            elif not actual_risks_set and item["item_id"] == "advanced_seminar":
                # If student is stable, the advanced seminar is relevant
                relevant_items.add("advanced_seminar")
                
        # Recommended at K
        top_k_recs = [item["item_id"] for item in recs[:k]]
        for item_id in top_k_recs:
            recommended_unique.add(item_id)
            
        # Compute Precision@K
        hits = len(set(top_k_recs) & relevant_items)
        p_at_k.append(hits / k)
        
        # Compute Recall@K
        if len(relevant_items) > 0:
            r_at_k.append(hits / len(relevant_items))
        else:
            r_at_k.append(1.0)
            
        # Compute NDCG@K
        dcg = 0.0
        for rank_idx, item_id in enumerate(top_k_recs):
            if item_id in relevant_items:
                dcg += 1.0 / np.log2(rank_idx + 2)
                
        idcg = 0.0
        for rank_idx in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(rank_idx + 2)
            
        ndcg = dcg / idcg if idcg > 0.0 else 1.0
        ndcg_at_k.append(ndcg)
        
    coverage = len(recommended_unique) / total_catalog_size if total_catalog_size > 0 else 0.0
    return {
        f"precision_at_{k}": float(np.mean(p_at_k)),
        f"recall_at_{k}": float(np.mean(r_at_k)),
        f"ndcg_at_{k}": float(np.mean(ndcg_at_k)),
        f"coverage_at_{k}": float(coverage)
    }
