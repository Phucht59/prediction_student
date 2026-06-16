import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Setup path to import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS, MODELS_DIR, FIXED_SEEDS
from src.data_pipeline import (
    apply_feature_engineering,
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
    get_sequence_columns,
)
from scripts.run_pipeline import load_or_create_splits
from src.models import create_model
from src.recommender.risk_rules import generate_weak_labels
from src.recommender.candidate_generator import CandidateGenerator
from src.recommender.risk_head import train_risk_head
from src.recommender.knowledge_base import load_knowledge_base
from src.recommender.hybrid_scorer import HybridScorer
from src.recommender.path_planner import PathPlanner
from src.evaluation.recommender_eval import evaluate_risk_diagnosis, evaluate_ranking, evaluate_path_quality
from src.recommendation import extract_features
from src.utils import set_seed, setup_logger

logger = setup_logger("run_recommender_pipeline")

def infer_model_params_from_state_dict(best_params: dict, state_dict: dict) -> dict:
    inferred = dict(best_params)
    if "sequence_cnn.0.weight" in state_dict:
        inferred["cnn_channels"] = int(state_dict["sequence_cnn.0.weight"].shape[0])
    if "sequence_bilstm.weight_hh_l0" in state_dict:
        inferred["lstm_hidden_dim"] = int(state_dict["sequence_bilstm.weight_hh_l0"].shape[1])
    if "context_mlp.0.weight" in state_dict:
        inferred["context_hidden_dim"] = int(state_dict["context_mlp.0.weight"].shape[0])
    if "fusion.0.weight" in state_dict:
        inferred["fusion_hidden_dim"] = int(state_dict["fusion.0.weight"].shape[0])
    embedding_dims = {
        int(value.shape[1])
        for key, value in state_dict.items()
        if key.startswith("embeddings.") and key.endswith(".weight") and len(value.shape) == 2
    }
    if len(embedding_dims) == 1:
        inferred["embedding_dim"] = embedding_dims.pop()
    return inferred

def get_ensemble_probabilities(dataset_name: str, target_df: pd.DataFrame, train_pool: pd.DataFrame, locked_test: pd.DataFrame, best_params: dict, device: torch.device) -> np.ndarray:
    """
    Generate ensemble class probabilities on target_df by running all 11 seed models.
    """
    spec = DATASETS[dataset_name]
    batch_size = int(best_params["batch_size"])
    all_probs = []
    
    for seed in FIXED_SEEDS:
        set_seed(seed)
        labels = train_pool[spec.target_col].astype(int).to_numpy()
        indices = np.arange(len(train_pool))
        train_indices, val_indices = train_test_split_indices(
            indices,
            test_size=0.15,
            stratify=labels,
            random_state=seed,
        )
        train_sub = apply_feature_engineering(train_pool.iloc[train_indices].copy(), spec.kind)
        
        preprocessor = DataPreprocessor(
            target_col=spec.target_col,
            oversample_method=best_params["oversample_method"],
            smote_ratio=best_params.get("smote_ratio", 1.0),
            resampling_k_neighbors=best_params.get("resampling_k_neighbors", 5),
        )
        train_prep = preprocessor.fit_transform(train_sub)
        
        # Check if saved ensemble features metadata exists
        features_json_path = MODELS_DIR / f"{dataset_name}_3class_ensemble_features.json"
        
        if features_json_path.exists():
            ensemble_features = json.loads(features_json_path.read_text(encoding="utf-8"))
            seed_feat = ensemble_features[str(seed)]
            num_cols = seed_feat["num_cols"]
            cat_cols = seed_feat["cat_cols"]
            cat_cardinalities = seed_feat["cat_cardinalities"]
            num_numerical = len(num_cols)
            
            target_engineered = apply_feature_engineering(target_df.copy(), spec.kind)
            target_prep = preprocessor.transform(target_engineered)
            
            seq_cols = get_sequence_columns(spec.kind)
            cols_to_keep = [col for col in (num_cols + cat_cols + seq_cols) if col in target_prep.columns]
            if spec.target_col in target_prep.columns and spec.target_col not in cols_to_keep:
                cols_to_keep.append(spec.target_col)
            target_selected = target_prep[cols_to_keep]
        else:
            selector = FeatureSelector(
                target_col=spec.target_col,
                use_feature_selection=True,
                required_features=get_sequence_columns(spec.kind),
            )
            _ = selector.fit_transform(
                train_prep,
                preprocessor.numerical_cols,
                preprocessor.categorical_cols,
            )
            
            # Transform the target dataframe
            target_engineered = apply_feature_engineering(target_df.copy(), spec.kind)
            target_prep = preprocessor.transform(target_engineered)
            target_selected = selector.transform(target_prep)
            
        target_ds = StudentDataset(
            target_selected,
            spec.kind,
            spec.target_col,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols
        )
        target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=False)
        
        if not features_json_path.exists():
            cat_cardinalities = [len(preprocessor.label_encoders[col].classes_) for col in target_ds.cat_cols]
            num_numerical = len(target_ds.num_cols)
        
        model_path = MODELS_DIR / f"{dataset_name}_3class_cnn_bilstm_mlp_seed{seed}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Ensemble checkpoint not found at: {model_path}")

        state_dict = torch.load(model_path, map_location=device)
        checkpoint_params = infer_model_params_from_state_dict(best_params, state_dict)
        model = create_model(spec.kind, checkpoint_params, num_numerical, cat_cardinalities).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        seed_probabilities = []
        with torch.no_grad():
            for batch in target_loader:
                seq_x, num_x, cat_x, _, _ = batch[:5]
                probabilities = model.predict_proba(
                    seq_x.to(device),
                    num_x.to(device),
                    cat_x.to(device),
                )
                seed_probabilities.extend(probabilities.cpu().numpy())
        all_probs.append(np.asarray(seed_probabilities))
        
    mean_probabilities = np.mean(np.asarray(all_probs), axis=0)
    return mean_probabilities

def train_test_split_indices(indices, test_size=0.15, stratify=None, random_state=42):
    """
    Helper to stratify split and return train/val indices.
    """
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )
    return train_idx, val_idx

def main():
    parser = argparse.ArgumentParser(description="RA-HLPR Recommender Pipeline")
    parser.add_argument("--dataset", choices=["student-mat", "student-por", "xapi"], required=True, help="Dataset to process")
    args = parser.parse_args()
    
    logger.info("Starting RA-HLPR pipeline for dataset: %s", args.dataset)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load splits
    train_pool, locked_test = load_or_create_splits(args.dataset, "3class")
    logger.info("Loaded dataset splits. Train Pool: %d rows, Locked Test: %d rows", len(train_pool), len(locked_test))
    
    # Load hyperparams
    best_params_path = MODELS_DIR / f"{args.dataset}_3class_best_params.json"
    if not best_params_path.exists():
        raise FileNotFoundError(f"Missing best params config at: {best_params_path}")
    best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
    
    # 2. Generate class probabilities on both train pool and test set
    logger.info("Generating ensemble class probabilities for Train Pool...")
    train_class_probs = get_ensemble_probabilities(args.dataset, train_pool, train_pool, locked_test, best_params, device)
    
    logger.info("Generating ensemble class probabilities for Locked Test...")
    test_class_probs = get_ensemble_probabilities(args.dataset, locked_test, train_pool, locked_test, best_params, device)
    
    # 3. Generate weak labels
    logger.info("Generating domain weak labels...")
    train_weak_labels = generate_weak_labels(train_pool, args.dataset)
    test_weak_labels = generate_weak_labels(locked_test, args.dataset)
    
    # 4. Extract student features for RiskDiagnosisHead
    train_features = extract_features(train_pool, args.dataset)
    test_features = extract_features(locked_test, args.dataset)
    
    # 5. Train RiskDiagnosisHead
    logger.info("Training RiskDiagnosisHead MLP on Train Pool...")
    risk_model = train_risk_head(
        features=train_features,
        class_probs=train_class_probs,
        targets=train_weak_labels,
        epochs=350,
        lr=0.005,
        device=str(device)
    )
    
    # 6. Diagnose risks on the test set
    logger.info("Diagnosing risks on Locked Test...")
    test_risk_probs = risk_model.predict_proba(test_features, test_class_probs, device=str(device))
    
    # 7. Create/load Intervention Knowledge Base
    output_dir = Path("outputs/recommender") / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_df, mapping_df = load_knowledge_base(output_dir)
    logger.info("Loaded educational interventions from knowledge base.")
    
    # 8. Score interventions and generate paths
    candidate_generator = CandidateGenerator(catalog_df)
    scorer = HybridScorer(catalog_df, mapping_df)
    planner = PathPlanner()
    
    kind = DATASETS[args.dataset].kind
    risk_codes = [
        "R1_LOW_PRIOR_PERFORMANCE", "R2_DECLINING_TREND", "R3_ATTENDANCE_RISK", "R4_LOW_ENGAGEMENT", "R5_INSUFFICIENT_STUDY_TIME", "R6_HIGH_FAILURE_PROBABILITY"
    ] if kind == "student" else [
        "R3_ATTENDANCE_RISK", "R4_LOW_ENGAGEMENT", "R6_HIGH_FAILURE_PROBABILITY"
    ]
    
    recommendation_results = []
    learning_paths = []
    all_recs_list = []
    
    for i in range(len(locked_test)):
        # Construct diagnosed risks dictionary
        student_diagnosed_risks = {risk_codes[j]: float(test_risk_probs[i, j]) for j in range(len(risk_codes))}
        student_features = locked_test.iloc[i].to_dict()
        class_probs = test_class_probs[i].tolist()
        pred_class = int(np.argmax(class_probs))
        
        # Filter candidates before scoring
        student_candidates_df = candidate_generator.generate_candidates(
            student_diagnosed_risks,
            pred_class,
            class_probabilities=class_probs,
        )
        
        # Score filtered interventions
        recs = scorer.score_student(student_features, student_diagnosed_risks, class_probs, pred_class, kind, candidates_df=student_candidates_df)
        all_recs_list.append(recs)
        
        # Save top recommendations
        for rank_idx, rec in enumerate(recs[:5]):
            recommendation_results.append({
                "student_index": i,
                "rank": rank_idx + 1,
                "item_id": rec["item_id"],
                "intervention_name": rec["intervention_name"],
                "score": round(rec["score"], 4),
                "explanation": rec["explanation"]
            })
            
        # Plan path
        path = planner.generate_path(recs, pred_class, student_diagnosed_risks)
        learning_paths.append({
            "student_index": i,
            "path": path
        })
        
    # 9. Evaluate
    logger.info("Evaluating RA-HLPR system performance...")
    # Format actual risks list
    actual_risks_list = []
    for i in range(len(locked_test)):
        student_actual = [risk_codes[j] for j in range(len(risk_codes)) if test_weak_labels[i, j] == 1.0]
        actual_risks_list.append(student_actual)
        
    risk_eval = evaluate_risk_diagnosis(test_weak_labels, test_risk_probs)
    ranking_eval = evaluate_ranking(all_recs_list, actual_risks_list, catalog_df, k=3)
    path_eval = evaluate_path_quality([lp["path"] for lp in learning_paths], actual_risks_list, catalog_df)
    
    metrics = {
        "dataset": args.dataset,
        "risk_diagnosis": risk_eval,
        "ranking": ranking_eval,
        "path_quality": path_eval
    }
    
    # 10. Save outputs
    # A. Risk Predictions
    risk_pred_df = pd.DataFrame(test_risk_probs, columns=risk_codes)
    risk_pred_df.insert(0, "student_index", range(len(locked_test)))
    risk_pred_df.to_csv(output_dir / "risk_predictions.csv", index=False)
    
    # B. Recommendation Results
    rec_results_df = pd.DataFrame(recommendation_results)
    rec_results_df.to_csv(output_dir / "recommendation_results.csv", index=False)
    
    # C. Learning Paths
    with open(output_dir / "learning_paths.json", "w", encoding="utf-8") as f:
        json.dump(learning_paths, f, indent=2, ensure_ascii=False)
        
    # D. Recommender Metrics
    with open(output_dir / "recommender_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    # E. Recommender Report with Case Studies
    generate_report(args.dataset, metrics, locked_test, test_class_probs, test_risk_probs, all_recs_list, learning_paths, risk_codes, output_dir)
    
    logger.info("RA-HLPR pipeline completed successfully. All outputs saved to outputs/recommender/")

def generate_report(dataset_name, metrics, locked_test, test_class_probs, test_risk_probs, all_recs_list, learning_paths, risk_codes, output_dir):
    """
    Generates a comprehensive Markdown report including evaluation metrics and 3 specific student case studies.
    """
    # Select Case Studies
    # Case 1: High Risk Student (predicted class = 0 / Low performance, which maps to high risk)
    # Case 2: Moderate Risk Student (predicted class = 1 / Medium performance)
    # Case 3: Stable Student (predicted class = 2 / High performance)
    
    class_predictions = np.argmax(test_class_probs, axis=1)
    
    high_risk_idx = None
    moderate_risk_idx = None
    stable_idx = None
    
    for idx in range(len(locked_test)):
        pred = class_predictions[idx]
        if pred == 0 and high_risk_idx is None:
            high_risk_idx = idx
        elif pred == 1 and moderate_risk_idx is None:
            moderate_risk_idx = idx
        elif pred == 2 and stable_idx is None:
            stable_idx = idx
            
    # Fallbacks if some classes aren't present
    if high_risk_idx is None:
        high_risk_idx = 0
    if moderate_risk_idx is None:
        moderate_risk_idx = min(1, len(locked_test)-1)
    if stable_idx is None:
        stable_idx = min(2, len(locked_test)-1)
        
    case_indices = [
        ("High Risk (Struggling)", high_risk_idx),
        ("Moderate Risk (Average)", moderate_risk_idx),
        ("Stable (High Performer)", stable_idx)
    ]
    
    report_content = f"""# Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) Evaluation Report
**Dataset**: {dataset_name}

---

## 1. Executive Summary of Metrics

### Risk Diagnosis Metrics
* **Micro F1**: {metrics["risk_diagnosis"]["f1_micro"]:.4f}
* **Macro F1**: {metrics["risk_diagnosis"]["f1_macro"]:.4f}
* **Micro Precision**: {metrics["risk_diagnosis"]["precision_micro"]:.4f}
* **Macro Precision**: {metrics["risk_diagnosis"]["precision_macro"]:.4f}
* **Micro Recall**: {metrics["risk_diagnosis"]["recall_micro"]:.4f}
* **Macro Recall**: {metrics["risk_diagnosis"]["recall_macro"]:.4f}
* **Hamming Loss**: {metrics["risk_diagnosis"]["hamming_loss"]:.4f}

### Ranking Metrics (at K=3)
* **Precision@3**: {metrics["ranking"]["precision_at_3"]:.4f}
* **Recall@3**: {metrics["ranking"]["recall_at_3"]:.4f}
* **NDCG@3**: {metrics["ranking"]["ndcg_at_3"]:.4f}
* **Catalog Coverage@3**: {metrics["ranking"]["coverage_at_3"]:.4f}

### Path Quality Metrics
* **Risk Coverage Rate**: {metrics["path_quality"]["risk_coverage_rate"]:.4f}
* **Workload Balance (std hours/week)**: {metrics["path_quality"]["workload_balance_std"]:.4f}
* **Difficulty Progression Rate**: {metrics["path_quality"]["difficulty_progression_rate"]:.4f}
* **Prerequisite Violation Rate**: {metrics["path_quality"]["prerequisite_violation_rate"]:.4f}

---

## 2. Student Case Studies

"""
    
    for profile_name, s_idx in case_indices:
        record = locked_test.iloc[s_idx]
        class_probs = test_class_probs[s_idx]
        pred_class = int(np.argmax(class_probs))
        class_names = ["Low", "Medium", "High"]
        
        risks_dict = {risk_codes[j]: float(test_risk_probs[s_idx, j]) for j in range(len(risk_codes))}
        recs = all_recs_list[s_idx]
        path = learning_paths[s_idx]["path"]
        
        # Select important features to display
        if "student" in dataset_name.lower():
            key_feats = f"Absences: {record.get('absences', 'N/A')}, Study Time: {record.get('studytime', 'N/A')}/4, Failures: {record.get('failures', 'N/A')}, G1: {record.get('G1', 'N/A')}, G2: {record.get('G2', 'N/A')}"
        else:
            key_feats = f"Raised Hands: {record.get('raisedhands', 'N/A')}, Visited Resources: {record.get('VisITedResources', 'N/A')}, Discussion: {record.get('Discussion', 'N/A')}, Absences: {record.get('StudentAbsenceDays', 'N/A')}"
            
        report_content += f"""### Case Study: {profile_name} Student (Test Index {s_idx})
* **Student Context**: {key_feats}
* **Predicted Academic Performance**: Class {pred_class} ({class_names[pred_class]}) - Probabilities: [Low: {class_probs[0]:.2f}, Medium: {class_probs[1]:.2f}, High: {class_probs[2]:.2f}]
* **Diagnosed Academic Risks**:
"""
        for code, val in risks_dict.items():
            report_content += f"  - `{code}`: {val:.4f} probability\n"
            
        report_content += f"\n* **Top 3 Recommended Interventions**:\n"
        for r_idx, r in enumerate(recs[:3]):
            report_content += f"  {r_idx+1}. **{r['intervention_name']}** (Score: {r['score']:.4f})\n     * {r['description']}\n     * *Score Breakdown*: {r['explanation']}\n"
            
        report_content += f"\n* **Generated 4-Week Learning Path**:\n"
        for week_name, w_info in path["weeks"].items():
            report_content += f"  * **{week_name} - Theme: {w_info['theme']}**\n"
            report_content += f"    * *Objective*: {w_info['objective']}\n"
            report_content += f"    * *Recommended Actions*:\n"
            for action in w_info["recommended_actions"]:
                report_content += f"      - {action}\n"
            report_content += f"    * *Expected Outcome*: {w_info['expected_outcome']}\n"
            report_content += f"    * *Educational Rationale*: {w_info['explanation']}\n"
            
        report_content += "\n---\n\n"
        
    (Path(output_dir) / "recommender_report.md").write_text(report_content, encoding="utf-8")

if __name__ == "__main__":
    main()
