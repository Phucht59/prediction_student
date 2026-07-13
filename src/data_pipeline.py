import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE, SMOTENC
from src.config import DATASETS, STUDENT_G3_3CLASS_BINS, XAPI_CLASS_MAPPING
from src.utils import setup_logger

logger = setup_logger("data_pipeline")

logger = setup_logger("data_split")

SOURCE_ROW_NUMBER_COLUMN = "__source_row_number"
PROTECTED_METADATA_COLUMNS = frozenset({SOURCE_ROW_NUMBER_COLUMN})


def is_target_or_target_derived(column: str) -> bool:
    """Return true for G3 and every named G3 derivative used by this project."""
    normalized = str(column).lower().replace("-", "_")
    return normalized == "g3" or normalized.startswith("g3_") or "g3" in normalized.split("_")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def attach_source_row_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Attach zero-based raw CSV row numbers as protected lineage metadata."""
    df = df.copy()
    if SOURCE_ROW_NUMBER_COLUMN not in df.columns:
        df.insert(0, SOURCE_ROW_NUMBER_COLUMN, np.arange(len(df), dtype=int))
    return df


def drop_protected_metadata(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in PROTECTED_METADATA_COLUMNS if c in df.columns], errors="ignore")


def exclude_protected_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column not in PROTECTED_METADATA_COLUMNS]


def process_target_and_stratify(df: pd.DataFrame, target_col: str, kind: str, target_mode: str = "3class") -> pd.DataFrame:
    """Prepare target column for stratification."""
    if kind == "student":
        # Save raw continuous G3
        df["G3_raw"] = df[target_col]
        # Create bins for stratifying based on mode
        if target_mode == "3class":
            df[target_col] = pd.cut(df[target_col], bins=STUDENT_G3_3CLASS_BINS, labels=[0, 1, 2], include_lowest=True)
            df["_strat_target"] = df[target_col]
        else:
            # 5-class for student
            df[target_col] = pd.cut(df[target_col], bins=[0, 9, 11, 13, 15, 20], labels=[0, 1, 2, 3, 4], include_lowest=True)
            df["_strat_target"] = df[target_col]
    elif kind == "xapi":
        # xAPI is naturally 3-class (L, M, H)
        df[target_col] = df[target_col].map(XAPI_CLASS_MAPPING)
        df["_strat_target"] = df[target_col]
    else:
        df["_strat_target"] = df[target_col]
        
    return df

def check_no_leakage(train_indices, test_indices):
    """Safety check to ensure test indices do not leak into train."""
    intersection = set(train_indices).intersection(set(test_indices))
    if len(intersection) > 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED! {len(intersection)} samples overlap between train and test.")




logger = setup_logger("feature_eng")

XAPI_BEHAVIOR_DERIVED_CONTEXT_EXCLUSIONS = frozenset(
    {
        "engagement_score",
        "absence_risk",
        "hands_resource_ratio",
        "active_participation",
        "resource_engagement_ratio",
    }
)


def engineer_student_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer derived features for student-mat and student-por datasets."""
    df = df.copy()
    created_features = []
    
    if "G1" in df.columns and "G2" in df.columns:
        df["grade_growth"] = df["G2"] - df["G1"]
        df["grade_avg"] = (df["G1"] + df["G2"]) / 2
        created_features.extend(["grade_growth", "grade_avg"])
        
    if "absences" in df.columns and "studytime" in df.columns:
        # epsilon to avoid division by zero
        df["absence_study_ratio"] = df["absences"] / (df["studytime"] + 0.1)
        created_features.append("absence_study_ratio")
        
    if "failures" in df.columns and "absence_study_ratio" in df.columns:
        df["failure_risk"] = df["failures"] + df["absence_study_ratio"]
        created_features.append("failure_risk")
        
    if "Dalc" in df.columns and "Walc" in df.columns:
        df["alcohol_risk"] = df["Dalc"] + df["Walc"]
        created_features.append("alcohol_risk")
        
    if "goout" in df.columns and "freetime" in df.columns:
        df["social_risk"] = df["goout"] + df["freetime"]
        created_features.append("social_risk")
        
    logger.info(f"Student engineered features created: {created_features}")
    return df

def engineer_xapi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer derived features for xAPI-Edu-Data."""
    df = df.copy()
    created_features = []
    
    engagement_cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    if all(col in df.columns for col in engagement_cols):
        df["engagement_score"] = df[engagement_cols].sum(axis=1)
        created_features.append("engagement_score")
        
        
    if "StudentAbsenceDays" in df.columns:
        df["absence_risk"] = df["StudentAbsenceDays"].apply(lambda x: 1 if x == "Above-7" else 0)
        created_features.append("absence_risk")
        
    if "ParentAnsweringSurvey" in df.columns:
        df["parent_support_signal"] = df["ParentAnsweringSurvey"].apply(lambda x: 1 if x == "Yes" else 0)
        created_features.append("parent_support_signal")
        
    if "raisedhands" in df.columns and "VisITedResources" in df.columns:
        df["hands_resource_ratio"] = df["raisedhands"] / (df["VisITedResources"] + 1)
        created_features.append("hands_resource_ratio")
        
    if "raisedhands" in df.columns and "Discussion" in df.columns:
        df["active_participation"] = df["raisedhands"] * df["Discussion"]
        created_features.append("active_participation")
        
    if "VisITedResources" in df.columns and "engagement_score" in df.columns:
        df["resource_engagement_ratio"] = df["VisITedResources"] / (df["engagement_score"] + 1)
        created_features.append("resource_engagement_ratio")
        
    logger.info(f"xAPI engineered features created: {created_features}")
    return df

def apply_feature_engineering(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Apply specific feature engineering depending on dataset kind."""
    if kind == "student":
        return engineer_student_features(df)
    elif kind == "xapi":
        return engineer_xapi_features(df)
    return df


import numpy as np
from scipy.stats import pearsonr, chi2_contingency


logger = setup_logger("feature_selection")

class FeatureSelector:
    def __init__(
        self,
        target_col: str,
        use_feature_selection: bool = True,
        p_value_threshold: float = 0.1,
        required_features: list | None = None,
    ):
        self.target_col = target_col
        self.use_feature_selection = use_feature_selection
        self.p_value_threshold = p_value_threshold
        self.required_features = required_features or []
        self.selected_features = []
        
    def fit_transform(self, df: pd.DataFrame, numerical_cols: list, categorical_cols: list):
        # Exclude G3_raw from features to prevent leakage
        numerical_cols = [c for c in numerical_cols if c != "G3_raw" and c not in PROTECTED_METADATA_COLUMNS]
        categorical_cols = [c for c in categorical_cols if c != "G3_raw" and c not in PROTECTED_METADATA_COLUMNS]
        
        if not self.use_feature_selection:
            self.selected_features = numerical_cols + categorical_cols
            logger.info("Feature selection is disabled. Keeping all features.")
            return df
            
        y = df[self.target_col]
        selected = []
        
        # Pearson for numerical
        for col in numerical_cols:
            if col not in df.columns or df[col].nunique() <= 1:
                continue
            corr, p_value = pearsonr(df[col], y)
            if not np.isnan(p_value) and p_value < self.p_value_threshold:
                selected.append(col)
                
        # Chi-square for categorical
        for col in categorical_cols:
            if col not in df.columns or df[col].nunique() <= 1:
                continue
            contingency_table = pd.crosstab(df[col], y)
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            if not np.isnan(p_value) and p_value < self.p_value_threshold:
                selected.append(col)
                
        # Always keep target and some essential engineered features if they dropped accidentally but shouldn't be dropped?
        # Actually let's trust the stat test, but ensure we don't drop sequence features later.
        selected.extend(
            feature
            for feature in self.required_features
            if feature in df.columns and feature not in selected
        )
        self.selected_features = [f for f in selected if f != "G3_raw"]
        logger.info(f"Feature selection complete. Selected {len(self.selected_features)} / {len(numerical_cols) + len(categorical_cols)} features.")
        return self.transform(df)
        
    def transform(self, df: pd.DataFrame):
        if not self.use_feature_selection:
            return df
        
        cols_to_keep = [col for col in self.selected_features if col in df.columns]
        if self.target_col in df.columns and self.target_col not in cols_to_keep:
            cols_to_keep.append(self.target_col)
            
        return df[cols_to_keep]




logger = setup_logger("preprocessing")

class DataPreprocessor:
    def __init__(
        self,
        target_col: str,
        oversample_method: str = "none",
        smote_ratio: float = 1.0,
        resampling_k_neighbors: int = 5,
        oversampling_feature_columns: list[str] | None = None,
    ):
        self.target_col = target_col
        self.oversample_method = oversample_method.lower()
        self.smote_ratio = smote_ratio
        self.resampling_k_neighbors = resampling_k_neighbors
        self.oversampling_feature_columns = list(oversampling_feature_columns or [])
        self.numerical_cols = []
        self.categorical_cols = []
        self.scalers = {}
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        
    def fit_transform(self, df: pd.DataFrame, apply_oversampling: bool = True):
        """Fit on train pool and transform it. Also handles train-only oversampling."""
        df = df.copy()
        
        # Identify columns
        X = drop_protected_metadata(df.drop(columns=[self.target_col]))
        X = X.drop(columns=[column for column in X.columns if is_target_or_target_derived(column)], errors="ignore")
        
        self.numerical_cols = [
            c for c in X.select_dtypes(include=[np.number]).columns.tolist()
            if c != "G3_raw" and c not in PROTECTED_METADATA_COLUMNS
        ]
        self.categorical_cols = [
            c for c in X.select_dtypes(exclude=[np.number]).columns.tolist()
            if c != "G3_raw" and c not in PROTECTED_METADATA_COLUMNS
        ]
        
        # Fit & transform target
        y_encoded = self.target_encoder.fit_transform(df[self.target_col])
        
        # Fit & transform features
        for col in self.numerical_cols:
            scaler = MinMaxScaler()
            X[col] = scaler.fit_transform(X[[col]])
            self.scalers[col] = scaler
            
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Handle unknown labels gracefully by converting to string
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            
        df_out = X.copy()
        df_out[self.target_col] = y_encoded
        
        if apply_oversampling:
            df_out = self.apply_oversampling(df_out)
            
        return df_out
        
    def apply_oversampling(self, df: pd.DataFrame):
        """Apply oversampling to the preprocessed and potentially feature-selected DataFrame."""
        method = self.oversample_method
        if method in {"class_weight", "focal_loss"}:
            method = "none"

        if method == "none":
            return df
            
        df = df.copy()
        if self.oversampling_feature_columns:
            allowed_features = [
                column
                for column in self.oversampling_feature_columns
                if column in df.columns and column != self.target_col
            ]
            if not allowed_features:
                logger.warning("No allowed model-input features available for oversampling. Falling back to no oversampling.")
                return df
            df = df[allowed_features + [self.target_col]]

        X = df.drop(columns=[self.target_col])
        y_encoded = df[self.target_col]
        
        remaining_cat_cols = [col for col in self.categorical_cols if col in X.columns]
        
        if method == "adasyn" and remaining_cat_cols:
            logger.warning(
                "ADASYN is disabled for this pipeline because it can interpolate "
                "label-encoded categorical values. Falling back to SMOTENC/SMOTE."
            )
            method = "smotenc" if remaining_cat_cols else "smote"

        logger.info(f"Applying {method.upper()} on train set with ratio {self.smote_ratio}...")
        
        # Dynamically calculate sampling strategy for multiclass
        class_counts = pd.Series(y_encoded).value_counts()
        majority_count = class_counts.max()
        effective_k_neighbors = min(
            self.resampling_k_neighbors,
            max(1, int(class_counts.min()) - 1),
        )
        strategy = {}
        for cls, count in class_counts.items():
            if count == majority_count:
                strategy[cls] = count
            else:
                target = int(majority_count * self.smote_ratio)
                strategy[cls] = max(count, target) # Do not undersample if already larger
                
        if method in {"random", "random_oversampling", "ros"}:
            sampler = RandomOverSampler(
                sampling_strategy=strategy,
                random_state=42,
            )
        elif remaining_cat_cols or method == "smotenc":
            if not remaining_cat_cols:
                logger.warning("SMOTENC requested but no categorical columns remain; falling back to SMOTE.")
                sampler = SMOTE(
                    sampling_strategy=strategy,
                    random_state=42,
                    k_neighbors=effective_k_neighbors,
                )
            else:
                cat_indices = [X.columns.get_loc(c) for c in remaining_cat_cols]
                sampler = SMOTENC(
                    categorical_features=cat_indices,
                    sampling_strategy=strategy,
                    random_state=42,
                    k_neighbors=effective_k_neighbors,
                )
        elif method == "adasyn":
            sampler = ADASYN(
                sampling_strategy=strategy,
                random_state=42,
                n_neighbors=effective_k_neighbors,
            )
        elif method == "smote":
            sampler = SMOTE(
                sampling_strategy=strategy,
                random_state=42,
                k_neighbors=effective_k_neighbors,
            )
        else:
            logger.warning("Unknown oversampling method '%s'. Falling back to no oversampling.", self.oversample_method)
            return df
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y_encoded)
            X = pd.DataFrame(X_resampled, columns=X.columns)
            # Ensure resampled categorical variables are rounded and cast to integers
            for col in remaining_cat_cols:
                X[col] = X[col].round().astype(int)
            y_encoded = y_resampled
        except Exception as e:
            logger.warning(f"{self.oversample_method.upper()} failed (likely too few samples). Error: {e}. Falling back to no oversampling.")
            
        df_out = X.copy()
        df_out[self.target_col] = y_encoded
        return df_out
        
    def transform(self, df: pd.DataFrame):
        """Transform validation/test sets without fitting or oversampling."""
        df = df.copy()
        X = drop_protected_metadata(df.drop(columns=[self.target_col], errors='ignore'))
        X = X.drop(columns=[column for column in X.columns if is_target_or_target_derived(column)], errors="ignore")
        
        if self.target_col in df.columns:
            # For target, handle unseen classes by mapping to -1 or known
            known_classes = set(self.target_encoder.classes_)
            y_encoded = df[self.target_col].apply(lambda x: self.target_encoder.transform([x])[0] if x in known_classes else -1)
            df_out = X.copy()
            df_out[self.target_col] = y_encoded
        else:
            df_out = X.copy()
        
        for col in self.numerical_cols:
            if col in df_out.columns:
                df_out[col] = self.scalers[col].transform(df_out[[col]])
                
        for col in self.categorical_cols:
            if col in df_out.columns:
                # Handle unseen labels in categorical features
                le = self.label_encoders[col]
                known = set(le.classes_)
                df_out[col] = df_out[col].astype(str).apply(lambda x: le.transform([x])[0] if x in known else 0)
                
        return df_out



def get_sequence_columns(kind: str) -> list[str]:
    if kind == "student":
        return ["G1", "G2"]
    if kind == "xapi":
        return ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    raise ValueError(f"Unsupported dataset kind: {kind}")


def get_context_excluded_columns(kind: str) -> set[str]:
    if kind == "xapi":
        return set(XAPI_BEHAVIOR_DERIVED_CONTEXT_EXCLUSIONS)
    return set()


class StudentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, kind: str, target_col: str, numerical_cols: list, categorical_cols: list):
        self.target_col = target_col
        self.y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        
        # Load G3_raw or default to 0.0
        if kind == "xapi":
            self.reg_label = np.zeros(len(df), dtype=np.float32)
        elif "G3_raw" in df.columns:
            self.reg_label = df["G3_raw"].values.astype(np.float32)
        else:
            self.reg_label = np.zeros(len(df), dtype=np.float32)
            
        allowed_seq_cols = get_sequence_columns(kind)
        seq_cols = [c for c in allowed_seq_cols if c in df.columns]
        if len(seq_cols) < 2:
            raise ValueError(
                f"Dataset kind '{kind}' requires at least two chronological sequence columns; "
                f"available={seq_cols}, required={allowed_seq_cols}."
            )
        forbidden_sequence_inputs = {
            self.target_col,
            "G3",
            "G3_raw",
            "_strat_target",
            SOURCE_ROW_NUMBER_COLUMN,
            "record_id",
            "dataset_version_id",
            "prediction_id",
            "run_id",
        }
        if set(seq_cols) & forbidden_sequence_inputs:
            raise ValueError(f"Forbidden target or lineage metadata in sequence input: {sorted(set(seq_cols) & forbidden_sequence_inputs)}")
        if seq_cols != allowed_seq_cols:
            raise ValueError(f"Sequence columns must match the approved chronological allowlist: {allowed_seq_cols}")
        if not seq_cols:
            raise ValueError(f"No sequential features are available for dataset kind '{kind}'.")
        self.seq_cols = seq_cols
        self.seq_x = df[seq_cols].values[..., np.newaxis]
                
        # Final predictor is sequence-only. Profile/tabular columns may remain
        # in the dataframe for recommendation/explanation but are not exposed
        # to the active CNN-BiLSTM classifier.
        self.num_cols = []
        self.cat_cols = []

        self.num_x = np.zeros((len(df), 0), dtype=np.float32)
        self.cat_x = np.zeros((len(df), 0), dtype=int)
        
        # Original features for recommendation
        self.original_features = df.to_dict('records')
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        seq = torch.tensor(self.seq_x[idx], dtype=torch.float32)
        num = torch.tensor(self.num_x[idx], dtype=torch.float32)
        cat = torch.tensor(self.cat_x[idx], dtype=torch.long)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        reg_val = torch.tensor(self.reg_label[idx], dtype=torch.float32)
        return seq, num, cat, label, idx, reg_val


