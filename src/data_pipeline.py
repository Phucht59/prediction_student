import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from src.config import DEFAULT_SEED, LOCKED_TEST_SIZE, PROCESSED_DIR, DATASETS, STUDENT_G3_3CLASS_BINS, XAPI_CLASS_MAPPING
from src.utils import setup_logger

logger = setup_logger("data_pipeline")

from src.config import DEFAULT_SEED, LOCKED_TEST_SIZE, PROCESSED_DIR, DATASETS, STUDENT_G3_3CLASS_BINS, XAPI_CLASS_MAPPING


logger = setup_logger("v26_data_split")

def process_target_and_stratify(df: pd.DataFrame, target_col: str, kind: str, target_mode: str = "3class") -> pd.DataFrame:
    """Prepare target column for stratification."""
    if kind == "student":
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

def create_and_save_locked_test(df: pd.DataFrame, ds_name: str, target_mode: str = "3class"):
    """Split data into 80% train pool and 20% locked test, and save them."""
    spec = DATASETS[ds_name]
    df_strat = process_target_and_stratify(df.copy(), spec.target_col, spec.kind, target_mode)
    
    # Drop rows where strat target is null if any
    df_strat = df_strat.dropna(subset=["_strat_target"])
    
    train_pool, locked_test = train_test_split(
        df_strat,
        test_size=LOCKED_TEST_SIZE,
        stratify=df_strat["_strat_target"],
        random_state=DEFAULT_SEED
    )
    
    # Remove internal _strat_target
    train_pool = train_pool.drop(columns=["_strat_target"])
    locked_test = locked_test.drop(columns=["_strat_target"])
    
    train_path = PROCESSED_DIR / f"{ds_name}_{target_mode}_train_pool.csv"
    test_path = PROCESSED_DIR / f"{ds_name}_{target_mode}_locked_test.csv"
    
    train_pool.to_csv(train_path, index=False)
    locked_test.to_csv(test_path, index=False)
    
    logger.info(f"[{ds_name} - {target_mode}] Train pool: {len(train_pool)} rows. Locked test: {len(locked_test)} rows.")
    return train_path, test_path

def load_splits(ds_name: str, target_mode: str = "3class"):
    """Load the saved splits. Will raise an error if not found."""
    train_path = PROCESSED_DIR / f"{ds_name}_{target_mode}_train_pool.csv"
    test_path = PROCESSED_DIR / f"{ds_name}_{target_mode}_locked_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing split files for {ds_name} {target_mode}. Run create_locked_test script first.")
        
    return pd.read_csv(train_path), pd.read_csv(test_path)

def check_no_leakage(train_indices, test_indices):
    """Safety check to ensure test indices do not leak into train."""
    intersection = set(train_indices).intersection(set(test_indices))
    if len(intersection) > 0:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED! {len(intersection)} samples overlap between train and test.")




logger = setup_logger("v26_feature_eng")

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


logger = setup_logger("v26_feature_selection")

class V26FeatureSelector:
    def __init__(self, target_col: str, use_feature_selection: bool = True, p_value_threshold: float = 0.1):
        self.target_col = target_col
        self.use_feature_selection = use_feature_selection
        self.p_value_threshold = p_value_threshold
        self.selected_features = []
        
    def fit_transform(self, df: pd.DataFrame, numerical_cols: list, categorical_cols: list):
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
        self.selected_features = selected
        logger.info(f"Feature selection complete. Selected {len(selected)} / {len(numerical_cols) + len(categorical_cols)} features.")
        return self.transform(df)
        
    def transform(self, df: pd.DataFrame):
        if not self.use_feature_selection:
            return df
        
        cols_to_keep = [col for col in self.selected_features if col in df.columns]
        if self.target_col in df.columns and self.target_col not in cols_to_keep:
            cols_to_keep.append(self.target_col)
            
        return df[cols_to_keep]




logger = setup_logger("v26_preprocessing")

class V26Preprocessor:
    def __init__(self, target_col: str, oversample_method: str = "none"):
        self.target_col = target_col
        self.oversample_method = oversample_method.lower()
        self.numerical_cols = []
        self.categorical_cols = []
        self.scalers = {}
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        
    def fit_transform(self, df: pd.DataFrame):
        """Fit on train pool and transform it. Also handles SMOTE/ADASYN."""
        df = df.copy()
        
        # Identify columns
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Fit & transform target
        y_encoded = self.target_encoder.fit_transform(y)
        
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
            
        # Apply Oversampling ONLY on train
        if self.oversample_method in ["smote", "adasyn"]:
            # SMOTE/ADASYN requires numeric inputs, our categorical are label encoded so it's numeric now.
            logger.info(f"Applying {self.oversample_method.upper()} on train set...")
            sampler = SMOTE(random_state=42) if self.oversample_method == "smote" else ADASYN(random_state=42)
            try:
                X_resampled, y_resampled = sampler.fit_resample(X, y_encoded)
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y_encoded = y_resampled
            except Exception as e:
                logger.warning(f"{self.oversample_method.upper()} failed (likely too few samples). Error: {e}. Falling back to no oversampling.")
        
        df_out = X.copy()
        df_out[self.target_col] = y_encoded
        return df_out
        
    def transform(self, df: pd.DataFrame):
        """Transform validation/test sets without fitting or oversampling."""
        df = df.copy()
        X = df.drop(columns=[self.target_col], errors='ignore')
        
        if self.target_col in df.columns:
            # For target, handle unseen classes by mapping to -1 or known
            known_classes = set(self.target_encoder.classes_)
            # Map unseen to a default or keep as is (should not happen in target usually)
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



class V27Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, kind: str, target_col: str, numerical_cols: list, categorical_cols: list):
        self.y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        
        if kind == "student":
            seq_cols = [c for c in ["G1", "G2"] if c in df.columns]
            if len(seq_cols) == 0:
                self.seq_x = np.zeros((len(df), 1, 1))
            else:
                self.seq_x = df[seq_cols].values[..., np.newaxis] # (N, L, 1)
        else:
            # NO pseudo sequence for xAPI in V27 Tabular models
            seq_cols = []
            self.seq_x = np.zeros((len(df), 1, 1))
                
        # Context features
        num_cols = [c for c in numerical_cols if c in df.columns and c not in seq_cols]
        cat_cols = [c for c in categorical_cols if c in df.columns and c not in seq_cols]
        
        self.num_x = df[num_cols].values if len(num_cols) > 0 else np.zeros((len(df), 1))
        self.cat_x = df[cat_cols].values.astype(int) if len(cat_cols) > 0 else np.zeros((len(df), 1), dtype=int)
        
        # Original features for recommendation
        self.original_features = df.to_dict('records')
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        seq = torch.tensor(self.seq_x[idx], dtype=torch.float32)
        num = torch.tensor(self.num_x[idx], dtype=torch.float32)
        cat = torch.tensor(self.cat_x[idx], dtype=torch.long)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return seq, num, cat, label, idx


