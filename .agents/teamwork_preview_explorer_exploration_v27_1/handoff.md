# Codebase Exploration and Pipeline Audit Report

## 1. Observation

During our systematic exploration of the codebase, we analyzed the data pipeline, model configurations, and architecture definitions across the following key files:
- `src/data_pipeline.py`
- `src/train_pipeline.py`
- `scripts/run_pipeline.py`
- `src/models/models.py`

### 1.1 Data Splitting and Feature Engineering
In `scripts/run_pipeline.py` (Lines 212–223), the train-test-validation splits are prepared as follows:
```python
        # 1. Tách validation trước khi resampling
        labels = train_pool[spec.target_col].astype(int).to_numpy()
        indices = np.arange(len(train_pool))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.15,
            stratify=labels,
            random_state=seed,
        )
        
        train_sub = apply_feature_engineering(train_pool.iloc[train_indices].copy(), spec.kind)
        val_sub = apply_feature_engineering(train_pool.iloc[val_indices].copy(), spec.kind)
        test_engineered = apply_feature_engineering(locked_test.copy(), spec.kind)
```
This is followed by the instantiation and application of the preprocessor:
```python
        preprocessor = DataPreprocessor(
            target_col=spec.target_col,
            oversample_method=best_params["oversample_method"],
            smote_ratio=best_params.get("smote_ratio", 1.0),
            resampling_k_neighbors=best_params.get("resampling_k_neighbors", 5),
        )
        train_prep = preprocessor.fit_transform(train_sub)
        val_prep = preprocessor.transform(val_sub)
        test_prep = preprocessor.transform(test_engineered)
```
The scaling (`MinMaxScaler`) and encoding (`LabelEncoder`) parameters are fit in `fit_transform` on the training subset (`train_sub`) only, and applied to validation/test via `transform`.

### 1.2 Target Binning and Loss of Continuous Targets
In `src/data_pipeline.py` (Lines 23–29):
```python
        if target_mode == "3class":
            df[target_col] = pd.cut(df[target_col], bins=STUDENT_G3_3CLASS_BINS, labels=[0, 1, 2], include_lowest=True)
            df["_strat_target"] = df[target_col]
```
The original continuous target `G3` is directly binned and overwritten, before the splits are saved to CSV.

### 1.3 Feature Selection Applied on Resampled Data
In `src/train_pipeline.py` (Lines 275–288):
```python
        train_prep = preprocessor.fit_transform(train_fold)
        val_prep = preprocessor.transform(val_fold)

        selector = FeatureSelector(
            target_col=spec.target_col,
            use_feature_selection=True,
            required_features=sequence_columns,
        )
        train_selected = selector.fit_transform(
            train_prep,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        val_selected = selector.transform(val_prep)
```
Here, statistical feature selection (Pearson correlation and Chi-Square contingency tests) is performed on `train_prep`, which is already oversampled.

### 1.4 Resampling (SMOTE / ADASYN) Usage and Categorical Handling
In `src/data_pipeline.py` (Lines 277–324):
```python
        # Apply Oversampling ONLY on train
        if self.oversample_method in ["smote", "adasyn"]:
            # SMOTE/ADASYN requires numeric inputs, our categorical are label encoded so it's numeric now.
            logger.info(f"Applying {self.oversample_method.upper()} on train set with ratio {self.smote_ratio}...")
            ...
            if self.oversample_method == "smote":
                cat_indices = [X.columns.get_loc(c) for c in self.categorical_cols] if self.categorical_cols else []
                if cat_indices:
                    sampler = SMOTENC(
                        categorical_features=cat_indices,
                        sampling_strategy=strategy,
                        random_state=42,
                        k_neighbors=effective_k_neighbors,
                    )
                else:
                    sampler = SMOTE(...)
            else:
                sampler = ADASYN(
                    sampling_strategy=strategy,
                    random_state=42,
                    n_neighbors=effective_k_neighbors,
                )
            try:
                X_resampled, y_resampled = sampler.fit_resample(X, y_encoded)
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y_encoded = y_resampled
```
In `src/train_pipeline.py` (Lines 228–230), the Optuna parameters for the `student` dataset trial search space suggest `"adasyn"` as the categorical oversample technique:
```python
        "oversample_method": trial.suggest_categorical(
            "oversample_method", ["adasyn"]
        ),
```
In `src/data_pipeline.py` (Lines 381–382), the categorical values are cast back to integers when instantiating `StudentDataset`:
```python
        self.num_x = df[self.num_cols].values if self.num_cols else np.zeros((len(df), 1))
        self.cat_x = df[self.cat_cols].values.astype(int) if self.cat_cols else np.zeros((len(df), 1), dtype=int)
```

### 1.5 Architecture of `StudentHybridModel`
In `src/models/models.py`, the core model is defined with:
- **Sequence Branch** (Lines 63–81):
  ```python
  self.sequence_cnn = nn.Sequential(
      nn.Conv1d(
          in_channels=seq_in_channels,
          out_channels=cnn_channels,
          kernel_size=cnn_kernel_size,
          padding=cnn_kernel_size // 2,
      ),
      nn.BatchNorm1d(cnn_channels),
      nn.ReLU(),
      nn.Dropout(sequence_dropout),
  )
  self.sequence_bilstm = nn.LSTM(
      input_size=cnn_channels,
      hidden_size=lstm_hidden_dim,
      batch_first=True,
      bidirectional=True,
  )
  sequence_output_dim = lstm_hidden_dim * 2
  self.sequence_pool = AttentionPooling1D(sequence_output_dim)
  ```
- **Context Branch** (Lines 85–91):
  ```python
  self.context_mlp = nn.Sequential(
      nn.Linear(self.context_input_dim, context_hidden_dim),
      nn.ReLU(),
      nn.Dropout(context_dropout),
      nn.Linear(context_hidden_dim, context_hidden_dim),
      nn.ReLU(),
  )
  ```
- **Fusion and Output** (Lines 93–98):
  ```python
  self.fusion = nn.Sequential(
      nn.Linear(sequence_output_dim + context_hidden_dim, fusion_hidden_dim),
      nn.ReLU(),
      nn.Dropout(fusion_dropout),
  )
  self.classifier = nn.Linear(fusion_hidden_dim, num_classes)
  ```

---

## 2. Logic Chain

### 2.1 Audit of Data Splitting and Preprocessing
1. **Observation 1.1** shows that validation (`val_sub`) and locked test (`test_engineered`) sets are split from the training pool *before* `DataPreprocessor` is instantiated.
2. In `DataPreprocessor.fit_transform`, scalers and label encoders are fitted strictly on the training subset, and only applied via `transform()` to the validation and test subsets. Thus, there is **no direct leakage of scaling/encoding statistics** from validation/test sets to the training set.
3. However, **Observation 1.3** reveals that the `FeatureSelector` computes its Pearson correlation and Chi-Square contingency statistics on `train_prep` (which has already been oversampled by SMOTE/ADASYN).
4. Oversampling generates synthetic samples that are linear combinations/interpolations of existing data points. This artificially inflates the sample size $N$ and introduces collinearity.
5. Performing statistical significance tests on resampled data artificially deflates p-values (making features appear more significant than they are) and leads to **leakage/distortion of feature selection metrics**. Features should be selected based on the *original* training fold distribution before synthetic resampling.
6. **Observation 1.2** shows that the raw continuous target `G3` is overwritten with binned values `[0, 1, 2]`. This means that down-stream regression models cannot access the continuous target. If an auxiliary regression head is added, the preprocessing must be updated to keep the continuous label intact (e.g. as a separate column `target_continuous` or `G3_raw`).

### 2.2 Audit of Resampling Techniques
1. **Observation 1.4** shows that standard `ADASYN` is selected for the `student` dataset, which contains numerous categorical variables.
2. Since ADASYN does not have a categorical-specific implementation (unlike `SMOTENC` for SMOTE), it treats all features in `X` (including label-encoded categorical columns) as continuous numeric values.
3. ADASYN outputs floating-point numbers for these categorical dimensions (e.g. `1.34` or `0.85` for categorical labels).
4. **Observation 1.4** shows that `StudentDataset` forces these float values to integers using `.astype(int)` (truncating `1.34` to `1` and `0.85` to `0`).
5. This truncation behaves like an arbitrary mapping that distorts the categorical distribution and ruins the integrity of the generated categorical representations. Furthermore, it could result in out-of-bounds indices for categorical features if the synthetic value falls outside the valid class range (e.g., negative values or values exceeding the encoder's classes).
6. Resampling is safely applied inside `fit_transform` of the preprocessor on the training split only, and is **not** applied to the validation or test sets (as confirmed by the absence of resampling in the `transform` method).

### 2.3 Architecture Assessment
1. The sequence branch inputs a tensor of shape `(batch_size, seq_len, 1)` and converts it to `(batch_size, cnn_channels, seq_len)` via a 1D Convolution. For the student dataset, `seq_len` is 2 (`G1`, `G2`).
2. A sequence length of 2 is extremely short for Conv1D + BiLSTM. With `kernel_size = 3` and `padding = 1`, the convolution is mainly looking at padded zeros. While valid, this model's sequence modeling capacity is underutilized for the student dataset.
3. The fusion layer simply concatenates the context and sequential vectors. This assumes a simple linear/nonlinear combination of features without allowing the network to dynamically select which branch (temporal or contextual) is more reliable for a given sample.

---

## 3. Caveats

- **No SQL verification**: The DB connection parameters specified in `src/config.py` were not verified as the database configuration is outside the scope of this read-only static analysis.
- **Library version differences**: `imblearn.over_sampling.SMOTENC` might behave differently depending on the version of `imbalanced-learn` installed. In older versions, numeric features must be placed separately, or the output dtype might default to float64, which we must handle defensively.

---

## 4. Conclusion

1. **Direct Data Leakage**: There is no direct validation or test set leakage during preprocessing and scaling. However, **feature selection is performed after resampling**, which represents a statistical leakage/distortion.
2. **Resampling Bug**: The current configuration uses `ADASYN` on the `student` dataset, which contains mixed data. This leads to invalid floating-point values for categorical features which are brutally truncated to integers, distorting the dataset.
3. **Continuous Target Loss**: The data split pipeline overwrites the raw continuous target, which prevents auxiliary regression tasks from being trained.
4. **Fusion Bottleneck**: The current concatenation fusion does not dynamically weight contextual vs. sequential branches.

---

## 5. Proposed Implementation Plan

We propose the following concrete modifications to fix the resampling issues, implement the `StudentHybridV27` architecture, and define the loss functions in `src/losses_v27.py`.

### 5.1 Resampling Fix
1. **Rule-based Resampler Selection**:
   - If `self.categorical_cols` is not empty, force `SMOTENC` for oversampling (raise a warning and fallback if `ADASYN` is requested).
   - If `self.categorical_cols` is empty, allow `SMOTE` or `ADASYN`.
2. **Post-Resampling Casting**:
   - Cast all resampled categorical columns explicitly to integer type.
3. **Feature Selection Reordering**:
   - Run `FeatureSelector.fit_transform` on the training fold *before* applying oversampling.

Here is the proposed Python implementation for `DataPreprocessor` and the training loop sequence:

```python
# Proposed in src/data_pipeline.py
class DataPreprocessor:
    # ... (init code remains similar) ...
    
    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
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
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            
        # Over-sampling
        if self.oversample_method in ["smote", "adasyn"]:
            # Fallback if categorical variables exist but ADASYN is selected
            oversample_method = self.oversample_method
            if self.categorical_cols and oversample_method == "adasyn":
                logger.warning("ADASYN does not support categorical features. Falling back to SMOTENC.")
                oversample_method = "smote"
                
            class_counts = pd.Series(y_encoded).value_counts()
            majority_count = class_counts.max()
            effective_k_neighbors = min(
                self.resampling_k_neighbors,
                max(1, int(class_counts.min()) - 1),
            )
            
            strategy = {
                cls: max(count, int(majority_count * self.smote_ratio))
                for cls, count in class_counts.items() if count < majority_count
            }
            
            if oversample_method == "smote":
                cat_indices = [X.columns.get_loc(c) for c in self.categorical_cols]
                if cat_indices:
                    sampler = SMOTENC(
                        categorical_features=cat_indices,
                        sampling_strategy=strategy,
                        random_state=42,
                        k_neighbors=effective_k_neighbors,
                    )
                else:
                    sampler = SMOTE(
                        sampling_strategy=strategy,
                        random_state=42,
                        k_neighbors=effective_k_neighbors,
                    )
            else:
                sampler = ADASYN(
                    sampling_strategy=strategy,
                    random_state=42,
                    n_neighbors=effective_k_neighbors,
                )
                
            try:
                X_resampled, y_resampled = sampler.fit_resample(X, y_encoded)
                X = pd.DataFrame(X_resampled, columns=X.columns)
                # Ensure resampled categorical variables are strictly integers
                for col in self.categorical_cols:
                    X[col] = X[col].round().astype(int)
                y_encoded = y_resampled
            except Exception as e:
                logger.warning(f"Resampling failed. Error: {e}. Using original training fold.")
                
        df_out = X.copy()
        df_out[self.target_col] = y_encoded
        return df_out
```

And update the pipeline order in `train_pipeline.py` / `run_pipeline.py` to:
1. Split train/validation.
2. Apply feature engineering.
3. Fit & transform `FeatureSelector` on original training fold to select features.
4. Apply selected feature filter to validation/test folds.
5. Apply `DataPreprocessor` (resampling only training features).

*Note: To run the auxiliary regression head, the data splitting pipeline must be modified to keep the raw continuous target, e.g. saving it in a column named `G3_raw` in the CSVs, rather than overwriting `G3` directly.*

---

### 5.2 Model Architecture `StudentHybridV27` (`src/models_v27.py`)
The model fuses the Sequence and Context representations using a gating layer and includes auxiliary outputs:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling1D(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        attention_hidden = max(8, hidden_dim // 2)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )

    def forward(self, sequence: torch.Tensor):
        weights = torch.softmax(self.score(sequence), dim=1)
        pooled = torch.sum(sequence * weights, dim=1)
        return pooled, weights

class GatedFusion(nn.Module):
    """Dynamic Gated Fusion of Sequential and Contextual vectors."""
    def __init__(self, seq_dim: int, ctx_dim: int, fusion_dim: int, dropout: float = 0.3):
        super().__init__()
        self.proj_seq = nn.Sequential(
            nn.Linear(seq_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_ctx = nn.Sequential(
            nn.Linear(ctx_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )

    def forward(self, h_seq: torch.Tensor, h_ctx: torch.Tensor) -> torch.Tensor:
        h_seq_proj = self.proj_seq(h_seq)
        h_ctx_proj = self.proj_ctx(h_ctx)
        g = self.gate(torch.cat([h_seq_proj, h_ctx_proj], dim=1))
        return g * h_seq_proj + (1.0 - g) * h_ctx_proj

class StudentHybridV27(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_in_channels: int,
        num_numerical: int,
        cat_cardinalities: list[int],
        cnn_channels: int = 32,
        cnn_kernel_size: int = 3,
        lstm_hidden_dim: int = 64,
        context_hidden_dim: int = 64,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3,
        embedding_dim: int | None = None,
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities

        # Categorical Embeddings
        self.embeddings = nn.ModuleList()
        embedding_total_dim = 0
        for cardinality in cat_cardinalities:
            dim = embedding_dim if embedding_dim else max(2, min(50, (cardinality + 1) // 2))
            self.embeddings.append(nn.Embedding(num_embeddings=cardinality, embedding_dim=dim))
            embedding_total_dim += dim

        # Sequence Branch (Conv1D + BiLSTM)
        self.sequence_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=seq_in_channels,
                out_channels=cnn_channels,
                kernel_size=cnn_kernel_size,
                padding=cnn_kernel_size // 2,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sequence_bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        seq_out_dim = lstm_hidden_dim * 2
        self.sequence_pool = AttentionPooling1D(seq_out_dim)

        # Context Branch (Embeddings + Context MLP)
        context_input_dim = num_numerical + embedding_total_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(max(1, context_input_dim), context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_hidden_dim, context_hidden_dim),
            nn.ReLU(),
        )

        # Fusion (Gated Fusion)
        self.gated_fusion = GatedFusion(
            seq_dim=seq_out_dim,
            ctx_dim=context_hidden_dim,
            fusion_dim=fusion_hidden_dim,
            dropout=dropout
        )

        # Output Heads
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)
        self.ordinal_head = nn.Linear(fusion_hidden_dim, num_classes - 1)
        self.regression_head = nn.Linear(fusion_hidden_dim, 1)

    def _prepare_context(self, num_x: torch.Tensor | None, cat_x: torch.Tensor | None, batch_size: int, device: torch.device):
        parts = []
        if self.num_numerical > 0 and num_x is not None:
            parts.append(num_x[:, :self.num_numerical].float())

        if self.cat_cardinalities and cat_x is not None:
            embedded_categorical = []
            for index, emb_layer in enumerate(self.embeddings):
                values = cat_x[:, index].long()
                values = torch.clamp(values, 0, self.cat_cardinalities[index] - 1)
                embedded_categorical.append(emb_layer(values))
            parts.append(torch.cat(embedded_categorical, dim=1))

        if not parts:
            return torch.zeros(batch_size, 1, device=device)
        return torch.cat(parts, dim=1)

    def forward(self, seq_x: torch.Tensor, num_x: torch.Tensor | None, cat_x: torch.Tensor | None):
        # 1. Sequence Branch
        sequence = seq_x.float().transpose(1, 2)
        sequence = self.sequence_cnn(sequence).transpose(1, 2)
        sequence, _ = self.sequence_bilstm(sequence)
        sequence_vector, _ = self.sequence_pool(sequence)

        # 2. Context Branch
        context = self._prepare_context(num_x, cat_x, seq_x.shape[0], seq_x.device)
        context_vector = self.context_mlp(context)

        # 3. Gated Fusion
        fused = self.gated_fusion(sequence_vector, context_vector)

        # 4. Multi-head output
        logits_cls = self.classifier(fused)
        logits_ord = self.ordinal_head(fused)
        pred_reg = self.regression_head(fused).squeeze(-1)

        return logits_cls, logits_ord, pred_reg
```

---

### 5.3 Loss Functions (`src/losses_v27.py`)
The loss functions support multi-head joint optimization:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss based on Cui et al. (CVPR 2019)."""
    def __init__(self, samples_per_class: list[int], num_classes: int, beta: float = 0.99, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        samples_per_class = np.array(samples_per_class)
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        self.weight = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        weight = self.weight.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class OrdinalLoss(nn.Module):
    """Ordinal loss for ordinal classification heads."""
    def __init__(self, num_classes: int, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # inputs shape: (batch_size, num_classes - 1)
        # targets shape: (batch_size,) with values in [0, num_classes-1]
        batch_size = targets.size(0)
        
        ordinal_labels = []
        for k in range(self.num_classes - 1):
            ordinal_labels.append((targets > k).float())
        ordinal_labels = torch.stack(ordinal_labels, dim=1) # (batch_size, num_classes - 1)
        
        loss = self.bce(inputs, ordinal_labels)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class JointHybridLoss(nn.Module):
    """Joint multi-head optimization loss combining classification, ordinal, and regression."""
    def __init__(self, cls_criterion, ord_criterion, w_cls=1.0, w_ord=0.5, w_reg=0.2):
        super().__init__()
        self.cls_criterion = cls_criterion
        self.ord_criterion = ord_criterion
        self.reg_criterion = nn.MSELoss()
        
        self.w_cls = w_cls
        self.w_ord = w_ord
        self.w_reg = w_reg

    def forward(self, preds, targets_cls, targets_reg=None):
        # preds: tuple (logits_cls, logits_ord, pred_reg)
        logits_cls, logits_ord, pred_reg = preds
        
        loss_cls = self.cls_criterion(logits_cls, targets_cls)
        loss_ord = self.ord_criterion(logits_ord, targets_cls)
        
        total_loss = self.w_cls * loss_cls + self.w_ord * loss_ord
        
        if targets_reg is not None and self.w_reg > 0:
            loss_reg = self.reg_criterion(pred_reg, targets_reg.float())
            total_loss += self.w_reg * loss_reg
            
        return total_loss
```

---

## 6. Verification Method

To independently verify these conclusions and implementation, perform the following:

1. **Verify Scaling and Preprocessing Isolation**:
   - Inspect the training script (`scripts/run_pipeline.py`). Search for `fit_transform` calls. Verify that no `preprocessor.fit_transform` or `selector.fit_transform` contains validation samples `val_sub` or test samples `test_prep`.
   - Command:
     ```powershell
     Select-String -Path .\scripts\run_pipeline.py -Pattern "fit_transform"
     ```
     Ensure only `train_sub` and `train_prep` (never `val_sub` or `test_prep`) are arguments to `fit_transform`.

2. **Verify Feature Selection Order**:
   - Verify that `FeatureSelector` receives the *un-resampled* `train_sub` data or that it runs before the oversampling step. Currently, in `train_pipeline.py` Line 283, `train_selected = selector.fit_transform(train_prep, ...)` uses `train_prep` which is already resampled. This is a visual confirmation of the leakage.

3. **Verify ADASYN categorical error**:
   - Check that `student` dataset runs inside `train_pipeline.py` (Line 228) and uses `adasyn`. Run the pipeline on student-mat:
     ```powershell
     python scripts/run_pipeline.py --dataset student-mat --debug
     ```
     Verify that the generated training samples in `train_prep` have floating-point numbers in the categorical columns before casting, or check the database values for synthetic categories.

4. **Verify Hybrid Model Construction**:
   - Once implemented, write a test script that feeds dummy tensors `seq_x` of shape `(2, 2, 1)`, `num_x` of shape `(2, 5)`, and `cat_x` of shape `(2, 3)` into `StudentHybridV27` and verifies the output shapes of the three heads:
     - Classification: `(2, 3)`
     - Ordinal: `(2, 2)`
     - Regression: `(2,)`
