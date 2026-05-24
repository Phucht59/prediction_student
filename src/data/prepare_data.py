import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data.clean_data import clean_columns, remove_empty_rows
from src.data.load_data import load_dataset
from src.utils.config import load_config, make_path


def create_grade_class(grade: float) -> str:
    if grade <= 9:
        return "low"
    if grade <= 14:
        return "medium"
    return "high"


def create_target_labels(data: pd.DataFrame, dataset_name: str, target_column: str) -> pd.Series:
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found in dataset '{dataset_name}'.")

    if dataset_name in ["student_math", "student_por"]:
        grades = pd.to_numeric(data[target_column], errors="coerce")
        return grades.apply(create_grade_class)

    if dataset_name == "xapi":
        label_map = {"L": "low", "M": "medium", "H": "high"}
        labels = data[target_column].astype(str).str.strip()
        return labels.replace(label_map)

    raise ValueError(f"No target label rule defined for dataset '{dataset_name}'.")


def encode_features(features: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(features, drop_first=False, dtype=int)


def scale_numeric_columns(
    encoded_features: pd.DataFrame,
    original_features: pd.DataFrame,
    scaling_method: str,
) -> pd.DataFrame:
    scaled_features = encoded_features.copy()
    numeric_columns = original_features.select_dtypes(include=["number"]).columns
    numeric_columns = [column for column in numeric_columns if column in scaled_features.columns]

    if not numeric_columns:
        return scaled_features

    if scaling_method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaled_features[numeric_columns] = scaler.fit_transform(scaled_features[numeric_columns])
    return scaled_features


def prepare_dataset(dataset_name: str, config_path: str = "config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    dataset_config = config["datasets"][dataset_name]

    data = load_dataset(dataset_name, config_path)
    data = clean_columns(data)
    data = remove_empty_rows(data)

    target_column = dataset_config["target_column"]
    labels = create_target_labels(data, dataset_name, target_column)

    features = data.drop(columns=[target_column])
    features = features.fillna(features.mode().iloc[0])
    encoded_features = encode_features(features)
    scaled_features = scale_numeric_columns(
        encoded_features,
        features,
        config.get("scaling_method", "standard"),
    )

    prepared_data = scaled_features.copy()
    prepared_data["target"] = labels

    output_path = make_path(dataset_config["processed_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_data.to_csv(output_path, index=False)

    return prepared_data

