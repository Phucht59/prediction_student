from pathlib import Path

import pandas as pd


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        raise ValueError("This model does not provide feature importance.")

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False)


def save_feature_importance(importance: pd.DataFrame, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)

