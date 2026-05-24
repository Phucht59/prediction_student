import pandas as pd


def clean_columns(data: pd.DataFrame) -> pd.DataFrame:
    clean_data = data.copy()
    clean_data.columns = [
        str(column).strip().replace(" ", "_").replace("-", "_")
        for column in clean_data.columns
    ]
    return clean_data


def remove_empty_rows(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna(how="all").reset_index(drop=True)

