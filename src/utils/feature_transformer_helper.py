import json
from typing import List, Tuple
import pandas as pd


def clean_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Removes specified columns from train and test datasets and 
    generates a structured report summarizing the changes.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    test_df : pd.DataFrame
        Testing dataset.
    drop_cols : List[str]
        Columns to be removed from both datasets.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, dict]
        Cleaned training DataFrame, cleaned testing DataFrame, 
        and a metadata report dictionary.
    """
    # cols_to_drop = ["hour", "day_of_week", "day_of_month", "day_of_year"]

    # Initial metadata
    report = {
        "train_before_shape": train_df.shape,
        "test_before_shape": test_df.shape,
        "columns_removed": [],
        "columns_remaining": []
    }

    # Columns that actually exist
    existing_cols = [col for col in drop_cols if col in train_df.columns]
    report["columns_removed"] = existing_cols

    # Drop columns
    clean_train = train_df.drop(columns=existing_cols, errors="ignore")
    clean_test = test_df.drop(columns=existing_cols, errors="ignore")

    # Post-drop metadata
    report["train_after_shape"] = clean_train.shape
    report["test_after_shape"] = clean_test.shape
    report["columns_remaining"] = list(clean_train.columns)

    return clean_train, clean_test, report


