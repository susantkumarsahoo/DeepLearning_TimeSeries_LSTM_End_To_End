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



import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2, random_state=42, shuffle=True):
    """
    Splits dataset into train and test sets.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature matrix (independent variables).
    y : array-like or Series
        Target variable.
    test_size : float, optional (default=0.2)
        Proportion of dataset to include in test split.
    random_state : int, optional (default=42)
        Seed for reproducibility.
    shuffle : bool, optional (default=True)
        Whether to shuffle before splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays/DataFrames
        Split datasets.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Error during train-test split: {e}")



import pandas as pd

def load_train_test(train_path, test_path, target_column):
    """
    Load train and test CSV files and split into features (X) and target (y).
    
    Parameters
    ----------
    train_path : str
        Path to train.csv file
    test_path : str
        Path to test.csv file
    target_column : str
        Name of the target column in both datasets
    
    Returns
    -------
    X_train, y_train, X_test, y_test : DataFrames/Series
        Train and test splits
    """
    try:
        # Load CSVs
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Separate features and target
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        return X_train, y_train, X_test, y_test
    
    except Exception as e:
        raise ValueError(f"Error loading datasets: {e}")










