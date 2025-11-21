import json
import pandas as pd

def generate_data_profile(df: pd.DataFrame, save_path: str = None):
    """
    Generate a full data profile summary and optionally save it as a JSON file.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset for analysis.
    save_path : str, optional
        File path to save JSON output. Example: "artifacts/data_profile.json"

    Returns
    -------
    dict
        JSON-serializable dictionary containing dataset profile.
    """

    profile = {
        "basic_information": {
            "shape": df.shape,
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
        },
        "data_quality": {
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum())
        },
        "core_validation": {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "data_types": df.dtypes.astype(str).to_dict(),
            "column_names": df.columns.tolist()
        }
    }

    # Save JSON file if path given
    if save_path:
        with open(save_path, "w") as f:
            json.dump(profile, f, indent=4)

    return profile



from sklearn.model_selection import train_test_split
import pandas as pd

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split a DataFrame into train and test datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    test_size : float, default=0.2
        Size of the test dataset.
    random_state : int, default=42
        Random state for reproducibility.
    
    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    return train_df, test_df



import pandas as pd

def add_time_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Convert date column to datetime and generate core time-based features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    date_column : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added time features.
    """

    # Convert to datetime
    df[date_column] = pd.to_datetime(
        df[date_column],
        format='%d-%m-%Y %H:%M',
        errors='coerce'
    )

    # Generate time features
    df["hour"] = df[date_column].dt.hour
    df["day_of_week"] = df[date_column].dt.dayofweek
    df["day_of_month"] = df[date_column].dt.day
    df["day_of_year"] = df[date_column].dt.dayofyear

    return df


import pandas as pd
import json
import os

def generate_correlation_report(df: pd.DataFrame, output_path: str) -> dict:
    """
    Generates a correlation metrics report identifying:
    1. Strong correlations (> 0.80 or < -0.80)
    2. Weak correlations (< 0.01 and > -0.01)
    
    Saves the report as a JSON file and returns the report dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    output_path : str
        Path where the JSON report will be saved.

    Returns
    -------
    dict
        Correlation metrics summary.
    """

    # Compute correlation matrix
    corr_matrix = df.corr(numeric_only=True)

    strong_corr = []
    weak_corr = []

    # Identify correlations
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2:
                value = corr_matrix.loc[col1, col2]

                # Strong correlation > 0.80 or < -0.80
                if abs(value) >= 0.80:
                    strong_corr.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(value), 4)
                    })

                # Very weak correlation < 0.01
                if abs(value) <= 0.01:
                    weak_corr.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(value), 4)
                    })

    # Prepare JSON report
    report = {
        "strong_correlations_(>|0.80|)": strong_corr,
        "very_weak_correlations_(<0.01)": weak_corr
    }
    return df, report


import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt


def detect_outliers_iqr(df: pd.DataFrame, column: str, json_output_path: str):
    """
    Detect outliers using IQR method and return:
      - Cleaned dataframe (without outliers)
      - Detailed outlier report saved as JSON

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column on which outlier detection will be applied.
    json_output_path : str
        Path for saving the JSON report.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Dataframe without outliers.
    report : dict
        Dictionary containing outlier detection summary.
    """
    column = 'megawatthours'
    # Convert to NumPy array
    data_array = df[column].to_numpy()

    # Compute Q1, Q3 and IQR
    Q1 = np.percentile(data_array, 25)
    Q3 = np.percentile(data_array, 75)
    IQR = Q3 - Q1

    # Outlier limits
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Outliers
    outliers = data_array[(data_array < lower_bound) | (data_array > upper_bound)]

    # Create cleaned dataframe
    # cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # --------------------------
    # Prepare JSON Report
    # --------------------------
    report = {
        "column": column,
        "total_records": int(len(data_array)),
        "Q1": float(Q1),
        "Q3": float(Q3),
        "IQR": float(IQR),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "total_outliers": int(len(outliers)),
        "outlier_values": outliers.tolist(),
    }

    return df, report



