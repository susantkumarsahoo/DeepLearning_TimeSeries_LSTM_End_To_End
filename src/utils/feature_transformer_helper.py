import json
import os
import pandas as pd

def drop_time_features(df: pd.DataFrame, output_json_path: str):
    """
    Drops predefined time-related columns and saves a deterministic (DVC-safe) JSON report.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    output_json_path : str
        Destination path for the cleaning report.

    Returns
    -------
    tuple: (clean_df, report_dict)
    """

    cols_to_drop = [
        "hour",
        "day_of_week",
        "day_of_month",
        "day_of_year"
    ]

    initial_shape = df.shape
    initial_columns = list(df.columns)

    # Detect columns that exist
    existing_cols = [c for c in cols_to_drop if c in df.columns]

    # Drop columns
    df_clean = df.drop(columns=existing_cols, errors='ignore')

    # Build deterministic report (NO TIMESTAMP)
    report = {
        "initial_shape": initial_shape,
        "initial_columns": initial_columns,
        "columns_targeted": cols_to_drop,
        "columns_removed": existing_cols,
        "final_shape": df_clean.shape,
        "final_columns": list(df_clean.columns),
    }

    return df_clean, report

