
import pandas as pd
import numpy as np



def add_cyclic_features(df, report=None):
    """
    Add cyclic (periodic) time features: hour, day_of_week, day_of_month, day_of_year.
    Returns updated DataFrame and JSON report.
    """
    import numpy as np

    cyclic_df = df.copy()
    added_features = []

    # Hour (24-hour cycle)
    if "hour" in cyclic_df.columns:
        cyclic_df["hour_sin"] = np.sin(2 * np.pi * cyclic_df["hour"] / 24)
        cyclic_df["hour_cos"] = np.cos(2 * np.pi * cyclic_df["hour"] / 24)
        added_features.extend(["hour_sin", "hour_cos"])

    # Day of Week (7-day cycle)
    if "day_of_week" in cyclic_df.columns:
        cyclic_df["dayofweek_sin"] = np.sin(2 * np.pi * cyclic_df["day_of_week"] / 7)
        cyclic_df["dayofweek_cos"] = np.cos(2 * np.pi * cyclic_df["day_of_week"] / 7)
        added_features.extend(["dayofweek_sin", "dayofweek_cos"])

    # Day of Month (31-day cycle)
    if "day_of_month" in cyclic_df.columns:
        cyclic_df["dayofmonth_sin"] = np.sin(2 * np.pi * (cyclic_df["day_of_month"] - 1) / 31)
        cyclic_df["dayofmonth_cos"] = np.cos(2 * np.pi * (cyclic_df["day_of_month"] - 1) / 31)
        added_features.extend(["dayofmonth_sin", "dayofmonth_cos"])

    # Day of Year (365-day cycle)
    if "day_of_year" in cyclic_df.columns:
        cyclic_df["dayofyear_sin"] = np.sin(2 * np.pi * (cyclic_df["day_of_year"] - 1) / 365)
        cyclic_df["dayofyear_cos"] = np.cos(2 * np.pi * (cyclic_df["day_of_year"] - 1) / 365)
        added_features.extend(["dayofyear_sin", "dayofyear_cos"])

    # -----------------
    # JSON Report
    # -----------------
    report = {
        "total_rows": cyclic_df.shape[0],
        "total_columns_before": df.shape[1],
        "total_columns_after": cyclic_df.shape[1],
        "features_added": added_features,
        "added_feature_count": len(added_features)
    }

    return cyclic_df, report

