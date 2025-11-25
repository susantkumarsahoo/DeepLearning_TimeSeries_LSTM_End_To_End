import json
import pandas as pd
import numpy as np

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

def generate_correlation_report(df: pd.DataFrame, output_path: str = None) -> dict:
    """
    Generates a correlation metrics report:
    1. Strong correlations (>|0.80|)
    2. Very weak correlations (< 0.01)

    Optionally saves the report as JSON.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    output_path : str, optional
        File path to save JSON report.

    Returns
    -------
    dict
        Correlation summary report.
    """

    # Compute correlation matrix
    corr_matrix = df.corr(numeric_only=True)

    strong_corr = []
    weak_corr = []

    # Iterate through upper triangle only (avoid duplicate pairs)
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            value = corr_matrix.iloc[i, j]

            # Strong correlation
            if abs(value) >= 0.80:
                strong_corr.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(float(value), 4)
                })

            # Very weak correlation
            if abs(value) <= 0.01:
                weak_corr.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(float(value), 4)
                })

    # Prepare report
    report = {
        "strong_correlations_(>|0.80|)": strong_corr,
        "very_weak_correlations_(<0.01)": weak_corr
    }

    # Save JSON (if required)
    #if output_path is not None:
    #    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #    with open(output_path, "w") as json_file:
    #       json.dump(report, json_file, indent=4)



    return report



import numpy as np
import pandas as pd
import json
import os

def detect_outliers_iqr(df: pd.DataFrame, column: str, json_output_path: str = None):
    """
    Detect outliers using the IQR method and return:
      - Outlier report (dict)
      - Optionally save report to JSON

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column for outlier detection.
    json_output_path : str, optional
        Path to save the JSON outlier report.

    Returns
    -------
    report : dict
        Dictionary containing outlier detection summary.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    # Convert to NumPy array
    data_array = df[column].dropna().to_numpy()

    # Compute Q1, Q3 and IQR
    Q1 = np.percentile(data_array, 25)
    Q3 = np.percentile(data_array, 75)
    IQR = Q3 - Q1

    # Outlier limits
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Outliers
    outliers = data_array[(data_array < lower_bound) | (data_array > upper_bound)]

    # Prepare JSON Report
    report = {
        "column": column,
        "total_records": int(len(data_array)),
        "Q1": float(Q1),
        "Q3": float(Q3),
        "IQR": float(IQR),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "total_outliers": int(len(outliers)),
    
    }

    return report



import pandas as pd
import numpy as np
import json
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime

def analyze_seasonal_decomposition(df, column_name, date_column=None, model='additive', period=12):
    """
    Perform seasonal decomposition on time series data and return JSON report.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time series data
    column_name : str
        Name of the column to analyze
    date_column : str, optional
        Name of the date column to use as index (if not already indexed)
    model : str, default='additive'
        Type of seasonal component ('additive' or 'multiplicative')
    period : int, default=12
        Number of observations per cycle
        
    Returns:
    --------
    dict : JSON-formatted report with decomposition results
    """
    
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Validate inputs
    if column_name not in df_copy.columns:
        return {"error": f"Column '{column_name}' not found in DataFrame"}
    
    # Convert index to datetime if needed
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        if date_column is not None:
            if date_column not in df_copy.columns:
                return {"error": f"Date column '{date_column}' not found in DataFrame"}
            try:
                df_copy[date_column] = pd.to_datetime(df_copy[date_column])
                df_copy = df_copy.set_index(date_column)
                df_copy = df_copy.sort_index()
            except Exception as e:
                return {"error": f"Failed to convert '{date_column}' to datetime: {str(e)}"}
        else:
            # Try to find a date column automatically
            date_cols = [col for col in df_copy.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                try:
                    df_copy[date_cols[0]] = pd.to_datetime(df_copy[date_cols[0]])
                    df_copy = df_copy.set_index(date_cols[0])
                    df_copy = df_copy.sort_index()
                except Exception as e:
                    return {"error": f"Auto-detected date column '{date_cols[0]}' conversion failed. Please specify 'date_column' parameter: {str(e)}"}
            else:
                return {"error": "No DatetimeIndex found. Please specify 'date_column' parameter or ensure DataFrame has a datetime index"}
    
    # Extract time series
    ts = df_copy[column_name].dropna()
    
    if len(ts) < 2 * period:
        return {"error": f"Insufficient data points. Need at least {2*period}, got {len(ts)}"}
    
    # Perform decomposition
    try:
        result = seasonal_decompose(ts, model=model, period=period)
    except Exception as e:
        return {"error": f"Decomposition failed: {str(e)}"}
    
    # Extract components
    trend = result.trend.dropna()
    seasonal = result.seasonal.dropna()
    residual = result.resid.dropna()
    
    # Calculate statistics
    report = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "column_analyzed": column_name,
            "model_type": model,
            "period": period,
            "total_observations": len(ts),
            "date_range": {
                "start": ts.index.min().isoformat(),
                "end": ts.index.max().isoformat()
            }
        },
        "original_series": {
            "mean": float(ts.mean()),
            "std": float(ts.std()),
            "min": float(ts.min()),
            "max": float(ts.max()),
            "median": float(ts.median())
        },
        "trend_component": {
            "mean": float(trend.mean()),
            "std": float(trend.std()),
            "min": float(trend.min()),
            "max": float(trend.max()),
            "non_null_count": int(len(trend)),
            "trend_direction": "increasing" if trend.iloc[-1] > trend.iloc[0] else "decreasing",
            "change_percent": float(((trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0]) * 100) if trend.iloc[0] != 0 else None
        },
        "seasonal_component": {
            "mean": float(seasonal.mean()),
            "std": float(seasonal.std()),
            "min": float(seasonal.min()),
            "max": float(seasonal.max()),
            "amplitude": float(seasonal.max() - seasonal.min()),
            "non_null_count": int(len(seasonal))
        },
        "residual_component": {
            "mean": float(residual.mean()),
            "std": float(residual.std()),
            "min": float(residual.min()),
            "max": float(residual.max()),
            "non_null_count": int(len(residual)),
            "outliers_count": int(np.sum(np.abs(residual) > 3 * residual.std()))
        },
        "decomposition_quality": {
            "residual_variance_ratio": float((residual.var() / ts.var()) * 100),
            "seasonal_strength": float(1 - (residual.var() / (seasonal + residual).var())) if (seasonal + residual).var() != 0 else None,
            "trend_strength": float(1 - (residual.var() / (trend + residual).var())) if (trend + residual).dropna().var() != 0 else None
        },
        "sample_data": {
            "trend_head": trend.head(5).to_dict(),
            "seasonal_head": seasonal.head(5).to_dict(),
            "residual_head": residual.head(5).to_dict()
        }
    }
    
    return report




import os
import json
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def variance_threshold_report(df_clean: pd.DataFrame, threshold: float, json_path: str):
    """
    Perform Variance Threshold feature selection and save report as JSON.
    The internal logic and steps remain exactly as the user provided.
    """

    # ----- STEP 1: Select numeric features -----
    dfnumeric = df_clean.select_dtypes(include=[np.number])

    # ----- STEP 2: Variance calculation -----
    feature_variances = dfnumeric.var()

    # ----- STEP 3: Apply Variance Threshold -----
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(dfnumeric)

    selected_features = dfnumeric.columns[selector.get_support()].tolist()
    dropped_features = [
        col for col in dfnumeric.columns if col not in selected_features
    ]

    # ----- STEP 4: Create filtered DataFrame -----
    df_selected = dfnumeric[selected_features]

    # ----- STEP 7: Create metadata JSON -----
    metadata = {
        "threshold": threshold,
        "original_shape": list(dfnumeric.shape),
        "reduced_shape": list(df_selected.shape),
        "selected_features": selected_features,
        "dropped_features": dropped_features,
        "feature_variances": feature_variances.to_dict(),
        "columns_features": list(df_clean.columns)
    }


    return metadata


import json
import os

def save_readable_report(final_report: dict, output_path=None):

    lines = []

    for section_name, section_value in final_report.items():
        lines.append("=" * 60)
        lines.append(section_name.upper())
        lines.append("=" * 60)
        lines.append(json.dumps(section_value, indent=4))
        lines.append("\n\n")

    content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path



import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def multicollinearity_vif_report(df_clean, target_column='megawatthours'):
    """
    Run VIF multicollinearity analysis and return a JSON-ready report.
    No code logic changed — only wrapped into a function.
    """

    # ----- STEP 1: Prepare Data -----
    df_model = df_clean.copy()

    V = df_model.drop(columns=[target_column], errors='ignore')
    I = df_model[target_column] if target_column in df_model.columns else None

    V = V.select_dtypes(include=[np.number]).dropna()
    V_const = add_constant(V)

    print(f"\nFeatures selected for VIF calculation (excluding target '{target_column}'):")
    print(list(V.columns))

    # ----- STEP 2: Compute VIF -----
    vif_data = pd.DataFrame()
    vif_data["Feature"] = V_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(V_const.values, i)
        for i in range(V_const.shape[1])
    ]

    vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)
    print("\nVariance Inflation Factor (VIF) Results:\n")
    print(vif_data)

    # ----- STEP 4: Identify high-VIF features -----
    high_vif = vif_data[vif_data["VIF"] > 10]
    if not high_vif.empty:
        print("\nHigh multicollinearity detected (VIF > 10):")
        print(high_vif)
    else:
        print("\nNo critical multicollinearity detected (VIF ≤ 10).")

    # ----- RETURN JSON REPORT -----
    report = {
        "vif_results": vif_data.to_dict(orient="records"),
        "high_vif_features": high_vif.to_dict(orient="records"),
        "correlation_matrix_vif": V.corr().round(4).to_dict(),
        "total_features": len(V.columns),
        "high_vif_count": len(high_vif),
        "status": "High multicollinearity detected" if len(high_vif) > 0 else "Healthy"
    }

    return report




def anova_f_test_report(df_clean, target_column="megawatthours"):
    """
    Run ANOVA F-Test feature selection and return a JSON-ready report.
    No code logic changed — only wrapped into a function.
    """

    import pandas as pd
    import numpy as np
    from sklearn.feature_selection import SelectKBest, f_classif
    import matplotlib.pyplot as plt
    import seaborn as sns

    # -------------------------------
    # STEP 1: Load Data
    # -------------------------------
    anova_X = df_clean.drop(columns=[target_column], errors='ignore')
    anova_y = df_clean[target_column]

    X_numeric = anova_X.select_dtypes(include=[np.number]).dropna()

    # -------------------------------
    # STEP 2: Apply ANOVA F-Test
    # -------------------------------
    selector = SelectKBest(score_func=f_classif, k='all')
    X_new = selector.fit_transform(X_numeric, anova_y)

    anova_scores = selector.scores_
    anova_pvalues = selector.pvalues_

    anova_df = pd.DataFrame({
        'Feature': X_numeric.columns,
        'F_Score': anova_scores,
        'P_Value': anova_pvalues
    }).sort_values(by='F_Score', ascending=False).reset_index(drop=True)

    # -------------------------------
    # STEP 4: Filter Important Features (Optional)
    # -------------------------------
    significant_features = anova_df[anova_df['P_Value'] < 0.05]['Feature']
    print("\nSignificant Features (p < 0.05):")
    print(significant_features.tolist())

    # -------------------------------
    # STEP 5: Return JSON report
    # -------------------------------
    report = {
        "anova_results": anova_df.to_dict(orient="records"),
        "significant_features": significant_features.tolist(),
        "total_features": len(anova_df),
        "significant_feature_count": len(significant_features),
        "status": "Significant features found" if len(significant_features) > 0 else "No significant features"
    }

    return report

import pandas as pd
def remove_outliers_iqr(df, column, report_path=None):
    """
    Remove outliers from a DataFrame column using the IQR method.
    """

    # Validation
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric.")
    
    if df[column].isnull().all():
        raise ValueError(f"Column '{column}' contains only null values.")

    # IQR Calculation
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    df_clean.set_index('date', inplace=True)

    df_shape = df.shape
    df_clean_sapes = df_clean.shape
    column_name = df_clean.columns

    report = {
        "column_name": column_name,
        "df_shape": df_shape,
        "df_clean_shape": df_clean_sapes,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "IQR": IQR,
        "Q1": Q1,
        "Q3": Q3
    }


    return df_clean, report










