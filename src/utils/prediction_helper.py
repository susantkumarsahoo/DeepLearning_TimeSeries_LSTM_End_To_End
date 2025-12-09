import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_forecast_features(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate a feature-engineered datetime DataFrame for model prediction.
    
    Parameters
    ----------
    start_date : str  
        Start date in 'YYYY-MM-DD' format.
    end_date : str  
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame  
        DataFrame indexed by datetime with all required time & cyclic features.
    """

    # Convert input dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Create hourly date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')

    # Base DataFrame
    df = pd.DataFrame({'datetime': date_range})

    # ---- Core time-based features ----
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # ---- Cyclic features ----
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['dayofweek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df['dayofmonth_sin'] = np.sin(2 * np.pi * (df['day_of_month'] - 1) / 31)
    df['dayofmonth_cos'] = np.cos(2 * np.pi * (df['day_of_month'] - 1) / 31)

    df['dayofyear_sin'] = np.sin(2 * np.pi * (df['day_of_year'] - 1) / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * (df['day_of_year'] - 1) / 365)

    # ---- Drop raw columns ----
    df.drop(columns=['hour', 'day_of_week', 'day_of_month', 'day_of_year'], inplace=True)

    # ---- Set datetime index ----
    df.set_index('datetime', inplace=True)

    return df



import numpy as np
import pandas as pd

def create_sequence_df(df: pd.DataFrame, seq_length: int = 7) -> np.ndarray:
    """
    Convert a feature-engineered DataFrame into sequential 3D input 
    for LSTM / GRU / Deep Learning time-series models.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing time features (no target column).
    seq_length : int, default=7
        Number of time steps in each sequence.

    Returns
    -------
    np.ndarray
        3D array of shape (n_samples, seq_length, n_features)

    Raises
    ------
    ValueError
        If seq_length <= 0 or seq_length > len(df).
    """

    # --- Validation ---
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if seq_length <= 0:
        raise ValueError("seq_length must be a positive integer.")
    if seq_length > len(df):
        raise ValueError("seq_length cannot exceed number of rows in DataFrame.")

    # --- Copy to avoid side effects ---
    df = df.copy()

    # --- Build sequences ---
    total_rows = len(df)
    last_index = total_rows - seq_length + 1

    X_sequences = np.array([df.iloc[i : i + seq_length].values 
                            for i in range(last_index)])

    return X_sequences



import pandas as pd
import numpy as np

def generate_predictions(
        model,
        X_seq: np.ndarray,
        scaler_y,
        base_df: pd.DataFrame,
        seq_length: int
    ) -> pd.DataFrame:
    """
    Generate model predictions, inverse scale them, and align with datetime index.

    Parameters
    ----------
    model : object
        Trained Keras/TensorFlow model.
    X_seq : np.ndarray
        3D input data for prediction (samples, seq_length, features).
    scaler_y : object
        Fitted scaler used for target variable (e.g., MinMaxScaler).
    base_df : pd.DataFrame
        Source dataframe containing the datetime index.
    seq_length : int
        Number of time steps used to generate each sequence.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime with final predictions.
    """

    # ---- Model Prediction ----
    scaled_predictions = model.predict(X_seq, verbose=1)

    # ---- Inverse Scaling ----
    predictions = scaler_y.inverse_transform(scaled_predictions).flatten()

    # ---- Align Datetime Index ----
    start_idx = seq_length
    end_idx = start_idx + len(predictions)

    prediction_dates = base_df.index[start_idx:end_idx]

    # ---- Build Final DataFrame ----
    pred_df = pd.DataFrame(
        {
            "prediction": predictions
        },
        index=prediction_dates
    )

    pred_df.index.name = "datetime"

    return pred_df
