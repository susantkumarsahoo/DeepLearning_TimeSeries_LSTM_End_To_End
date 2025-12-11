import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
import sys

from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException

logger = get_logger(__name__)

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
        DataFrame with 'ds' column and all required time & cyclic features.
    """
    try:
        # Convert input dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Create hourly date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')

        # Base DataFrame
        df = pd.DataFrame({'ds': date_range})

        # ---- Core time-based features ----
        df['hour'] = df['ds'].dt.hour
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['day_of_month'] = df['ds'].dt.day
        df['day_of_year'] = df['ds'].dt.dayofyear

        # ---- Cyclic features ----
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['dayofweek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['dayofmonth_sin'] = np.sin(2 * np.pi * (df['day_of_month'] - 1) / 31)
        df['dayofmonth_cos'] = np.cos(2 * np.pi * (df['day_of_month'] - 1) / 31)

        df['dayofyear_sin'] = np.sin(2 * np.pi * (df['day_of_year'] - 1) / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * (df['day_of_year'] - 1) / 365)

        # ---- Drop raw columns but keep 'ds' ----
        df.drop(columns=['hour', 'day_of_week', 'day_of_month', 'day_of_year'], inplace=True)

        logger.info(f"Generated {len(df)} forecast features from {start_date} to {end_date}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error generating forecast features: {str(e)}")
        raise ProjectException(e, sys)


def generate_predictions(
    model,
    seq_df: pd.DataFrame,
    scaler_y,
    forecast_features: pd.DataFrame,
    seq_length: int
) -> pd.DataFrame:
    """
    Generate predictions using the trained model.
    
    Args:
        model: Trained LSTM model
        seq_df: DataFrame with sequences
        scaler_y: Scaler for inverse transformation
        forecast_features: Original forecast features with 'ds' date column
        seq_length: Sequence length used
        
    Returns:
        DataFrame with dates and predictions
    """
    try:
        logger.info("Generating predictions from sequences")
        
        # Extract sequences array
        sequences = np.array(seq_df['sequences'].tolist())
        
        logger.info(f"Sequences shape: {sequences.shape}")
        
        # Generate predictions
        predictions_scaled = model.predict(sequences, verbose=0)
        
        logger.info(f"Raw predictions shape: {predictions_scaled.shape}")
        
        # Inverse transform predictions
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        # Flatten predictions if needed
        predictions_flat = predictions.flatten()
        
        logger.info(f"Predictions length: {len(predictions_flat)}")
        logger.info(f"Total forecast days: {len(forecast_features)}")
        logger.info(f"Sequence length: {seq_length}")
        
        # CRITICAL FIX: Correct date alignment
        # When we create sequences, we have (len(data) - seq_length + 1) sequences
        # Each sequence uses seq_length consecutive time steps to predict the NEXT time step
        # So sequence[0] uses data[0:seq_length] to predict data[seq_length]
        # Therefore, predictions start from index seq_length (not seq_length-1)
        
        expected_predictions = len(forecast_features) - seq_length
        
        logger.info(f"Expected predictions: {expected_predictions}")
        logger.info(f"Actual predictions: {len(predictions_flat)}")
        
        # Adjust predictions length if needed
        if len(predictions_flat) > expected_predictions:
            logger.warning(f"Trimming predictions from {len(predictions_flat)} to {expected_predictions}")
            predictions_flat = predictions_flat[:expected_predictions]
        elif len(predictions_flat) < expected_predictions:
            logger.warning(f"Predictions shortage: {len(predictions_flat)} vs {expected_predictions}")
        
        # Get corresponding dates (starting from seq_length index)
        # First prediction corresponds to forecast_features[seq_length]
        prediction_dates = forecast_features['ds'].iloc[seq_length:seq_length+len(predictions_flat)].values
        
        logger.info(f"Prediction dates length: {len(prediction_dates)}")
        logger.info(f"Final predictions length: {len(predictions_flat)}")
        
        # Ensure lengths match exactly
        min_length = min(len(predictions_flat), len(prediction_dates))
        predictions_flat = predictions_flat[:min_length]
        prediction_dates = prediction_dates[:min_length]
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'ds': prediction_dates,
            'predicted_value': predictions_flat
        })
        
        # Add additional info
        pred_df['prediction_date'] = datetime.now()
        
        logger.info(f"Successfully generated {len(pred_df)} predictions")
        logger.info(f"Date range: {pred_df['ds'].min()} to {pred_df['ds'].max()}")
        
        return pred_df
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        logger.error(f"Sequences shape: {sequences.shape if 'sequences' in locals() else 'Not created'}")
        logger.error(f"Predictions shape: {predictions_scaled.shape if 'predictions_scaled' in locals() else 'Not created'}")
        raise ProjectException(e, sys)


def create_sequence_df(forecast_features: pd.DataFrame, seq_length: int) -> pd.DataFrame:
    """
    Create sequences for LSTM model prediction.
    
    Args:
        forecast_features: DataFrame with forecast features (must include 'ds' column)
        seq_length: Sequence length for LSTM
        
    Returns:
        DataFrame with sequences prepared for prediction
    """
    try:
        logger.info(f"Creating sequences with length {seq_length}")
        
        if len(forecast_features) < seq_length:
            raise ValueError(f"Not enough data. Need at least {seq_length} rows, got {len(forecast_features)}")
        
        # Select feature columns (exclude 'ds' date column)
        feature_cols = [col for col in forecast_features.columns if col != 'ds']
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Get feature array
        feature_array = forecast_features[feature_cols].values
        
        logger.info(f"Feature array shape: {feature_array.shape}")
        
        # Create sequences
        # Each sequence of length seq_length predicts the next value
        sequences = []
        for i in range(len(feature_array) - seq_length):
            seq = feature_array[i:i + seq_length]
            sequences.append(seq)
        
        logger.info(f"Created {len(sequences)} sequences")
        
        if len(sequences) == 0:
            raise ValueError(f"No sequences created. Check if data length ({len(feature_array)}) > seq_length ({seq_length})")
        
        # Convert to numpy array
        sequences_array = np.array(sequences)
        
        logger.info(f"Sequences array shape: {sequences_array.shape}")
        
        # Return as DataFrame with sequence info
        seq_df = pd.DataFrame({
            'sequence_idx': range(len(sequences)),
            'sequences': list(sequences_array)
        })
        
        return seq_df
        
    except Exception as e:
        logger.error(f"Error creating sequences: {str(e)}")
        raise ProjectException(e, sys)


    

# python src/utils/prediction_helper.py