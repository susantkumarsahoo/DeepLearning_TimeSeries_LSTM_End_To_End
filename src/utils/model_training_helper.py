import pandas as pd
import numpy as np
from src.constants.paths import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



# ============================================
# STEP 4: CREATE SEQUENCES AFTER SCALING
# ============================================
def create_sequences(X, y, seq_length=seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X.iloc[i:i+seq_length].values)
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)



import numpy as np
import pandas as pd
import json

def split_dataset_report(
    X_train_scaled, X_test_scaled,
    y_train_scaled, y_test_scaled,
    X_train_seq, y_train_seq,
    X_test_seq, y_test_seq
):
    import numpy as np
    import pandas as pd

    def get_info(name, data):
        """Extract safe dataset info without crashing on non-numeric values."""
        
        # Convert DataFrame/Series to ndarray
        if isinstance(data, (pd.DataFrame, pd.Series)):
            arr = data.values
        else:
            arr = np.array(data)

        # Try converting to float
        try:
            numeric_arr = arr.astype(float)
            is_numeric = True
        except Exception:
            numeric_arr = None
            is_numeric = False

        info = {
            "name": name,
            "type": str(type(data)),
            "shape": arr.shape,
            "dtype": str(arr.dtype),
        }

        if is_numeric and numeric_arr.size > 0:
            info.update({
                "min": float(np.nanmin(numeric_arr)),
                "max": float(np.nanmax(numeric_arr)),
                "mean": float(np.nanmean(numeric_arr)),
                "std": float(np.nanstd(numeric_arr)),
                "nan_count": int(np.isnan(numeric_arr).sum())
            })
        else:
            # Non-numeric fallback
            info.update({
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "nan_count": None,
                "note": "Non-numeric data — statistics skipped"
            })

        return info

    datasets = [
        get_info("X_train_scaled", X_train_scaled),
        get_info("X_test_scaled", X_test_scaled),
        get_info("y_train_scaled", y_train_scaled),
        get_info("y_test_scaled", y_test_scaled),
        get_info("X_train_seq", X_train_seq),
        get_info("y_train_seq", y_train_seq),
        get_info("X_test_seq", X_test_seq),
        get_info("y_test_seq", y_test_seq),
    ]

    return datasets



def build_and_train_lstm(
    X_train_seq,
    y_train_seq,
    X_test_seq,
    y_test_seq,
    seq_length,
    epochs=100,
    batch_size=64
):
    """
    Builds, compiles, trains an optimized LSTM model and returns both
    the trained model and the training history report.

    Parameters
    ----------
    X_train_seq : ndarray
    y_train_seq : ndarray
    X_test_seq : ndarray
    y_test_seq : ndarray
    seq_length : int
        Sequence length for input_shape.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Batch size.

    Returns
    -------
    model : keras.Model
        Trained Keras model.
    report : dict
        Training history including loss/val_loss etc.
    """


    # ================================
    # BUILD MODEL
    # ================================
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, X_train_seq.shape[2])),
        Dropout(0.3),

        LSTM(64, return_sequences=True),
        Dropout(0.3),

        LSTM(32, return_sequences=False),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # ================================
    # CALLBACKS
    # ================================
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # ================================
    # TRAIN MODEL
    # ================================
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ================================
    # RETURN MODEL + REPORT
    # ================================
    report = history.history
    return model, report


import pandas as pd

def prepare_datetime_index(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts the specified date column to datetime, sets it as index,
    and returns clean train/test dataframes.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    test_df : pd.DataFrame
        Testing dataset.
    date_col : str
        Column name to convert to datetime and set as index.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Processed train and test dataframes with datetime index.
    """

    # Convert to datetime
    train_df[date_col] = pd.to_datetime(train_df[date_col], errors="coerce")
    test_df[date_col]  = pd.to_datetime(test_df[date_col], errors="coerce")

    # Set index
    train_df = train_df.set_index(date_col).sort_index()
    test_df  = test_df.set_index(date_col).sort_index()

    return train_df, test_df



























