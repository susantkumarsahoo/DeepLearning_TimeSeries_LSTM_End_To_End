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
    def get_info(name, data):
        """Extract dataset info into a dictionary."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            arr = data.values
        else:
            arr = np.array(data)

        return {
            "name": name,
            "type": str(type(data)),
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(np.min(arr)) if arr.size > 0 else None,
            "max": float(np.max(arr)) if arr.size > 0 else None,
            "mean": float(np.mean(arr)) if arr.size > 0 else None,
            "std": float(np.std(arr)) if arr.size > 0 else None,
            "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else None
        }

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
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

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
