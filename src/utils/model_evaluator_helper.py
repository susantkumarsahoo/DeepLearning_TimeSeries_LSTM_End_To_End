import os
import pickle
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# -------------------------------------------------------------
# FUNCTION 1: Load Saved LSTM Model + Scaler
# -------------------------------------------------------------
def load_artifacts(model_path: str, scaler_path: str) -> tuple:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    model = load_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler_y = pickle.load(f)

    return model, scaler_y

# -------------------------------------------------------------
# FUNCTION 2: Make Predictions and Generate Performance Report
# -------------------------------------------------------------
def evaluate_lstm_model(
    model,
    scaler_y,
    X_train_seq,
    X_test_seq,
    y_train_seq,
    y_test_seq
):
    """
    Evaluate LSTM model and return comprehensive performance report in JSON format.
    
    Parameters:
    -----------
    model : keras.Model
        Trained LSTM model
    scaler_y : sklearn.preprocessing scaler
        Fitted scaler for target variable
    X_train_seq, X_test_seq : numpy.ndarray
        Training and testing sequences
    y_train_seq, y_test_seq : numpy.ndarray
        Training and testing target values
    
    Returns:
    --------
    tuple : (report_json, y_train_pred, y_test_pred, y_train_actual, y_test_actual)
        - report_json: JSON string of performance metrics
        - Predictions and actual values for further analysis
    """
    
    # -----------------------------
    # Step 1: Predictions (scaled)
    # -----------------------------
    y_train_pred_scaled = model.predict(X_train_seq, verbose=0)
    y_test_pred_scaled = model.predict(X_test_seq, verbose=0)

    # -----------------------------
    # Step 2: Inverse Transform
    # -----------------------------
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()
    y_train_actual = scaler_y.inverse_transform(y_train_seq).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test_seq).flatten()

    # -----------------------------
    # Step 3: Metrics Function
    # -----------------------------
    def evaluate(y_true, y_pred):
        """Calculate regression metrics."""
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        
        # Handle division by zero in MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = float('inf')
        
        return {
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "MAPE": round(mape, 4) if mape != float('inf') else "N/A"
        }

    # -----------------------------
    # Step 4: Compute Metrics
    # -----------------------------
    train_metrics = evaluate(y_train_actual, y_train_pred)
    test_metrics = evaluate(y_test_actual, y_test_pred)

    # -----------------------------
    # Step 5: Comprehensive Report
    # -----------------------------
    generalization_gap = abs(train_metrics["RMSE"] - test_metrics["RMSE"])
    
    report = {
        "model_evaluation": {
            "training_set": {
                "metrics": train_metrics,
                "samples": int(len(y_train_actual))
            },
            "test_set": {
                "metrics": test_metrics,
                "samples": int(len(y_test_actual))
            },
            "summary": {
                "train_rmse": train_metrics["RMSE"],
                "test_rmse": test_metrics["RMSE"],
                "generalization_gap": round(generalization_gap, 4),
                "overfitting_detected": generalization_gap >= 5,
                "comment": (
                    "Model performance is stable with low overfitting."
                    if generalization_gap < 5
                    else "Model may be overfitting; consider regularization or more data."
                )
            },
            "performance_comparison": {
                "rmse_diff_percent": round((generalization_gap / train_metrics["RMSE"]) * 100, 2),
                "r2_train_vs_test": {
                    "train": train_metrics["R2"],
                    "test": test_metrics["R2"],
                    "difference": round(train_metrics["R2"] - test_metrics["R2"], 4)
                }
            }
        }
    }

    # Convert to JSON string
    report_json = json.dumps(report, indent=4)
    
    return report_json, y_train_pred, y_test_pred, y_train_actual, y_test_actual



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# -------------------------------------------------------------
# COMPREHENSIVE VISUALIZATION FUNCTION
# -------------------------------------------------------------
def visualize_lstm_results(
    y_train_actual,
    y_train_pred,
    y_test_actual,
    y_test_pred,
    history=None,
    save_path='lstm_evaluation_report.png',
    model_name='LSTM Model'
):
    """
    Create comprehensive visualization of LSTM model performance and save as PNG.
    
    Parameters:
    -----------
    y_train_actual : numpy.ndarray
        Actual training values
    y_train_pred : numpy.ndarray
        Predicted training values
    y_test_actual : numpy.ndarray
        Actual test values
    y_test_pred : numpy.ndarray
        Predicted test values
    history : keras.callbacks.History, optional
        Training history object from model.fit()
    save_path : str
        Path to save the PNG file (default: 'lstm_evaluation_report.png')
    model_name : str
        Name of the model for the title
    
    Returns:
    --------
    str : Path to saved PNG file
    """
    
    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE with zero handling
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    train_metrics = calculate_metrics(y_train_actual, y_train_pred)
    test_metrics = calculate_metrics(y_test_actual, y_test_pred)
    
    # Determine number of subplots
    n_plots = 6 if history is not None else 4
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle(f'{model_name} - Comprehensive Evaluation Report', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # -----------------------------
    # 1. Training Predictions vs Actual
    # -----------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(y_train_actual, label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
    ax1.plot(y_train_pred, label='Predicted', color='#A23B72', linewidth=2, alpha=0.8)
    ax1.set_title('Training Set: Predictions vs Actual', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample Index', fontsize=10)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Add metrics text box
    textstr = f"RMSE: {train_metrics['RMSE']:.4f}\nMAE: {train_metrics['MAE']:.4f}\nR²: {train_metrics['R2']:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # -----------------------------
    # 2. Test Predictions vs Actual
    # -----------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(y_test_actual, label='Actual', color='#2E86AB', linewidth=2, alpha=0.8)
    ax2.plot(y_test_pred, label='Predicted', color='#F18F01', linewidth=2, alpha=0.8)
    ax2.set_title('Test Set: Predictions vs Actual', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sample Index', fontsize=10)
    ax2.set_ylabel('Value', fontsize=10)
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Add metrics text box
    textstr = f"RMSE: {test_metrics['RMSE']:.4f}\nMAE: {test_metrics['MAE']:.4f}\nR²: {test_metrics['R2']:.4f}"
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # -----------------------------
    # 3. Scatter Plot - Training
    # -----------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_train_actual, y_train_pred, alpha=0.5, color='#2E86AB', edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_train_actual.min(), y_train_pred.min())
    max_val = max(y_train_actual.max(), y_train_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax3.set_title('Training: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Actual Values', fontsize=10)
    ax3.set_ylabel('Predicted Values', fontsize=10)
    ax3.legend(loc='best', frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    
    # -----------------------------
    # 4. Scatter Plot - Test
    # -----------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(y_test_actual, y_test_pred, alpha=0.5, color='#F18F01', edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test_actual.min(), y_test_pred.min())
    max_val = max(y_test_actual.max(), y_test_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax4.set_title('Test: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Actual Values', fontsize=10)
    ax4.set_ylabel('Predicted Values', fontsize=10)
    ax4.legend(loc='best', frameon=True, shadow=True)
    ax4.grid(True, alpha=0.3)
    
    # -----------------------------
    # 5. Residual Plot - Training
    # -----------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    train_residuals = y_train_actual - y_train_pred
    ax5.scatter(y_train_pred, train_residuals, alpha=0.5, color='#2E86AB', edgecolors='k', linewidth=0.5)
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax5.set_title('Training: Residual Plot', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Predicted Values', fontsize=10)
    ax5.set_ylabel('Residuals', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # -----------------------------
    # 6. Residual Plot - Test
    # -----------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    test_residuals = y_test_actual - y_test_pred
    ax6.scatter(y_test_pred, test_residuals, alpha=0.5, color='#F18F01', edgecolors='k', linewidth=0.5)
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax6.set_title('Test: Residual Plot', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Predicted Values', fontsize=10)
    ax6.set_ylabel('Residuals', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # -----------------------------
    # 7. Error Distribution - Training
    # -----------------------------
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(train_residuals, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax7.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax7.set_title('Training: Error Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Residual Value', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # -----------------------------
    # 8. Error Distribution - Test
    # -----------------------------
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(test_residuals, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    ax8.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax8.set_title('Test: Error Distribution', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Residual Value', fontsize=10)
    ax8.set_ylabel('Frequency', fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # -----------------------------
    # 9. Metrics Comparison Bar Chart
    # -----------------------------
    ax9 = fig.add_subplot(gs[2, 2])
    
    metrics_names = ['RMSE', 'MAE', 'R²']
    train_values = [train_metrics['RMSE'], train_metrics['MAE'], train_metrics['R2']]
    test_values = [test_metrics['RMSE'], test_metrics['MAE'], test_metrics['R2']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax9.bar(x - width/2, train_values, width, label='Training', 
                    color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax9.bar(x + width/2, test_values, width, label='Test', 
                    color='#F18F01', alpha=0.8, edgecolor='black')
    
    ax9.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Value', fontsize=10)
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics_names)
    ax9.legend(loc='best', frameon=True, shadow=True)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # -----------------------------
    # Optional: Training History
    # -----------------------------
    if history is not None:
        # Replace ax9 with training history plots
        ax9.clear()
        
        # Loss plot
        if 'loss' in history.history:
            color = '#2E86AB'
            ax9.plot(history.history['loss'], label='Training Loss', 
                    color=color, linewidth=2, alpha=0.8)
            if 'val_loss' in history.history:
                ax9.plot(history.history['val_loss'], label='Validation Loss', 
                        color='#F18F01', linewidth=2, alpha=0.8)
            
            ax9.set_title('Training History: Loss', fontsize=12, fontweight='bold')
            ax9.set_xlabel('Epoch', fontsize=10)
            ax9.set_ylabel('Loss', fontsize=10)
            ax9.legend(loc='best', frameon=True, shadow=True)
            ax9.grid(True, alpha=0.3)
    
            # Add footer with timestamp and summary
            # Add footer with summary only (no timestamp)
            footer_text  = f"Train Samples: {len(y_train_actual)} | "
            footer_text += f"Test Samples: {len(y_test_actual)} | "
            footer_text += f"Generalization Gap (RMSE): {abs(train_metrics['RMSE'] - test_metrics['RMSE']):.4f}"

            fig.text(
                0.5, 0.01, footer_text,
                ha='center', fontsize=10,
                style='italic',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
            )
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Close to free memory
    plt.close()
    
    return save_path


import matplotlib.pyplot as plt

def plot_and_save_history(report, save_path="training_report.png"):
    """
    Plot training/validation loss and metrics from Keras history report,
    and save the visualization as a PNG image.

    Parameters:
    -----------
    report : dict
        The history.history dictionary from model.fit()
    save_path : str
        File path to save the PNG image (default: 'training_report.png')
    """

    plt.figure(figsize=(12, 5))

    # Plot training vs validation loss
    plt.subplot(1, 2, 1)
    plt.plot(report['loss'], label='Train Loss')
    plt.plot(report['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot first available metric (other than loss)
    metric_keys = [k for k in report.keys() if not k.startswith('val_') and k != 'loss']
    if metric_keys:
        metric = metric_keys[0]  # pick the first metric (e.g., 'accuracy', 'mae', 'rmse')
        plt.subplot(1, 2, 2)
        plt.plot(report[metric], label=f'Train {metric}')
        plt.plot(report[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f"Model {metric.capitalize()}")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

