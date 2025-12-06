import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tensorflow as tf
import joblib
from datetime import datetime
from src.constants.paths import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from src.entity.model_config_entity import ModelTrainingConfig, ModelEvaluationConfig
from src.entity.artifact_entity import TrainingArtifact, EvaluationArtifact, TransformationArtifact
from src.exceptions.exception import ProjectException
from src.logging.logger import get_logger
from src.utils.helpers import save_json
from src.utils.model_training_helper import create_sequences, split_dataset_report, build_and_train_lstm, prepare_datetime_index
from src.utils.model_evaluator_helper import (
    load_artifacts,
    evaluate_lstm_model,
    visualize_lstm_results
)
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


logger = get_logger(__name__)


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_training_artifact: TrainingArtifact, transformation_artifact: TransformationArtifact) -> None:
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_training_artifact = model_training_artifact
            self.transformation_artifact = transformation_artifact

            logger.info("=" * 70)
            logger.info("MODEL EVALUATION INITIATED")
            logger.info("=" * 70)

        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_model_evaluation(self):
        try:
            logger.info("Model Evaluation Started")
            
            # --------------------------------------------------------
            # Load transformed datasets
            # --------------------------------------------------------
            train_df = pd.read_csv(self.transformation_artifact.train_transformed_file)
            test_df = pd.read_csv(self.transformation_artifact.test_transformed_file)

            # datetime setup
            train_df, test_df = prepare_datetime_index(train_df, test_df, 'date')

            X_train = train_df.drop(columns=TARGET_COLUMN)
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=TARGET_COLUMN)
            y_test = test_df[TARGET_COLUMN]

            # --------------------------------------------------------
            # Load Model and Scaler (MOVED BEFORE TARGET SCALING)
            # --------------------------------------------------------
            model = load_model(self.model_training_artifact.trained_model_file)
            scaler_y = joblib.load(self.model_training_artifact.scalling_preprocessor_pkl_file)

            # --------------------------------------------------------
            # Target Scaling (USING LOADED SCALER)
            # --------------------------------------------------------
            y_train_scaled = scaler_y.transform(y_train.values.reshape(-1, 1))
            y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

            # --------------------------------------------------------
            # Sequence Creation (NEED seq_length DEFINITION)
            # --------------------------------------------------------
            # Get seq_length from config or define it
            #seq_length = self.model_evaluation_config.seq_length if hasattr(self.model_evaluation_config, 'seq_length') else 60
            
            X_train_seq, y_train_seq = create_sequences(X_train, y_train_scaled, seq_length)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test_scaled, seq_length)

            # --------------------------------------------------------
            # Evaluate model
            # --------------------------------------------------------
            report_json, y_train_pred, y_test_pred, y_train_actual, y_test_actual = evaluate_lstm_model(
                model=model,
                scaler_y=scaler_y,
                X_train_seq=X_train_seq,
                X_test_seq=X_test_seq,
                y_train_seq=y_train_seq,
                y_test_seq=y_test_seq
            )

            # --------------------------------------------------------
            # Visualize LSTM Results
            # --------------------------------------------------------
            visualize_lstm_results(
                y_train_actual=y_train_actual,
                y_train_pred=y_train_pred,
                y_test_actual=y_test_actual,
                y_test_pred=y_test_pred,
                save_path=self.model_evaluation_config.eval_model_png_path,
                model_name='Stock Price Prediction LSTM'
            )

            # --------------------------------------------------------
            # Save evaluation report
            # --------------------------------------------------------
            save_json(report_json, self.model_evaluation_config.eval_model_report_path)

            # --------------------------------------------------------
            # Create evaluation artifact
            # --------------------------------------------------------
            model_evaluation_artifact = EvaluationArtifact(
                evaluation_report_file=self.model_evaluation_config.eval_model_report_path,
                evaluated_model_png_file=self.model_evaluation_config.eval_model_png_path
            )

            logger.info("Model Evaluation Completed Successfully")
            return model_evaluation_artifact

        except Exception as e:
            raise ProjectException(e, sys)