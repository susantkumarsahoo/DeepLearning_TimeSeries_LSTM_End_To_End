import os
import sys
import pandas as pd
import numpy as np
import joblib
from src.constants.paths import *
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.model_config_entity import ModelTrainingConfig,ModelEvaluationConfig    
from src.entity.artifact_entity import TransformationArtifact,TrainingArtifact
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from src.utils.helpers import save_json
from src.utils.model_training_helper import create_sequences,split_dataset_report,build_and_train_lstm,prepare_datetime_index
from src.models.evaluator import ModelEvaluation
from src.utils.model_evaluator_helper import (
    load_artifacts,
    evaluate_lstm_model,
    visualize_lstm_results
)

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, transformation_artifact: TransformationArtifact, model_training_config: ModelTrainingConfig, model_evaluation_config: ModelEvaluationConfig,) -> None:
        try:
            self.transformation_artifact = transformation_artifact
            self.model_training_config = model_training_config
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise ProjectException(e, sys)
        
    def initiate_model_training(self) -> TrainingArtifact:
        try:
            logger.info("Model Training Initiated")

            # --------------------------------------------------------
            # Load transformed datasets
            # --------------------------------------------------------
            train_df = pd.read_csv(self.transformation_artifact.train_transformed_file)
            test_df = pd.read_csv(self.transformation_artifact.test_transformed_file)

            # date time setup
            train_df,test_df = prepare_datetime_index(train_df,test_df,'date')


            X_train = train_df.drop(columns=TARGET_COLUMN)
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=TARGET_COLUMN)
            y_test = test_df[TARGET_COLUMN]

            # --------------------------------------------------------
            # Target Scaling
            # --------------------------------------------------------
            scaler_y = RobustScaler()

            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
            y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

            joblib.dump(scaler_y, self.model_training_config.scaler_preprocess_path)

            # --------------------------------------------------------
            # Sequence Creation
            # --------------------------------------------------------
            X_train_seq, y_train_seq = create_sequences(X_train, y_train_scaled, seq_length)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test_scaled, seq_length)

            # --------------------------------------------------------
            # Prepare & Save Split Report
            # --------------------------------------------------------
            split_data_report = split_dataset_report(
                X_train, X_test, y_train_scaled, y_test_scaled,
                X_train_seq, y_train_seq, X_test_seq, y_test_seq
            )

            save_json(
                split_data_report,
                self.model_training_config.train_model_report_path
            )

            # --------------------------------------------------------
            # LSTM Model Training
            # --------------------------------------------------------
            model, model_report = build_and_train_lstm(
                X_train_seq, y_train_seq,
                X_test_seq, y_test_seq,
                seq_length
            )

            # Save model training report separately
            save_json(
                model_report,
                self.model_training_config.train_model_report_path
            )

            # --------------------------------------------------------
            # Save trained model
            # --------------------------------------------------------
            model.save(self.model_training_config.train_model_path)

            # --------------------------------------------------------
            # Create Artifact
            # --------------------------------------------------------
            training_artifact = TrainingArtifact(
                trained_model_file=self.model_training_config.train_model_path,
                training_report_file=self.model_training_config.train_model_report_path,
                scalling_preprocessor_pkl_file=self.model_training_config.scaler_preprocess_path
            )

            logger.info("Model Training Completed Successfully")
            logger.info("Model Training Artifact Created")

            return training_artifact

        except Exception as e:
            raise ProjectException(e, sys)



