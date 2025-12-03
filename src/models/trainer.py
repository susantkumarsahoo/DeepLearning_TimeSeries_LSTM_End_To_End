import os
import sys
import pandas as pd
import numpy as np
import joblib
from src.constants.paths import *
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.model_config_entity import ModelTrainingConfig
from src.entity.artifact_entity import TransformationArtifact,TrainingArtifact
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from src.utils.helpers import save_json
from src.utils.model_training_helper import create_sequences,split_dataset_report,build_and_train_lstm

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, transformation_artifact: TransformationArtifact, model_training_config: ModelTrainingConfig) -> None:
        try:
            self.transformation_artifact = transformation_artifact
            self.model_training_config = model_training_config

        except Exception as e:
            raise ProjectException(e, sys)
        
    def initiate_model_training(self) -> TrainingArtifact:
        try:
            logger.info("Model Training Initiated")
            train_df = pd.read_csv(self.transformation_artifact.train_transformed_file)
            test_df = pd.read_csv(self.transformation_artifact.test_transformed_file)

            # train test splited data
            X_train = train_df.drop(columns=TARGET_COLUMN)
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=TARGET_COLUMN)
            y_test = test_df[TARGET_COLUMN]

            # Initialize scaler (no need for 'params' unless you have specific kwargs)
            scaler_y = RobustScaler()

            # Fit on training target and transform both train & test
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
            y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1, 1))

            # Save the fitted scaler to pkl file
            joblib.dump(scaler_y, self.model_training_config.scaler_preprocess_path)

            # Keep X as-is (no scaling required)
            X_train_scaled = X_train.copy()
            X_test_scaled  = X_test.copy()

            # create sequences
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
            X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)

            # save reports
            split_data_report = split_dataset_report(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_train_seq, y_train_seq, X_test_seq, y_test_seq)

            save_json(split_data_report, self.model_training_config.train_model_report_path)

            # build model
            model, model_report = build_and_train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq, seq_length)

            # save report
            save_json(model_report, self.model_training_config.train_model_report_path)

            # save model
            model.save(self.model_training_config.train_model_path)


            training_report = {
                'split_data_report':  split_data_report,
                'model_report': model_report

            }
            logger.info("Model Training Completed")

            training_artifact = TrainingArtifact(
                trained_model_file=self.model_training_config.train_model_path,
                training_report_file=self.model_training_config.train_model_report_path,
                scalling_preprocessor_pkl_file=self.model_training_config.scaler_preprocess_path
            )            
            logger.info("Model Training Artifact Created")
            
            return training_artifact
        except Exception as e:
            raise ProjectException(
                e, sys          
            )


