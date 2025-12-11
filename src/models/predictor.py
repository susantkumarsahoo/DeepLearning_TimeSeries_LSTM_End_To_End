import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from src.constants.paths import *
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.utils.prediction_helper import (
    generate_forecast_features,
    create_sequence_df,
    generate_predictions
)
from src.entity.artifact_entity import PredictorArtifact, DeploymentArtifact
from src.entity.model_config_entity import PredictorConfig
from tensorflow.keras.models import load_model


logger = get_logger(__name__)


class ModelPredictor:
    def __init__(self, predictor_config: PredictorConfig, deployment_artifact: DeploymentArtifact):
        """
        Initialize ModelPredictor with configuration and deployment artifacts.
        
        Args:
            predictor_config: Configuration for prediction
            deployment_artifact: Deployment artifact containing model paths
        """
        try:
            self.predictor_config = predictor_config
            self.deployment_artifact = deployment_artifact
            logger.info("ModelPredictor initialized successfully")
        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_model_prediction(self, input_data: pd.DataFrame) -> PredictorArtifact:
        """
        Generate predictions for the given input data.
        
        Args:
            input_data: DataFrame containing start_date and end_date columns
            
        Returns:
            PredictorArtifact: Artifact containing prediction file path
        """
        try:
            logger.info("Starting model prediction process")
            
            # Validate input data
            if input_data is None or input_data.empty:
                raise ValueError("Input data cannot be None or empty")
            
            if 'start_date' not in input_data.columns or 'end_date' not in input_data.columns:
                raise ValueError("Input data must contain 'start_date' and 'end_date' columns")
            
            # Ensure directory exists for output
            os.makedirs(os.path.dirname(self.predictor_config.model_Predictor_path), exist_ok=True)
            
            # Load model and scaler
            logger.info(f"Loading model from: {self.deployment_artifact.deployed_model_file}")
            if not os.path.exists(self.deployment_artifact.deployed_model_file):
                raise FileNotFoundError(f"Model file not found: {self.deployment_artifact.deployed_model_file}")
            model = load_model(self.deployment_artifact.deployed_model_file)
            
            logger.info(f"Loading scaler from: {self.deployment_artifact.deployed_preprocessor_file}")
            if not os.path.exists(self.deployment_artifact.deployed_preprocessor_file):
                raise FileNotFoundError(f"Scaler file not found: {self.deployment_artifact.deployed_preprocessor_file}")
            scaler_y = joblib.load(self.deployment_artifact.deployed_preprocessor_file)

            # Generate forecast feature data
            start_date = pd.to_datetime(input_data['start_date'].iloc[0])
            end_date = pd.to_datetime(input_data['end_date'].iloc[0])
            logger.info(f"Generating forecast features from {start_date} to {end_date}")
            
            forecast_features = generate_forecast_features(start_date, end_date)
            
            if forecast_features is None or len(forecast_features) == 0:
                raise ValueError("Failed to generate forecast features")

            # Prepare sequence data
            logger.info(f"Creating sequences with length: {self.predictor_config.seq_length}")
            seq_df = create_sequence_df(
                forecast_features,
                self.predictor_config.seq_length
            )
            
            if seq_df is None or len(seq_df) == 0:
                raise ValueError("Failed to create sequence dataframe")

            # Execute predictions
            logger.info("Generating predictions")
            predictions = generate_predictions(
                model,
                seq_df,
                scaler_y,
                forecast_features,
                self.predictor_config.seq_length
            )
            
            if predictions is None or len(predictions) == 0:
                raise ValueError("Failed to generate predictions")
            
            # Save predictions
            predictions.to_csv(self.predictor_config.model_Predictor_path, index=False)
            logger.info(f"Predictions saved to: {self.predictor_config.model_Predictor_path}")
            logger.info(f"Predictions completed. Generated {len(predictions)} predictions")

            # Return predictions artifact
            predictions_artifact = PredictorArtifact(
                predictor_file=self.predictor_config.model_Predictor_path,
            )
            return predictions_artifact

        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            raise ProjectException(e, sys)


# python src/models/predictor.py