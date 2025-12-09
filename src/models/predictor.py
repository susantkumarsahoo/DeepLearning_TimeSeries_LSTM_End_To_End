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
    def __init__(self, modelprediction_config: PredictorConfig, deployment_artifact: DeploymentArtifact):
        try:
            self.modelprediction_config = modelprediction_config
            self.deployment_artifact = deployment_artifact
        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_model_prediction(self, input_data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Load model and scaler
            model = load_model(self.deployment_artifact.deployed_model_file)
            scaler_y = joblib.load(self.deployment_artifact.deployed_preprocessor_file)

            # Generate forecast feature data
            start_date = input_data['10/12/2025'].iloc[0]
            end_date = input_data['11/1/2026'].iloc[0]
            forecast_features = generate_forecast_features(start_date, end_date)

            # Prepare sequence data
            seq_df = create_sequence_df(
                forecast_features,
                self.modelprediction_config.seq_length
            )

            # Execute predictions
            predictions = generate_predictions(
                model,
                seq_df,
                scaler_y,
                forecast_features,
                self.modelprediction_config.seq_length
            )

            return predictions

        except Exception as e:
            raise ProjectException(e, sys)

    

    