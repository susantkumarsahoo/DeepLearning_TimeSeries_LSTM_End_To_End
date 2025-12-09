import pandas as pd
import sys
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import RobustScaler

from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException

from src.entity.artifact_entity import DeploymentArtifact
from tensorflow.keras.models import load_model

from src.models.predictor import ModelPredictor

logger = get_logger(__name__)


def run_prediction_pipeline(deployment_artifact: DeploymentArtifact, input_data: pd.DataFrame) -> pd.DataFrame:
    try:
        # IMPORTANT: ModelPredictor requires TWO arguments:
        # (modelprediction_config, deployment_artifact)
        # You were giving only one → corrected with placeholder config.
        model_predictor = ModelPredictor(
            modelprediction_config=None,       # Placeholder if config not required
            deployment_artifact=deployment_artifact
        )

        predictions = model_predictor.initiate_model_prediction(input_data=input_data)
        return predictions

    except Exception as e:
        raise ProjectException(e, sys)


if __name__ == "__main__":
    try:

        # -------------------------------------------------------------------
        # TODO: Replace these placeholders with actual objects
        deployment_artifact = None   # <-- update with real DeploymentArtifact instance
        input_data = pd.DataFrame()  # <-- update with real input data
        # -------------------------------------------------------------------

        predictions = run_prediction_pipeline(
            deployment_artifact=deployment_artifact,
            input_data=input_data
        )

        print(predictions)

    except Exception as e:
        raise ProjectException(e, sys)


# python src/pipelines/prediction_pipeline.py