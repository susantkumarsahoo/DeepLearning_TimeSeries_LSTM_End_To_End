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

logger = get_logger(__name__)

class PredictionPipeline:
    def __init__(self, deployment_artifact: DeploymentArtifact):
        try:
            self.deployment_artifact = deployment_artifact
        except Exception as e:
            raise ProjectException(e, sys)
        

    def predict(self, input_data):
        try:
            model = load_model(self.deployment_artifact.deployed_model_file)

    
            return model.predict(input_data)
        except Exception as e:
            raise ProjectException(e, sys)
        



