import os
import sys
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from src.entity.model_config_entity import ModelTrainingConfig, ModelEvaluationConfig
from src.entity.artifact_entity import TrainingArtifact,EvaluationArtifact
from src.exceptions.exception import ProjectException
from src.logging.logger import get_logger
from src.utils.helpers import save_json
from src.utils.model_evaluator_helper import load_artifacts
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



logger = get_logger(__name__)



class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, training_artifact: TrainingArtifact) -> None:
        try:
            self.model_evaluation_config = model_evaluation_config
            self.training_artifact = training_artifact
            logger.info("="*70)
            logger.info("MODEL EVALUATION INITIATED")
            logger.info("="*70)
        except Exception as e:
            raise ProjectException(e, sys)
        
    def initiate_model_evaluation(self) -> EvaluationArtifact:
        try:
            logger.info("Model Evaluation Initiated")
            # lode model and scaler
            model = load_artifacts(self.training_artifact.trained_model_file)

            scaler_y = load_artifacts(self.training_artifact.scalling_preprocessor_pkl_file)    

            # perfonance evaluation

            logger.info("Model Evaluation Completed")
            return

        except Exception as e:
            raise ProjectException(e, sys)











