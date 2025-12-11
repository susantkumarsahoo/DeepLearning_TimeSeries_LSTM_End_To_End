import pandas as pd
import sys
import os
from datetime import datetime, timedelta
import numpy as np

from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException

from src.entity.artifact_entity import DeploymentArtifact, PredictorArtifact
from src.entity.model_config_entity import PredictorConfig

from src.models.predictor import ModelPredictor

logger = get_logger(__name__)


def run_prediction_pipeline(start_date: datetime, end_date: datetime, deployment_artifact: DeploymentArtifact) -> PredictorArtifact:
    """
    Run the prediction pipeline for the given date range.
    
    Args:
        start_date (datetime): Start date for predictions
        end_date (datetime): End date for predictions
        deployment_artifact (DeploymentArtifact): Artifact containing deployed model information
    
    Returns:
        PredictorArtifact: Artifact containing prediction file path
    """
    try:
        logger.info("=" * 80)
        logger.info("STARTING PREDICTION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Prediction range: {start_date} to {end_date}")
        
        # Validate date inputs
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        # Validate deployment artifact
        if deployment_artifact is None:
            raise ValueError("deployment_artifact cannot be None")
        
        # Create input data DataFrame
        input_data = pd.DataFrame({
            'start_date': [start_date],
            'end_date': [end_date]
        })
        
        # Initialize configuration
        predictor_config = PredictorConfig()
        logger.info(f"Predictor config initialized with seq_length: {predictor_config.seq_length}")
       
        # Create predictor instance
        model_predictor = ModelPredictor(
            predictor_config=predictor_config,
            deployment_artifact=deployment_artifact
        )

        # Generate predictions
        logger.info("Starting prediction generation...")
        model_predictions_artifact = model_predictor.initiate_model_prediction(input_data=input_data)
        
        logger.info("=" * 80)
        logger.info("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return model_predictions_artifact

    except Exception as e:
        logger.error("=" * 80)
        logger.error("PREDICTION PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise ProjectException(e, sys)
    

# python src/pipelines/model_prediction_pipeline.py