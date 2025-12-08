import os
import sys
from src.entity.model_config_entity import ModelDeploymentConfig
from src.entity.artifact_entity import TrainingArtifact, DeploymentArtifact
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from tensorflow.keras.models import load_model
import joblib

# Imports from helpers and metrics
from src.utils.helpers import save_json, save_model, save_object
from src.utils.metrics import load_model_and_generate_report, load_scaler_and_generate_report


logger = get_logger(__name__)


class ModelRegistry:
    def __init__(
        self,
        model_registry_config: ModelDeploymentConfig,
        training_artifact: TrainingArtifact,
        deployment_artifact: DeploymentArtifact
    ) -> None:
        try:
            self.model_registry_config = model_registry_config
            self.training_artifact = training_artifact
            self.deployment_artifact = deployment_artifact
        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_model_registry(self):
        try:
            # Load model and scaler
            model = load_model(self.training_artifact.trained_model_file)
            scaler_y = joblib.load(self.training_artifact.scalling_preprocessor_pkl_file)

            # -----------------------------
            # FIX: Provide save_path argument
            # -----------------------------
            model_report = load_model_and_generate_report(self.training_artifact.trained_model_file)

            scaler_y_report = load_scaler_and_generate_report(self.training_artifact.scalling_preprocessor_pkl_file)

            final_report = {
                "model_report": model_report,
                "scaler_y_report": scaler_y_report
            }

            # Save consolidated report
            save_json(final_report, self.model_registry_config.deployment_report_path)

            # Save model + scaler to deployment directory
            save_model(model, self.model_registry_config.deployed_model_path)
            save_object(scaler_y, self.model_registry_config.deployed_preprocessor_path)

            # Create final deployment artifact
            deployment_artifact = DeploymentArtifact(
                deployed_model_file=self.model_registry_config.deployed_model_path,
                deployed_preprocessor_file=self.model_registry_config.deployed_preprocessor_path,
                deployment_report_file=self.model_registry_config.deployment_report_path
            )

            logger.info("Model registry completed successfully")

            return deployment_artifact

        except Exception as e:
            raise ProjectException(e, sys)

