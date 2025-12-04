import os
import sys
from src.constants.paths import *


class ModelTrainingConfig:
    def __init__(self):
        self.model_dir = os.path.join(ARTIFACTS_DIR, MODEL_TRAINING_DIR)
        os.makedirs(self.model_dir, exist_ok=True)

        self.train_model_path = os.path.join(self.model_dir,MODEL_TRAINING_FILE)
        self.train_model_report_path = os.path.join(self.model_dir, MODEL_TRAINING_REPORT_FILE)
        self.scaler_preprocess_path = os.path.join(self.model_dir, SCALLING_TRANSFORMATION_PKL_FILE)


class ModelEvaluationConfig:
    def __init__(self):
        self.eval_mode_dir = os.path.join(ARTIFACTS_DIR, MODEL_EVALUATION_DIR)
        os.makedirs(self.eval_mode_dir, exist_ok=True)

        self.eval_model_report_path = os.path.join(self.eval_mode_dir, MODEL_EVALUATION_REPORT_FILE)
        self.eval_model_png_path = os.path.join(self.eval_mode_dir, MODEL_EVALUATION_PNG_FILE)

class ModelDeploymentConfig:
    def __init__(self):
        self.model_deployment_dir = os.path.join(DEPLOYED_ARTIFACTS_DIR, MODEL_DEPLOYMENT_DIR)
        os.makedirs(self.model_deployment_dir, exist_ok=True)

        self.deployed_model_path = os.path.join(self.model_deployment_dir, MODEL_DEPLOYMENT_MODEL_FILE)
        self.deployed_preprocessor_path = os.path.join(self.model_deployment_dir, MODEL_DEPLOYMENT_PREPROCESSOR_FILE)
        self.deployment_report_path = os.path.join(self.model_deployment_dir, MODEL_DEPLOYMENT_REPORT_FILE)


