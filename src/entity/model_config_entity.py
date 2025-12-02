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

