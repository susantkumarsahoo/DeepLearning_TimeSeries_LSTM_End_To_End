import os
import sys
from src.constants.paths import *

class IngestionConfig:
    def __init__(self, dataset_path: str, test_size: float = TRAIN_TEST_SPLIT_RATIO, random_state: int = RANDOM_STATE):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state

        # Root ingestion directory
        self.ingestion_dir = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_DIR)

        # Subdirectories
        self.raw_data_dir = os.path.join(self.ingestion_dir, DATA_INGESTION_RAW_DIR)
        self.processed_data_dir = os.path.join(self.ingestion_dir, DATA_INGESTION_PROCESSED_DIR)
        self.split_data_dir = os.path.join(self.ingestion_dir, DATA_INGESTION_SPLIT_DIR)

        # Create directories
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.split_data_dir, exist_ok=True)

        # File paths
        self.raw_data_path = os.path.join(self.raw_data_dir, DATA_INGESTION_RAW_FILE)
        self.processed_data_path = os.path.join(self.processed_data_dir, DATA_INGESTION_PROCESSED_FILE)
        self.train_data_path = os.path.join(self.split_data_dir, DATA_INGESTION_TRAIN_FILE)
        self.test_data_path = os.path.join(self.split_data_dir, DATA_INGESTION_TEST_FILE)

        # Metadata & schema paths (stored in ingestion root)
        self.metadata_path = os.path.join(self.ingestion_dir, DATA_INGESTION_METADATA_FILE)
        self.schema_path = os.path.join(self.ingestion_dir, DATA_INGESTION_SCHEMA_FILE)

class ValidationConfig:
    def __init__(self):
        # Base folder for data validation
        self.validation_dir = os.path.join(ARTIFACTS_DIR, DATA_VALIDATION_DIR)

        # Auto-create directories here
        os.makedirs(self.validation_dir, exist_ok=True)

        # Individual report paths
        self.drift_report_path = os.path.join(self.validation_dir, DATA_VALIDATION_DRIFT_REPORT)
        self.validation_status_path = os.path.join(self.validation_dir, DATA_VALIDATION_STATUS_FILE)
        self.quality_report_path = os.path.join(self.validation_dir, DATA_VALIDATION_QUALITY_REPORT)
        self.schema_validation_path = os.path.join(self.validation_dir, SCHEMA_STRUCTURE_VALIDATION_REPORT)
        self.statistical_validation_path = os.path.join(self.validation_dir, STATISTICAL_VALIDATION_REPORT)
        self.time_series_validation_path = os.path.join(self.validation_dir, TIME_SERIES_VALIDATION_REPORT)

class PreprossingConfig:
    def __init__(self):
        self.preprocess_dir = os.path.join(ARTIFACTS_DIR, DATA_PREPROCESSING_DIR)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        self.train_preprocessed_path = os.path.join(self.preprocess_dir, DATA_PREPROCESSING_TRAIN_FILE)
        self.test_preprocessed_path = os.path.join(self.preprocess_dir, DATA_PREPROCESSING_TEST_FILE)
        self.preprocessing_report_path = os.path.join(self.preprocess_dir, DATA_PREPROCESSING_REPORT_FILE)


class FeatureEngineeringConfig:
    def __init__(self):
        self.feature_engineering_dir = os.path.join(ARTIFACTS_DIR, FEATURE_ENGINEERING_DIR)
        os.makedirs(self.feature_engineering_dir, exist_ok=True)
        self.feature_engineering_report_path = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_REPORT_FILE)
        self.train_feature_engineering_path = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_TRAIN_FILE)
        self.test_feature_engineering_path = os.path.join(self.feature_engineering_dir, FEATURE_ENGINEERING_TEST_FILE)

        

  