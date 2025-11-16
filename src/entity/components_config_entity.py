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

        # Timestamped run directory
        self.timestamp_dir = os.path.join(self.ingestion_dir, TIMESTAMP)

        # Subdirectories
        self.raw_data_dir = os.path.join(self.timestamp_dir, DATA_INGESTION_RAW_DIR)
        self.processed_data_dir = os.path.join(self.timestamp_dir, DATA_INGESTION_PROCESSED_DIR)
        self.split_data_dir = os.path.join(self.timestamp_dir, DATA_INGESTION_SPLIT_DIR)

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
        self.metadata_path = os.path.join(self.timestamp_dir, DATA_INGESTION_METADATA_FILE)
        self.schema_path = os.path.join(self.timestamp_dir, DATA_INGESTION_SCHEMA_FILE)

import os

class ValidationConfig:
    def __init__(self):
        # Base folder for data validation
        self.validation_dir = os.path.join(ARTIFACTS_DIR, DATA_VALIDATION_DIR, TIMESTAMP)

        # Auto-create directories here
        os.makedirs(self.validation_dir, exist_ok=True)

        # Individual report paths
        self.drift_report_file = os.path.join(self.validation_dir, DATA_VALIDATION_DRIFT_REPORT)
        self.validation_status_file = os.path.join(self.validation_dir, DATA_VALIDATION_STATUS_FILE)
        self.quality_report_file = os.path.join(self.validation_dir, DATA_VALIDATION_QUALITY_REPORT)
        self.schema_validation_file = os.path.join(self.validation_dir, SCHEMA_STRUCTURE_VALIDATION_REPORT)
        self.statistical_validation_file = os.path.join(self.validation_dir, STATISTICAL_VALIDATION_REPORT)
        self.time_series_validation_file = os.path.join(self.validation_dir, TIME_SERIES_VALIDATION_REPORT)
