import os
import sys
from src.constants.paths import *

class DataIngestionConfig:
    def __init__(self, dataset_path: str, test_size: float = 0.2, random_state: int = 42):
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
