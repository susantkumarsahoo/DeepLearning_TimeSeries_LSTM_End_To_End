"""
Example Usage of Data Ingestion Component
File: main.py or pipeline.py
"""

import sys
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.components_config_entity import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

logger = get_logger(__name__)


def run_data_ingestion_pipeline():
    """
    Main function to run the data ingestion pipeline.
    """
    try:
        logger.info("Starting ML Pipeline")
        logger.info("="*70)
        
        # Configuration
        dataset_path = r"C:\Users\TPWODL\New folder_Content\DeepLearning_TimeSeries_LSTM_End_To_End\data\raw\Energy Demand Hourly.csv"  # Change this to your dataset path
        # Initialize configuration
        config = DataIngestionConfig(dataset_path=dataset_path,)
        
        # Initialize and run data ingestion
        data_ingestion = DataIngestion(config=config)
        ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        logger.info("ML Pipeline completed successfully")
        
        return ingestion_artifact
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise ProjectException(e, sys)


if __name__ == "__main__":
    try:
        artifact = run_data_ingestion_pipeline()
        print("\n" + "="*70)
        print("DATA INGESTION ARTIFACTS:")
        print("="*70)
        print(f"Raw Data: {artifact.raw_data_file}")
        print(f"Processed Data: {artifact.processed_data_file}")
        print(f"Train Data: {artifact.train_file}")
        print(f"Test Data: {artifact.test_file}")
        print(f"Metadata: {artifact.metadata_file}")
        print(f"Schema: {artifact.schema_file}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)