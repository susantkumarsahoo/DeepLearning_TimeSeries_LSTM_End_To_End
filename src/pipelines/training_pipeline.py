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


def run_data_ingestion_pipeline(dataset_path: str) -> None:
    """
    Main function to run the data ingestion pipeline.
    """
    try:
        logger.info("Starting ML Pipeline")
        logger.info("="*70)

        config = DataIngestionConfig(dataset_path=dataset_path,)       
        # Initialize and run data ingestion
        data_ingestion = DataIngestion(config=config)
        ingestion_artifact = data_ingestion.initiate_data_ingestion()   
        
        logger.info("ML Pipeline completed successfully")
        
        return ingestion_artifact
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise ProjectException(e, sys)
