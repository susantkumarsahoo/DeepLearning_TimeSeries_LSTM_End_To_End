import sys
from src.logging.logger import get_logger
from src.constants.paths import dataset_path
from src.exceptions.exception import ProjectException
from src.entity.components_config_entity import IngestionConfig
from src.components.data_ingestion import DataIngestion

logger = get_logger(__name__)


def run_ingestion_pipeline(dataset_path: str):
    """
    Run the Data Ingestion pipeline independently.
    """
    try:
        logger.info("=== DATA INGESTION PIPELINE STARTED ===")

        ingestion_config = IngestionConfig(dataset_path=dataset_path)
        ingestion = DataIngestion(config=ingestion_config)

        ingestion_artifact = ingestion.initiate_data_ingestion()

        logger.info("=== DATA INGESTION PIPELINE COMPLETED ===")
        return ingestion_artifact

    except Exception as e:
        logger.error(f"Data Ingestion Failed: {str(e)}", exc_info=True)
        raise ProjectException(e, sys)


if __name__ == "__main__":
    dataset_path = dataset_path
    artifact = run_ingestion_pipeline(dataset_path)
    print(artifact)


    # python src/pipelines/data_ingestion_pipeline.py


