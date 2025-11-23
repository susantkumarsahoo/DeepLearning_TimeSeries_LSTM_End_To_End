import sys
from src.logging.logger import get_logger
from src.constants.paths import dataset_path
from src.exceptions.exception import ProjectException
from src.entity.components_config_entity import ValidationConfig
from src.components.data_validation import DataValidation

logger = get_logger(__name__)


def run_validation_pipeline(data_ingestion_artifact):
    """
    Run Data Validation independently, using DataIngestionArtifact.
    """
    try:
        logger.info("=== DATA VALIDATION PIPELINE STARTED ===")

        validation_config = ValidationConfig()
        validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=validation_config
        )

        validation_artifact = validation.initiate_data_validation()

        logger.info("=== DATA VALIDATION PIPELINE COMPLETED ===")
        return validation_artifact

    except Exception as e:
        logger.error(f"Data Validation Failed: {str(e)}", exc_info=True)
        raise ProjectException(e, sys)


if __name__ == "__main__":
    from src.pipelines.data_ingestion_pipeline import run_ingestion_pipeline

    dataset_path = dataset_path
    ingestion_artifact = run_ingestion_pipeline(dataset_path)

    validation_artifact = run_validation_pipeline(ingestion_artifact)
    print(validation_artifact)


    # python src/pipelines/data_validation_pipeline.py


