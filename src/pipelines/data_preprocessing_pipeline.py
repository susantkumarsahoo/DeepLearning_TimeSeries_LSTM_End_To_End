import sys
from src.logging.logger import get_logger
from src.constants.paths import dataset_path
from src.exceptions.exception import ProjectException
from src.entity.components_config_entity import PreprossingConfig
from src.components.data_preprocessing import Preprocessing

logger = get_logger(__name__)


def run_preprocessing_pipeline(data_ingestion_artifact, data_validation_artifact):
    """
    Execute preprocessing using both ingestion + validation artifacts.
    """
    try:
        logger.info("=== PREPROCESSING PIPELINE STARTED ===")

        preprocessing_config = PreprossingConfig()
        preprocessing = Preprocessing(
            preprocessing_config=preprocessing_config,
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_artifact=data_validation_artifact
        )

        preprocessing_artifact = preprocessing.initiate_preprocessing()

        logger.info("=== PREPROCESSING PIPELINE COMPLETED ===")
        return preprocessing_artifact

    except Exception as e:
        logger.error(f"Preprocessing Failed: {str(e)}", exc_info=True)
        raise ProjectException(e, sys)


if __name__ == "__main__":
    
    from src.pipelines.data_ingestion_pipeline import run_ingestion_pipeline
    from src.pipelines.data_validation_pipeline import run_validation_pipeline

    dataset_path = dataset_path

    ingestion_artifact = run_ingestion_pipeline(dataset_path)
    validation_artifact = run_validation_pipeline(ingestion_artifact)

    preprocessing_artifact = run_preprocessing_pipeline(ingestion_artifact, validation_artifact)
    print(preprocessing_artifact)


    # python src/pipelines/data_preprocessing_pipeline.py
