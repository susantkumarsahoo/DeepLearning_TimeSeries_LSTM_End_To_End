import sys
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.components_config_entity import IngestionConfig, ValidationConfig, PreprossingConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_preprocessing import Preprocessing

logger = get_logger(__name__)


def run_data_ingestion_pipeline(dataset_path: str):
    """
    Executes the full ML pipeline: Data Ingestion → Data Validation.
    Returns the final validation artifact.
    """
    try:
        logger.info("ML Pipeline Execution Started")
        logger.info("=" * 70)

        # ------------------------------------------------------------------
        # DATA INGESTION STAGE
        # ------------------------------------------------------------------
        data_ingestion_config = IngestionConfig(dataset_path=dataset_path)
        data_ingestion = DataIngestion(config=data_ingestion_config) 

        logger.info("Running Data Ingestion Stage...")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info("Data Ingestion Stage Completed Successfully")

        # ------------------------------------------------------------------
        # DATA VALIDATION STAGE
        # ------------------------------------------------------------------
        data_validation_config = ValidationConfig()
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                         data_validation_config=data_validation_config)

        logger.info("Running Data Validation Stage...")
        data_validation_artifact  = data_validation.initiate_data_validation()
        logger.info("Data Validation Stage Completed Successfully")

        logger.info("=" * 70)
        logger.info("ML Pipeline Finished Successfully")


        # ------------------------------------------------------------------
        # PREPROCESSING STAGE
        # ------------------------------------------------------------------
        preprocessing_config = PreprossingConfig()
        preprocessing = Preprocessing(preprocessing_config=preprocessing_config,
                                      data_ingestion_artifact=data_ingestion_artifact,
                                      data_validation_artifact=data_validation_artifact)

        logger.info("Running Preprocessing Stage...")
        preprocessing_artifact = preprocessing.initiate_preprocessing()
        logger.info("Preprocessing Stage Completed Successfully")

        logger.info("=" * 70)
        logger.info("DATA PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
            
        return preprocessing_artifact

    except Exception as e:
        logger.error(f"Pipeline failed due to: {str(e)}", exc_info=True)
        raise ProjectException(e, sys)

