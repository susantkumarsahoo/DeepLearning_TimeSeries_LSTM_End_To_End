import sys
from src.logging.logger import get_logger
from src.constants.paths import dataset_path
from src.exceptions.exception import ProjectException

from src.entity.components_config_entity import FeatureEngineeringConfig
from src.components.feature_engineering import FeatureEngineering

from src.entity.artifact_entity import (
    IngestionArtifact,
    ValidationArtifact,
    PreprocessingArtifact
)

logger = get_logger(__name__)


def run_feature_engineering_pipeline(preprocessing_artifact):
    """
    Execute Feature Engineering Pipeline
    """
    try:
        logger.info("=== FEATURE ENGINEERING STARTED ===")

        feature_engineering_config = FeatureEngineeringConfig()

        feature_engineering = FeatureEngineering(
            feature_engineering_config=feature_engineering_config,
            preprocessing_artifact=preprocessing_artifact
        )

        feature_engineering_artifact = feature_engineering.initiate_feature_engineering()

        logger.info("=== FEATURE ENGINEERING COMPLETED ===")
        return feature_engineering_artifact

    except Exception as e:
        raise ProjectException(e, sys)


if __name__ == "__main__":
    try:
        # Import pipelines
        from src.pipelines.data_ingestion_pipeline import run_ingestion_pipeline
        from src.pipelines.data_validation_pipeline import run_validation_pipeline
        from src.pipelines.data_preprocessing_pipeline import run_preprocessing_pipeline

        dataset_path = dataset_path

        # 1. Run Data Ingestion
        ingestion_artifact = run_ingestion_pipeline(dataset_path)

        # 2. Run Data Validation
        validation_artifact = run_validation_pipeline(ingestion_artifact)

        # 3. Run Preprocessing
        preprocessing_artifact = run_preprocessing_pipeline(ingestion_artifact,validation_artifact)

        # 4. Run Feature Engineering
        feature_engineering_artifact = run_feature_engineering_pipeline(preprocessing_artifact)

        print(feature_engineering_artifact)

    except Exception as e:
        raise ProjectException(e, sys)





    # python src/pipelines/feature_engineering_pipeline.py