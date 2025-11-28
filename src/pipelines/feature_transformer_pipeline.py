import sys
from src.logging.logger import get_logger
from src.constants.paths import dataset_path
from src.exceptions.exception import ProjectException

from src.entity.components_config_entity import TransformationConfig
from src.components.feature_transformer import FeatureTransformer

from src.entity.artifact_entity import (
    IngestionArtifact,
    ValidationArtifact,
    PreprocessingArtifact,
    FeatureEngineeringArtifact,
    TransformationArtifact
)


logger = get_logger(__name__)


def run_feature_transformer_pipeline(feature_engineering_artifact) -> None:
        try:
            transformation_config = TransformationConfig()

            feature_transformer = FeatureTransformer(
                transformation_config=transformation_config, 
                feature_engineering_artifact=feature_engineering_artifact)
            
            feature_transformer_artifact = feature_transformer.initiate_feature_transformation()

            return feature_transformer_artifact
        
            logger.info("Feature Transformer Completed")

        except Exception as e:
            raise ProjectException(e, sys)
        
if __name__ == "__main__":

    try:
        # Import pipelines
        from src.pipelines.data_ingestion_pipeline import run_ingestion_pipeline
        from src.pipelines.data_validation_pipeline import run_validation_pipeline
        from src.pipelines.data_preprocessing_pipeline import run_preprocessing_pipeline
        from src.pipelines.feature_engineering_pipeline import run_feature_engineering_pipeline


        dataset_path = dataset_path

        # 1. Run Data Ingestion
        ingestion_artifact = run_ingestion_pipeline(dataset_path)

        # 2. Run Data Validation
        validation_artifact = run_validation_pipeline(ingestion_artifact)

        # 3. Run Preprocessing
        preprocessing_artifact = run_preprocessing_pipeline(ingestion_artifact,validation_artifact)

        # 4. Run Feature Engineering
        feature_engineering_artifact = run_feature_engineering_pipeline(preprocessing_artifact)

        # 5. Run Feature Transformer
        feature_transformer_artifact = run_feature_transformer_pipeline(feature_engineering_artifact)

        print(feature_transformer_artifact)

    except Exception as e:
        raise ProjectException(e, sys)
    
# python src/pipelines/feature_transformer_pipeline.py
