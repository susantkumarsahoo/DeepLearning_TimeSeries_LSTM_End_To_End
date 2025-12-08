import sys
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.artifact_entity import TrainingArtifact, DeploymentArtifact
from src.entity.model_config_entity import ModelDeploymentConfig
from src.models.model_registry import ModelRegistry
from src.constants.paths import dataset_path

logger = get_logger(__name__)


def run_model_registry_pipeline(training_artifact: TrainingArtifact):
    try:
        logger.info("=== MODEL REGISTRY PIPELINE STARTED ===")

        # Prepare configuration
        model_registry_config = ModelDeploymentConfig()

        # DeploymentArtifact is required by ModelRegistry __init__
        deployment_artifact = DeploymentArtifact(
            deployed_model_file=None,
            deployed_preprocessor_file=None,
            deployment_report_file=None
        )

        # Initialize Model Registry
        model_registry = ModelRegistry(
            training_artifact=training_artifact,
            model_registry_config=model_registry_config,
            deployment_artifact=deployment_artifact
        )

        # Execute registry process
        model_registry_artifact = model_registry.initiate_model_registry()

        logger.info("=== MODEL REGISTRY PIPELINE COMPLETED ===")
        return model_registry_artifact

    except Exception as e:
        logger.error(f"Model Registry Failed: {str(e)}", exc_info=True)
        raise ProjectException(e, sys)


if __name__ == '__main__':
    try:
        # Import pipelines
        from src.pipelines.data_ingestion_pipeline import run_ingestion_pipeline
        from src.pipelines.data_validation_pipeline import run_validation_pipeline
        from src.pipelines.data_preprocessing_pipeline import run_preprocessing_pipeline
        from src.pipelines.feature_engineering_pipeline import run_feature_engineering_pipeline
        from src.pipelines.feature_transformer_pipeline import run_feature_transformer_pipeline
        from src.pipelines.model_training_pipeline import run_initiate_model_training
        from src.pipelines.model_evaluator_pipeline import run_initiate_model_evaluation

        # 1. Data Ingestion
        ingestion_artifact = run_ingestion_pipeline(dataset_path)

        # 2. Data Validation
        validation_artifact = run_validation_pipeline(ingestion_artifact)

        # 3. Preprocessing
        preprocessing_artifact = run_preprocessing_pipeline(ingestion_artifact, validation_artifact)

        # 4. Feature Engineering
        feature_engineering_artifact = run_feature_engineering_pipeline(preprocessing_artifact)

        # 5. Feature Transformer
        feature_transformer_artifact = run_feature_transformer_pipeline(feature_engineering_artifact)

        # 6. Model Training
        model_training_artifact = run_initiate_model_training(feature_transformer_artifact)

        # 7. Model Evaluation
        model_evaluation_artifact = run_initiate_model_evaluation(
            feature_transformer_artifact,
            model_training_artifact
        )

        # 8. Model Registry
        model_registry_artifact = run_model_registry_pipeline(model_training_artifact)

    except Exception as e:
        raise ProjectException(e, sys)

    
# python src/pipelines/registry_pipeline.py