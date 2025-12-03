import sys
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.artifact_entity import TransformationArtifact
from src.entity.model_config_entity import ModelTrainingConfig
from src.models.trainer import ModelTrainer
from src.constants.paths import dataset_path

logger = get_logger(__name__)


def initiate_model_training(transformation_artifact) -> None:
    try:
        logger.info("Model Training Initiated")

        model_training_config = ModelTrainingConfig()

        model_trainer = ModelTrainer(
            transformation_artifact=transformation_artifact,
            model_training_config=model_training_config
        )

        model_training_artifact = model_trainer.initiate_model_training()

        logger.info("Model Training Completed")

        return model_training_artifact

    except Exception as e:
        raise ProjectException(e, sys)

    
if __name__ == '__main__':


    try:
        # Import pipelines
        from src.pipelines.data_ingestion_pipeline import run_ingestion_pipeline
        from src.pipelines.data_validation_pipeline import run_validation_pipeline
        from src.pipelines.data_preprocessing_pipeline import run_preprocessing_pipeline
        from src.pipelines.feature_engineering_pipeline import run_feature_engineering_pipeline
        from src.pipelines.feature_transformer_pipeline import run_feature_transformer_pipeline


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

        # 6. Run Model Training
        model_training_artifact = initiate_model_training(feature_transformer_artifact)


    except Exception as e:
        raise ProjectException(e, sys)
    
# python src/pipelines/feature_transformer_pipeline.py

