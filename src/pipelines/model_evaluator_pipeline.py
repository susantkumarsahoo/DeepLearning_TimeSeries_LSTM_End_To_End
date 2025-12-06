import sys
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.artifact_entity import TransformationArtifact,TrainingArtifact
from src.entity.model_config_entity import ModelEvaluationConfig
from src.models.evaluator import ModelEvaluation
from src.constants.paths import dataset_path

logger = get_logger(__name__)

def run_initiate_model_evaluation(transformation_artifact:TransformationArtifact,training_artifact:TrainingArtifact):
    try:



        model_evaluation_config = ModelEvaluationConfig()

        model_evaluation = ModelEvaluation(model_evaluation_config=model_evaluation_config,
                                           transformation_artifact=transformation_artifact, 
                                           model_training_artifact=training_artifact)
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

        return model_evaluation_artifact
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
        from src.pipelines.model_training_pipeline import run_initiate_model_training


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
        model_training_artifact = run_initiate_model_training(feature_transformer_artifact)

        # 7. Run Model Evaluation
        model_evaluation_artifact = run_initiate_model_evaluation(feature_transformer_artifact, model_training_artifact)


    except Exception as e:
        raise ProjectException(e, sys)

# python src/pipelines/model_evaluator_pipeline.py
