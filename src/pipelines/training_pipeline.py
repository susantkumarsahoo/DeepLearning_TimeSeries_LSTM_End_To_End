import sys
from src.logging.logger import get_logger
from src.constants.paths import dataset_path as default_dataset_path
from src.exceptions.exception import ProjectException

# Import all pipeline functions
from src.pipelines.data_ingestion_pipeline import run_ingestion_pipeline
from src.pipelines.data_validation_pipeline import run_validation_pipeline
from src.pipelines.data_preprocessing_pipeline import run_preprocessing_pipeline
from src.pipelines.feature_engineering_pipeline import run_feature_engineering_pipeline
from src.pipelines.feature_transformer_pipeline import run_feature_transformer_pipeline
from src.pipelines.model_training_pipeline import run_initiate_model_training
from src.pipelines.model_evaluator_pipeline import run_initiate_model_evaluation
from src.pipelines.registry_pipeline import run_model_registry_pipeline

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Complete End-to-End ML Training Pipeline
    Orchestrates all components from data ingestion to model registry
    """

    def __init__(self, dataset_path: str = None):
        """
        Initialize Training Pipeline
        
        Args:
            dataset_path: Path to the dataset. If None, uses default from constants
        """
        self.dataset_path = dataset_path if dataset_path else default_dataset_path
        logger.info(f"Training Pipeline initialized with dataset path: {self.dataset_path}")

    def run_pipeline(self):
        """
        Execute complete training pipeline
        
        Returns:
            dict: Dictionary containing all artifacts from each pipeline stage
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING COMPLETE TRAINING PIPELINE")
            logger.info("=" * 80)

            # Store all artifacts
            artifacts = {}

            # Step 1: Data Ingestion
            logger.info("\n[STEP 1/8] Running Data Ingestion Pipeline...")
            ingestion_artifact = run_ingestion_pipeline(self.dataset_path)
            artifacts['ingestion'] = ingestion_artifact
            logger.info(f"[SUCCESS] Data Ingestion Completed: {ingestion_artifact}")

            # Step 2: Data Validation
            logger.info("\n[STEP 2/8] Running Data Validation Pipeline...")
            validation_artifact = run_validation_pipeline(ingestion_artifact)
            artifacts['validation'] = validation_artifact
            logger.info(f"[SUCCESS] Data Validation Completed: {validation_artifact}")

            # Step 3: Data Preprocessing
            logger.info("\n[STEP 3/8] Running Data Preprocessing Pipeline...")
            preprocessing_artifact = run_preprocessing_pipeline(
                ingestion_artifact, 
                validation_artifact
            )
            artifacts['preprocessing'] = preprocessing_artifact
            logger.info(f"[SUCCESS] Data Preprocessing Completed: {preprocessing_artifact}")

            # Step 4: Feature Engineering
            logger.info("\n[STEP 4/8] Running Feature Engineering Pipeline...")
            feature_engineering_artifact = run_feature_engineering_pipeline(
                preprocessing_artifact
            )
            artifacts['feature_engineering'] = feature_engineering_artifact
            logger.info(f"[SUCCESS] Feature Engineering Completed: {feature_engineering_artifact}")

            # Step 5: Feature Transformation
            logger.info("\n[STEP 5/8] Running Feature Transformation Pipeline...")
            feature_transformer_artifact = run_feature_transformer_pipeline(
                feature_engineering_artifact
            )
            artifacts['feature_transformation'] = feature_transformer_artifact
            logger.info(f"[SUCCESS] Feature Transformation Completed: {feature_transformer_artifact}")

            # Step 6: Model Training
            logger.info("\n[STEP 6/8] Running Model Training Pipeline...")
            model_training_artifact = run_initiate_model_training(
                feature_transformer_artifact
            )
            artifacts['model_training'] = model_training_artifact
            logger.info(f"[SUCCESS] Model Training Completed: {model_training_artifact}")

            # Step 7: Model Evaluation
            logger.info("\n[STEP 7/8] Running Model Evaluation Pipeline...")
            model_evaluation_artifact = run_initiate_model_evaluation(
                feature_transformer_artifact,
                model_training_artifact
            )
            artifacts['model_evaluation'] = model_evaluation_artifact
            logger.info(f"[SUCCESS] Model Evaluation Completed: {model_evaluation_artifact}")

            # Step 8: Model Registry
            logger.info("\n[STEP 8/8] Running Model Registry Pipeline...")
            model_registry_artifact = run_model_registry_pipeline(
                model_training_artifact
            )
            artifacts['model_registry'] = model_registry_artifact
            logger.info(f"[SUCCESS] Model Registry Completed: {model_registry_artifact}")

            # Pipeline completion
            logger.info("\n" + "=" * 80)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            self._print_summary(artifacts)
            
            return artifacts

        except Exception as e:
            logger.error(f"Training Pipeline Failed: {str(e)}", exc_info=True)
            raise ProjectException(e, sys)

    def _print_summary(self, artifacts: dict):
        """
        Print summary of all pipeline artifacts
        
        Args:
            artifacts: Dictionary containing all pipeline artifacts
        """
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        for stage_name, artifact in artifacts.items():
            logger.info(f"\n{stage_name.upper().replace('_', ' ')}:")
            logger.info(f"  {artifact}")


def main():
    """
    Main function to execute training pipeline
    """
    try:
        # Initialize and run training pipeline
        pipeline = TrainingPipeline(dataset_path=default_dataset_path)
        artifacts = pipeline.run_pipeline()
        
        # Print final results
        print("\n" + "=" * 80)
        print("FINAL ARTIFACTS")
        print("=" * 80)
        for stage, artifact in artifacts.items():
            print(f"\n{stage}:")
            print(f"  {artifact}")
        
        return artifacts

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
        raise ProjectException(e, sys)


