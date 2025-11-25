import sys
from src.logging.logger import get_logger
from src.constants.paths import dataset_path
from src.exceptions.exception import ProjectException
from src.entity.components_config_entity import FeatureEngineeringConfig
from src.components.feature_engineering import FeatureEngineering

logger = get_logger(__name__)


def run_feature_engineering_pipeline(preprocessing_artifact):
    try:
        logger.info("Feature Engineering Initiated")
        featurengineering_config = FeatureEngineeringConfig()
        feature_engineering = FeatureEngineering(
            featurengineering_config=featurengineering_config,
            preprocessing_artifact=preprocessing_artifact)
        feature_engineering_artifact = feature_engineering.initiate_feature_engineering()
        logger.info("Feature Engineering Completed")
        return feature_engineering_artifact
    except Exception as e:
        raise ProjectException(e, sys)
    
if __name__ == "__main__":

    from src.pipelines.data_preprocessing_pipeline import run_preprocessing_pipeline
    preprocessing_artifact = run_preprocessing_pipeline(dataset_path)
    preprocessing_artifact = preprocessing_artifact
    feature_engineering_artifact = run_feature_engineering_pipeline(preprocessing_artifact)
    print(feature_engineering_artifact)


    # python src/pipelines/feature_engineering_pipeline.py