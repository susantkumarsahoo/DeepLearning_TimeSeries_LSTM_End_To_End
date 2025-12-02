import os
import sys
from src.constants.paths import *
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.model_config_entity import ModelTrainingConfig
from src.entity.artifact_entity import TransformationArtifact,TrainingArtifact
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from src.utils.helpers import save_json
from src.utils.model_training_helper import *

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, transformation_artifact: TransformationArtifact, model_training_config: ModelTrainingConfig) -> None:
        try:
            self.transformation_artifact = transformation_artifact
            self.model_training_config = model_training_config

        except Exception as e:
            raise ProjectException(e, sys)
        
    def initiate_model_training(self) -> TrainingArtifact:
        try:
            logger.info("Model Training Initiated")
            train_df = pd.read_csv(self.transformation_artifact.train_transformed_file)
            test_df = pd.read_csv(self.transformation_artifact.test_transformed_file)

            # train model
            model = self.model_training_config.model
            model.fit(train_df, test_df)

            # save model
            save_object(file_path=self.model_training_config.model_trainer_path, obj=model)

            # save metrics
            train_metrics = model.evaluate(train_df)
            test_metrics = model.evaluate(test_df)

            training_report = {
                "train_loss": train_metrics[0],
                "train_accuracy": train_metrics[1],
                "test_loss": test_metrics[0],
                "test_accuracy": test_metrics[1]
            }

            save_json(path=self.model_training_config.training_report_path, data=training_report)

            training_artifact = TrainingArtifact(
                trained_model_file=self.model_training_config.model_trainer_path,
                training_report_file=self.model_training_config.training_report_path
            )            

            return training_artifact
        except Exception as e:
            raise ProjectException(
                e, sys          
            )


