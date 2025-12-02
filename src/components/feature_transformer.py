import sys
import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict
from src.constants.paths import *

import numpy as np
import pandas as pd

from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException


from src.entity.components_config_entity import TransformationConfig
from src.entity.artifact_entity import FeatureEngineeringArtifact, TransformationArtifact
from src.utils.helpers import save_json
from src.utils.feature_transformer_helper import clean_features,load_train_test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

logger = get_logger(__name__)


class FeatureTransformer:
    def __init__(self, transformation_config: TransformationConfig, feature_engineering_artifact: FeatureEngineeringArtifact) -> None:

        try:
            self.transformation_config = transformation_config
            self.feature_engineering_artifact = feature_engineering_artifact
            logger.info("Feature Transformer Initialized")

        except Exception as e:
            raise ProjectException(e, sys)
        

    def initiate_feature_transformation(self) -> TransformationArtifact:
        try:
            logger.info("Feature Transformation Initiated")

            # load data
            train_df = pd.read_csv(self.feature_engineering_artifact.train_feature_file)
            test_df = pd.read_csv(self.feature_engineering_artifact.test_feature_file)



            
        

            # drop time features
            train_clean, test_clean, info_report = clean_features(train_df, test_df, cols_to_drop)

            # save data
            train_clean.to_csv(self.transformation_config.train_transformation_path, index=False)
            test_clean.to_csv(self.transformation_config.test_transformation_path, index=False)

            # save report
            
            save_json(info_report, self.transformation_config.transformation_report_path)

            # save artifact
            transformation_artifact = TransformationArtifact(
                train_transformed_file=self.transformation_config.train_transformation_path,
                test_transformed_file=self.transformation_config.test_transformation_path,
                transformation_report_file=self.transformation_config.transformation_report_path
            )

            logger.info("Feature Transformation Completed")

            return transformation_artifact

        except Exception as e:
            raise ProjectException(e, sys)




