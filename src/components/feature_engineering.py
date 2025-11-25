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
from src.entity.components_config_entity import FeatureEngineeringConfig   
from src.entity.artifact_entity import PreprocessingArtifact, FeatureEngineeringArtifact
from src.utils.features_engineering_helpers import add_cyclic_features
from src.utils.helpers import save_json


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


logger = get_logger(__name__)



class FeatureEngineering:
    def __init__(self, feature_engineering_config: FeatureEngineeringConfig, preprocessing_artifact: PreprocessingArtifact):
        try:
            self.feature_engineering_config = feature_engineering_config
            self.preprocessing_artifact = preprocessing_artifact
            logger.info("Feature Engineering Initialized")
        except Exception as e:
            raise ProjectException(e, sys)


    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        try:
            logger.info("Feature Engineering Initiated")
            logger.info("Feature Engineering Completed")

            # lode data
            train_preprocessed_df = pd.read_csv(self.preprocessing_artifact.train_preprocessed_file)
            test_preprocessed_df = pd.read_csv(self.preprocessing_artifact.train_preprocessed_file)

            # add cyclic features
            train_df, report = add_cyclic_features(train_preprocessed_df)
            test_df, report = add_cyclic_features(test_preprocessed_df)

            # save report
            save_json(report, self.feature_engineering_config.feature_engineering_report_path)


            # save data
            train_df.to_csv(self.feature_engineering_config.train_feature_engineering_path, index=False)
            test_df.to_csv(self.feature_engineering_config.test_feature_engineering_path, index=False)

            feature_engineering_artifact = FeatureEngineeringArtifact(
                train_feature_file=self.feature_engineering_config.train_feature_engineering_path,
                test_feature_file=self.feature_engineering_config.test_feature_engineering_path,
                feature_report_file=self.feature_engineering_config.feature_engineering_report_path
            )
            logger.info("Feature Engineering Artifact Created")

            return feature_engineering_artifact
        except Exception as e:
            raise ProjectException(e, sys)
        
    

