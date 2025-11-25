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
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.seasonal import seasonal_decompose


from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.components_config_entity import ValidationConfig, PreprossingConfig
from src.entity.artifact_entity import IngestionArtifact, ValidationArtifact, PreprocessingArtifact
from src.utils.helpers import save_json,save_json_new,save_json_data
from src.utils.pre_helper import (generate_data_profile,train_test_split, add_time_features,generate_correlation_report,detect_outliers_iqr,
                                  analyze_seasonal_decomposition,variance_threshold_report,save_readable_report,multicollinearity_vif_report)




logger = get_logger(__name__)


class Preprocessing:
    def __init__(self,
        preprocessing_config: PreprossingConfig,
        data_ingestion_artifact: IngestionArtifact,
        data_validation_artifact: ValidationArtifact) -> None:
        """
        Initialize preprocessing component
        """
        try:
            self.preprocessing_config = preprocessing_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

            logger.info("Preprocessing initialized successfully.")

        except Exception as e:
            raise ProjectException(e, sys)


    def initiate_preprocessing(self) -> PreprocessingArtifact:
        """
        Execute preprocessing pipeline:
        1. Load train & test files
        2. Merge
        3. Generate data profile JSON
        4. Train-test split again
        5. Save processed train/test
        6. Return PreprocessingArtifact
        """
        try:

            # -------------------------------------------------------
            # 1. Validation gate
            # -------------------------------------------------------
            if not self.data_validation_artifact.validation_status_file:
                raise ProjectException(
                    "Data validation failed. Cannot proceed to preprocessing.",
                    sys
                )

            logger.info("Preprocessing started...")

            # -------------------------------------------------------
            # 2. Load train and test files
            # -------------------------------------------------------
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file)

            # Merge into one dataset
            df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M', errors='coerce')
            logger.info(f"Merged dataset shape: {df.shape}")

            # -------------------------------------------------------
            # 3. Generate data profile JSON
            # -------------------------------------------------------
            data_profile = generate_data_profile(df)

            logger.info("Data profile JSON saved successfully.")

            # Add time features
            df = add_time_features(df)

            logger.info("Time features added successfully.")

            # Generate correlation report
            correlation_report = generate_correlation_report(df)

            logger.info("Correlation report saved successfully.")

            # Detect outliers
            outlayers_report = detect_outliers_iqr(df,column='megawatthours',json_output_path=None)

            logger.info("Outliers detected successfully.")

            # sesonal decomposition
            decomposition_report = analyze_seasonal_decomposition(df,column_name='megawatthours' , model='additive', period=12)

            # variance threshold
            variance_threshold = variance_threshold_report(df, threshold=0.01, json_path=None)

            # variance_inflation_factor
            variance_inflation = multicollinearity_vif_report(df, target_column="megawatthours")

            # 1. Create final report dictionary
            final_report = {
                "data_profile": data_profile,
                "correlation_report": correlation_report,
                "outliers_report": outlayers_report,
                "decomposition_report": decomposition_report,
                "variance_threshold": variance_threshold,
                "variance_inflation": variance_inflation
            }


            # 3. Save readable text report (human-friendly)
            save_json_new(
                final_report,
                self.preprocessing_config.preprocessing_report_path
            )
            logger.info(f"JSON report saved successfully at: {self.preprocessing_config.preprocessing_report_path}")

            # -------------------------------------------------------
            # 4. Re-split processed dataset
            # -------------------------------------------------------
            processed_train_df, processed_test_df = train_test_split(
                df,
                test_size=TRAIN_TEST_SPLIT_RATIO,
                random_state=RANDOM_STATE)

            # -------------------------------------------------------
            # 5. Save processed train & test datasets
            # -------------------------------------------------------
            processed_train_df.to_csv(
                self.preprocessing_config.train_preprocessed_path,
                index=False
            )

            processed_test_df.to_csv(
                self.preprocessing_config.test_preprocessed_path,
                index=False
            )

            logger.info("Processed train/test datasets saved successfully.")

            # -------------------------------------------------------
            # 6. Create Preprocessing Artifact
            # -------------------------------------------------------
            preprocessing_artifact = PreprocessingArtifact(
                train_preprocessed_file=self.preprocessing_config.train_preprocessed_path,
                test_preprocessed_file=self.preprocessing_config.test_preprocessed_path,
                preprocessing_report_file=self.preprocessing_config.preprocessing_report_path
            )
            logger.info("Preprocessing completed successfully.")
            return preprocessing_artifact

        except Exception as e:
            raise ProjectException(e, sys)
        



