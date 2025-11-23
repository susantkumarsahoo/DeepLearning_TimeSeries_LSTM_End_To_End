import os
from datetime import datetime

# =====================================================================
# ARTIFACT DIRECTORIES
# =====================================================================
ARTIFACTS_DIR          = os.path.join("artifacts")
DEPLOYED_ARTIFACTS_DIR = os.path.join("deployed_artifacts")

# Timestamp
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# =====================================================================
# DATA INGESTION CONSTANTS
# =====================================================================
DATA_INGESTION_DIR            = "data_ingestion"
DATA_INGESTION_RAW_DIR        = "raw"
DATA_INGESTION_PROCESSED_DIR  = "processed"
DATA_INGESTION_SPLIT_DIR      = "split"

DATA_INGESTION_RAW_FILE       = "raw_data.csv"
DATA_INGESTION_PROCESSED_FILE = "processed_data.csv"
DATA_INGESTION_TRAIN_FILE     = "train.csv"
DATA_INGESTION_TEST_FILE      = "test.csv"

DATA_INGESTION_METADATA_FILE  = "metadata.json"
DATA_INGESTION_SCHEMA_FILE    = "schema.json"

# Parameters
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# =====================================================================
# DATA VALIDATION CONSTANTS
# =====================================================================
DATA_VALIDATION_DIR                 = "data_validation"
DATA_VALIDATION_DRIFT_REPORT        = "data_drift_report.json"
DATA_VALIDATION_STATUS_FILE         = "validation_status.json"
DATA_VALIDATION_QUALITY_REPORT      = "data_quality_integrity_validation.json"
SCHEMA_STRUCTURE_VALIDATION_REPORT  = "schema_structure_validation.json"
STATISTICAL_VALIDATION_REPORT       = "statistical_validation.json"
TIME_SERIES_VALIDATION_REPORT       = "time_series_validation.json"

# Validation Thresholds
MAX_MISSING_THRESHOLD          = 0.05
NUMERICAL_COLUMN_THRESHOLD     = 0.05
CATEGORICAL_COLUMN_THRESHOLD   = 0.05
MAX_DUPLICATE_THRESHOLD        = 0.05
OUTLIER_IQR_THRESHOLD          = 1.5
DRIFT_THRESHOLD                = 0.05
MIN_CORRELATION_THRESHOLD      = 0.10
MAX_SKEWNESS_THRESHOLD         = 3.0


# =====================================================================
# DATA PREPROCESSING CONSTANTS
# =====================================================================
DATA_PREPROCESSING_DIR          = "data_preprocessing"
DATA_PREPROCESSING_TRAIN_FILE   = "train_preprocessed.csv"
DATA_PREPROCESSING_TEST_FILE    = "test_preprocessed.csv"
DATA_PREPROCESSING_REPORT_FILE  = "preprocessing_report.json"


# =====================================================================
# FEATURE ENGINEERING CONSTANTS
# =====================================================================
FEATURE_ENGINEERING_DIR            = "feature_engineering"
FEATURE_ENGINEERING_TRAIN_FILE     = "train_features.csv"
FEATURE_ENGINEERING_TEST_FILE      = "test_features.csv"
FEATURE_ENGINEERING_REPORT_FILE    = "feature_engineering_report.json"


# =====================================================================
# DATA TRANSFORMATION CONSTANTS
# =====================================================================
DATA_TRANSFORMATION_DIR          = "data_transformation"
DATA_TRANSFORMATION_X_FILE       = "features_X.npy"
DATA_TRANSFORMATION_Y_FILE       = "target_y.npy"
DATA_TRANSFORMATION_OBJECT_FILE  = "preprocessor.pkl"
DATA_TRANSFORMATION_REPORT_FILE  = "transformation_report.json"


# =====================================================================
# MODEL TRAINING CONSTANTS
# =====================================================================
MODEL_TRAINING_DIR         = "model_training"
MODEL_TRAINING_BEST_FILE   = "best_model.pkl"
MODEL_TRAINING_REPORT_FILE = "model_report.json"


# =====================================================================
# MODEL EVALUATION CONSTANTS
# =====================================================================
MODEL_EVALUATION_DIR         = "model_evaluation"
MODEL_EVALUATION_REPORT_FILE = "model_evaluation_report.json"


# =====================================================================
# MODEL DEPLOYMENT CONSTANTS
# =====================================================================
MODEL_DEPLOYMENT_DIR               = "deployment_artifacts"
MODEL_DEPLOYMENT_MODEL_FILE        = "best_model.pkl"
MODEL_DEPLOYMENT_PREPROCESSOR_FILE = "preprocessor.pkl"
MODEL_DEPLOYMENT_REPORT_FILE       = "model_deployment_report.json"


# =====================================================================
# DATABASE CONSTANTS
# =====================================================================
DATABASE_DIR         = "database"
DATABASE_NAME        = "student_performance_db"
DATABASE_FILE        = "student_performance_db.csv"
DATABASE_COLLECTION  = "student_performance_collection"
DATABASE_REPORT_FILE = "database_report.json"


# =====================================================================
# LOGGING CONSTANTS
# =====================================================================
LOGGING_DIR         = "logs"
LOGGING_LOGGER_NAME = "logger"
LOGGING_REPORT_FILE = "application.log"


# =====================================================================
# TARGET COLUMN
# =====================================================================
TARGET_COLUMN = "xxxxxxxxx"

# =====================================================================
# DATA PATHS
# =====================================================================
dataset_path = r"C:\Users\TPWODL\New folder_Content\DeepLearning_TimeSeries_LSTM_End_To_End\data\raw\Energy Demand Hourly.csv"
