import os
from datetime import datetime

# Artifact Directories
ARTIFACTS_DIR             = os.path.join("artifacts")
DEPLOYED_ARTIFACTS_DIR    = os.path.join("deployed_artifacts")

# Data Ingestion Constants
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

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# Data Validation Constants
DATA_VALIDATION_DIR              = "data_validation"
DATA_VALIDATION_DRIFT_REPORT     = "data_drift_report.json"
DATA_VALIDATION_STATUS_FILE      = "validation_status.json"
DATA_VALIDATION_MISSING_FILE     = "missing_columns.json"
DATA_VALIDATION_TYPES_FILE       = "data_types.json"
DATA_VALIDATION_REPORT_FILE      = "validation_report.json"


# Validation Thresholds
MAX_MISSING_THRESHOLD        = 0.3
DRIFT_THRESHOLD              = 0.05
NUMERICAL_COLUMN_THRESHOLD   = 0.5
CATEGORICAL_COLUMN_THRESHOLD = 0.5


# Data Preprocessing Constants
DATA_PREPROCESSING_DIR          = "data_preprocessing"
DATA_PREPROCESSING_TRAIN_FILE   = "train_preprocessed.csv"
DATA_PREPROCESSING_TEST_FILE    = "test_preprocessed.csv"
DATA_PREPROCESSING_REPORT_FILE  = "preprocessing_report.json"


# Feature Engineering Constants
FEATURE_ENGINEERING_DIR            = "feature_engineering"
FEATURE_ENGINEERING_TRAIN_FILE     = "train_features.csv"
FEATURE_ENGINEERING_TEST_FILE      = "test_features.csv"
FEATURE_ENGINEERING_REPORT_FILE    = "feature_engineering_report.json"


# Data Transformation Constants
DATA_TRANSFORMATION_DIR          = "data_transformation"
DATA_TRANSFORMATION_X_FILE       = "features_X.npy"
DATA_TRANSFORMATION_Y_FILE       = "target_y.npy"
DATA_TRANSFORMATION_OBJECT_FILE  = "preprocessor.pkl"
DATA_TRANSFORMATION_REPORT_FILE  = "transformation_report.json"


# Model Training Constants
MODEL_TRAINING_DIR        = "model_training"
MODEL_TRAINING_BEST_FILE  = "best_model.pkl"
MODEL_TRAINING_REPORT_FILE = "model_report.json"


# Model Evaluation Constants
MODEL_EVALUATION_DIR         = "model_evaluation"
MODEL_EVALUATION_REPORT_FILE = "model_evaluation_report.json"


# Model Deployment Constants
MODEL_DEPLOYMENT_DIR             = "deployment_artifacts"
MODEL_DEPLOYMENT_MODEL_FILE      = "best_model.pkl"
MODEL_DEPLOYMENT_PREPROCESSOR_FILE = "preprocessor.pkl"
MODEL_DEPLOYMENT_REPORT_FILE     = "model_deployment_report.json"


# Database Constants
DATABASE_DIR              = "database"
DATABASE_NAME             = "student_performance_db"
DATABASE_FILE             = "student_performance_db.csv"
DATABASE_COLLECTION       = "student_performance_collection"
DATABASE_REPORT_FILE      = "database_report.json"


# Logging Constants
LOGGING_DIR         = "logs"
LOGGING_LOGGER_NAME = "logger"
LOGGING_REPORT_FILE = "application.log"


# Target column
TARGET_COLUMN = "math_score"