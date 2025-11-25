from dataclasses import dataclass

# ----------------------------------------------------------------------
# DATA INGESTION ARTIFACT
# ----------------------------------------------------------------------
@dataclass
class IngestionArtifact:
    raw_data_file: str
    processed_data_file: str
    train_file: str
    test_file: str
    metadata_file: str
    schema_file: str


# ----------------------------------------------------------------------
# DATA VALIDATION ARTIFACT
# ----------------------------------------------------------------------
@dataclass
class ValidationArtifact:   
    drift_report_file: str
    validation_status_file: str
    quality_report_file: str
    schema_validation_file: str
    statistical_validation_file: str
    time_series_validation_file: str


# ----------------------------------------------------------------------
# DATA PREPROCESSING ARTIFACT
# ----------------------------------------------------------------------
@dataclass
class PreprocessingArtifact:
    train_preprocessed_file: str
    test_preprocessed_file: str
    preprocessing_report_file: str


# ----------------------------------------------------------------------
# FEATURE ENGINEERING ARTIFACT
# ----------------------------------------------------------------------
@dataclass
class FeatureEngineeringArtifact:
    train_feature_file: str
    test_feature_file: str
    feature_report_file: str


# ----------------------------------------------------------------------
# DATA TRANSFORMATION ARTIFACT
# ----------------------------------------------------------------------
@dataclass
class TransformationArtifact:
    train_transformed_file: str
    test_transformed_file: str
    transformer_object_file: str
    transformation_report_file: str


# ----------------------------------------------------------------------
# MODEL TRAINING ARTIFACT
# ----------------------------------------------------------------------
@dataclass
class TrainingArtifact:
    trained_model_file: str
    training_report_file: str


# ----------------------------------------------------------------------
# MODEL EVALUATION ARTIFACT
# ----------------------------------------------------------------------
@dataclass
class EvaluationArtifact:
    evaluation_report_file: str
    accepted: bool


# ----------------------------------------------------------------------
# MODEL DEPLOYMENT ARTIFACT
# ----------------------------------------------------------------------
@dataclass
class DeploymentArtifact:
    deployed_model_file: str
    deployed_preprocessor_file: str
    deployment_report_file: str


# ----------------------------------------------------------------------
# DATABASE ARTIFACT
# ----------------------------------------------------------------------
@dataclass
class DatabaseArtifact:
    database_file: str
    collection_file: str
    database_report_file: str


# ----------------------------------------------------------------------
# LOGGING ARTIFACT
# ----------------------------------------------------------------------
@dataclass
class LogArtifact:
    log_file: str
    log_report_file: str
