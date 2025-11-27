
"""
Data Validation Component
Performs comprehensive validation on training and validation datasets including:
- Schema and structure validation
- Data quality and integrity checks
- Statistical validation
- Data drift detection
- Time series validation
"""
import sys
import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict

from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.components_config_entity import ValidationConfig
from src.entity.artifact_entity import IngestionArtifact, ValidationArtifact
from src.utils.helpers import save_json


logger = get_logger(__name__)

class DataValidation:
    """
    Handles all data validation tasks including schema validation,
    data quality checks, statistical validation, and drift detection.
    """

    def __init__(self,data_ingestion_artifact: IngestionArtifact,data_validation_config: ValidationConfig):
        """
        Initialize DataValidation component.
        
        Args:
            data_ingestion_artifact: Artifact from data ingestion stage
            data_validation_config: Configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
                    
            logger.info("DataValidation initialized successfully")
            
        except Exception as e:
            raise ProjectException(e, sys)
        

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and validation datasets."""
        try:
            logger.info("Loading training and validation datasets")
            
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file)
            
            logger.info(f"Train shape: {train_df.shape}, Validation shape: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            raise ProjectException(e, sys)
        
    def validate_schema_structure(self,train_df: pd.DataFrame,test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate schema and structure consistency between train and validation sets.
        
        Args:
            train_df: Training dataframe
            test_df: Validation dataframe
            
        Returns:
            Dictionary containing schema validation results
        """
        try:
            logger.info("Starting schema and structure validation")
            
            validation_report = {
                "timestamp": datetime.now().isoformat(),
                "train_shape": train_df.shape,
                "val_shape": test_df.shape,
                "schema_match": True,
                "issues": []
            }
            
            # Check column names match
            train_cols = set(train_df.columns)
            val_cols = set(test_df.columns)
            
            if train_cols != val_cols:
                validation_report["schema_match"] = False
                missing_in_val = train_cols - val_cols
                extra_in_val = val_cols - train_cols
                
                if missing_in_val:
                    validation_report["issues"].append({
                        "type": "missing_columns_in_validation",
                        "columns": list(missing_in_val)
                    })
                
                if extra_in_val:
                    validation_report["issues"].append({
                        "type": "extra_columns_in_validation",
                        "columns": list(extra_in_val)
                    })
            
            # Check data types match
            dtype_mismatches = []
            common_cols = train_cols & val_cols
            
            for col in common_cols:
                if train_df[col].dtype != test_df[col].dtype:
                    dtype_mismatches.append({
                        "column": col,
                        "train_dtype": str(train_df[col].dtype),
                        "val_dtype": str(test_df[col].dtype)
                    })
            
            if dtype_mismatches:
                validation_report["schema_match"] = False
                validation_report["issues"].append({
                    "type": "dtype_mismatch",
                    "details": dtype_mismatches
                })
            
            # Column information
            validation_report["columns"] = {
                "total_columns": len(train_cols),
                "column_names": list(train_cols),
                "data_types": {col: str(dtype) for col, dtype in train_df.dtypes.items()}
            }
            
            logger.info(f"Schema validation completed. Match: {validation_report['schema_match']}")
            
            return validation_report
            
        except Exception as e:
            raise ProjectException(e, sys)
        
        
    def validate_data_quality(self,train_df: pd.DataFrame,test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality and integrity checks.
        
        Args:
            train_df: Training dataframe
            test_df: Validation dataframe
            
        Returns:
            Dictionary containing quality validation results
        """
        try:
            logger.info("Starting data quality validation")
            
            quality_report = {
                "timestamp": datetime.now().isoformat(),
                "train_quality": {},
                "val_quality": {},
                "overall_status": "PASSED"
            }
            
            for df_name, df in [("train", train_df), ("test", test_df)]:
                quality_metrics = {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "missing_values": {},
                    "duplicate_rows": int(df.duplicated().sum()),
                    "duplicate_percentage": round(df.duplicated().sum() / len(df) * 100, 2),
                    "constant_columns": [],
                    "high_cardinality_columns": [],
                    "outlier_detection": {}
                }
                
                # Missing values analysis
                for col in df.columns:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        quality_metrics["missing_values"][col] = {
                            "count": int(missing_count),
                            "percentage": round(missing_count / len(df) * 100, 2)
                        }
                
                # Constant columns (single unique value)
                for col in df.columns:
                    if df[col].nunique() == 1:
                        quality_metrics["constant_columns"].append(col)
                
                # High cardinality check for categorical columns
                for col in df.select_dtypes(include=['object']).columns:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio > 0.5:  # More than 50% unique values
                        quality_metrics["high_cardinality_columns"].append({
                            "column": col,
                            "unique_count": int(df[col].nunique()),
                            "unique_ratio": round(unique_ratio, 3)
                        })
                
                # Outlier detection for numerical columns using IQR method
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    
                    if outlier_count > 0:
                        quality_metrics["outlier_detection"][col] = {
                            "count": outlier_count,
                            "percentage": round(outlier_count / len(df) * 100, 2),
                            "lower_bound": float(lower_bound),
                            "upper_bound": float(upper_bound)
                        }
                
                quality_report[f"{df_name}_quality"] = quality_metrics
                
                # Check for critical issues
                if quality_metrics["duplicate_percentage"] > 10:
                    quality_report["overall_status"] = "WARNING"
                
                missing_pct = [v["percentage"] for v in quality_metrics["missing_values"].values()]
                if missing_pct and max(missing_pct) > 50:
                    quality_report["overall_status"] = "FAILED"
            
            logger.info(f"Data quality validation completed. Status: {quality_report['overall_status']}")
            
            return quality_report
            
        except Exception as e:
            raise ProjectException(e, sys)
        


    def validate_statistical_properties(self,train_df: pd.DataFrame,test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate statistical properties of numerical features.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            
        Returns:
            Dictionary containing statistical validation results
        """
        try:
            logger.info("Starting statistical validation")
            
            statistical_report = {
                "timestamp": datetime.now().isoformat(),
                "numerical_features": {},
                "statistical_tests": {},
                "warnings": []
            }
            
            numerical_cols = train_df.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                train_values = train_df[col].dropna()
                val_values = test_df[col].dropna()
                
                # Basic statistics
                statistical_report["numerical_features"][col] = {
                    "train_stats": {
                        "mean": float(train_values.mean()),
                        "std": float(train_values.std()),
                        "min": float(train_values.min()),
                        "max": float(train_values.max()),
                        "median": float(train_values.median()),
                        "skewness": float(train_values.skew()),
                        "kurtosis": float(train_values.kurtosis())
                    },
                    "val_stats": {
                        "mean": float(val_values.mean()),
                        "std": float(val_values.std()),
                        "min": float(val_values.min()),
                        "max": float(val_values.max()),
                        "median": float(val_values.median()),
                        "skewness": float(val_values.skew()),
                        "kurtosis": float(val_values.kurtosis())
                    }
                }
                
                # Kolmogorov-Smirnov test for distribution similarity
                try:
                    ks_statistic, ks_pvalue = stats.ks_2samp(train_values, val_values)
                    statistical_report["statistical_tests"][col] = {
                        "ks_test": {
                            "statistic": float(ks_statistic),
                            "p_value": float(ks_pvalue),
                            "distributions_similar": ks_pvalue > 0.05
                        }
                    }
                    
                    if ks_pvalue <= 0.05:
                        statistical_report["warnings"].append({
                            "column": col,
                            "issue": "Distribution mismatch between train and validation",
                            "ks_pvalue": float(ks_pvalue)
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not perform KS test for {col}: {str(e)}")
            
            logger.info("Statistical validation completed")
            
            return statistical_report
            
        except Exception as e:
            raise ProjectException(e, sys)
        
    def detect_data_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift between training and test datasets.

        Args:
            train_df (DataFrame): Training dataset.
            test_df  (DataFrame): Test/Validation dataset.

            Returns:
            Dict[str, Any]: Drift analysis report (JSON-serializable).
        """
        try:
            logger.info("Starting data drift detection")

            drift_report: Dict[str, Any] = {
                    "timestamp": datetime.now().isoformat(),
                    "drift_detected": False,
                    "feature_drift": {},
                    "drift_score": 0.0,
                    "drifted_features": [],
                    "drift_percentage": 0.0
                }

            drift_features = []


            # -----------------------------------------------
            # NUMERICAL FEATURES
            # -----------------------------------------------
            numerical_cols = train_df.select_dtypes(include=[np.number]).columns

            for col in numerical_cols:
                train_values = train_df[col].dropna()
                test_values = test_df[col].dropna()

                # Skip if either side empty
                if train_values.empty or test_values.empty:
                    drift_report["feature_drift"][col] = {
                        "type": "numerical",
                        "drift_detected": False,
                        "reason": "Insufficient data for comparison"
                    }
                    continue

                mean_diff = float(abs(train_values.mean() - test_values.mean()))
                std_diff = float(abs(train_values.std() - test_values.std()))

                ks_statistic, ks_pvalue = stats.ks_2samp(train_values, test_values)

                drift_detected = bool(ks_pvalue < 0.05)

                drift_report["feature_drift"][col] = {
                    "type": "numerical",
                    "drift_detected": drift_detected,
                    "mean_difference": mean_diff,
                    "std_difference": std_diff,
                    "ks_statistic": float(ks_statistic),
                    "ks_pvalue": float(ks_pvalue),
                }

                if drift_detected:
                    drift_features.append(col)

                # -----------------------------------------------
                # CATEGORICAL FEATURES
                # -----------------------------------------------
                categorical_cols = train_df.select_dtypes(include=["object", "category"]).columns

                for col in categorical_cols:

                    train_dist = train_df[col].value_counts(normalize=True)
                    test_dist = test_df[col].value_counts(normalize=True)

                    all_categories = set(train_dist.index) | set(test_dist.index)

                    dist_diff = float(sum(
                        abs(float(train_dist.get(cat, 0)) - float(test_dist.get(cat, 0)))
                        for cat in all_categories
                    ))

                    drift_detected = bool(dist_diff > 0.2)      # Configurable threshold

                    drift_report["feature_drift"][col] = {
                        "type": "categorical",
                        "drift_detected": drift_detected,
                        "distribution_difference": dist_diff,
                        "train_unique_values": int(train_df[col].nunique()),
                        "test_unique_values": int(test_df[col].nunique()),
                    }

                    if drift_detected:
                        drift_features.append(col)

                # -----------------------------------------------
                # FINAL METRICS
                # -----------------------------------------------
                drift_report["drift_detected"] = bool(len(drift_features) > 0)
                drift_report["drifted_features"] = drift_features

                total_cols = len(train_df.columns)
                drift_report["drift_score"] = float(round(len(drift_features) / total_cols, 3))
                drift_report["drift_percentage"] = float(round(
                    len(drift_features) / total_cols * 100, 2
                ))

                logger.info(
                    f"Data drift detection completed. Drift detected: {drift_report['drift_detected']}"
                )

                return drift_report

        except Exception as e:
                raise ProjectException(e, sys)

    
    def validate_time_series(self,train_df: pd.DataFrame,test_df: pd.DataFrame,date_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate time series specific properties if applicable.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            date_column: Name of the date/time column if exists
            
        Returns:
            Dictionary containing time series validation results
        """
        try:
            logger.info("Starting time series validation")
            
            ts_report = {
                "timestamp": datetime.now().isoformat(),
                "is_time_series": False,
                "validation_results": {}
            }
            
            # Auto-detect date column if not provided
            if date_column is None:
                for col in train_df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        date_column = col
                        break
            
            if date_column and date_column in train_df.columns:
                ts_report["is_time_series"] = True
                ts_report["date_column"] = date_column
                

                # Convert to datetime safely with explicit format
                datetime_format = "%d-%m-%Y %H:%M"

                train_df[date_column] = pd.to_datetime(
                    train_df[date_column],
                    format=datetime_format,
                    errors="coerce",
                    dayfirst=True
                )

                test_df[date_column] = pd.to_datetime(
                    test_df[date_column],
                    format=datetime_format,
                    errors="coerce",
                    dayfirst=True
                )
                
                # Check for temporal ordering
                train_sorted = train_df[date_column].is_monotonic_increasing
                val_sorted = test_df[date_column].is_monotonic_increasing
                
                ts_report["validation_results"] = {
                    "train_sorted": train_sorted,
                    "val_sorted": val_sorted,
                    "train_date_range": {
                        "start": str(train_df[date_column].min()),
                        "end": str(train_df[date_column].max())
                    },
                    "val_date_range": {
                        "start": str(test_df[date_column].min()),
                        "end": str(test_df[date_column].max())
                    },
                    "temporal_gap": str(test_df[date_column].min() - train_df[date_column].max()),
                    "train_missing_dates": int(train_df[date_column].isnull().sum()),
                    "val_missing_dates": int(test_df[date_column].isnull().sum())
                }
                
                # Check for temporal leakage (validation dates before training dates)
                if test_df[date_column].min() < train_df[date_column].max():
                    ts_report["validation_results"]["temporal_leakage_warning"] = True
                else:
                    ts_report["validation_results"]["temporal_leakage_warning"] = False
            
            logger.info("Time series validation completed")
            
            return ts_report
            
        except Exception as e:
            raise ProjectException(e, sys)
        
    def _convert_to_serializable(self, obj):
        """
        Recursively convert NumPy, Pandas, and unsupported data types 
        into JSON-serializable Python native types.
        """

        # Handle numpy boolean
        if isinstance(obj, (np.bool_,)):
            return bool(obj)

        # Handle numpy integers
        if isinstance(obj, (np.int32, np.int64, np.uint32, np.uint64)):
            return int(obj)

        # Handle numpy floats
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)

        # Handle pandas Series / Index
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()

        # Handle pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")

        # Handle dictionaries (recursive)
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}

        # Handle lists / tuples (recursive)
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(i) for i in obj]

        # Return all other types unchanged
        return obj

    def _save_report(self, report: Dict[str, Any], file_path: str) -> None:
        """
        Save validation report to JSON file after converting all values
        into JSON-serializable format.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            serializable_report = self._convert_to_serializable(report)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_report, f, indent=4, ensure_ascii=False)

            logger.info(f"Report saved successfully -> {file_path}")

        except Exception as e:
            raise ProjectException(
                f"Failed to save validation report at {file_path}: {e}",
                sys
            )
                       
    def initiate_data_validation(self) -> ValidationArtifact:
        """
        Execute all data validation steps and generate reports.
        
        Returns:
            ValidationArtifact containing paths to all validation reports
        """
        try:
            logger.info("=" * 70)
            logger.info("DATA VALIDATION STARTED")
            logger.info("=" * 70)
            
            # Load data
            train_df, test_df = self._load_data()
            
            # 1. Schema validation
            logger.info("Performing schema validation...")
            schema_report = self.validate_schema_structure(train_df, test_df)
            self._save_report(schema_report, self.data_validation_config.schema_validation_path)
            
            # 2. Data quality validation
            logger.info("Performing data quality validation...")
            quality_report = self.validate_data_quality(train_df, test_df)
            self._save_report(quality_report, self.data_validation_config.quality_report_path)
            
            # 3. Statistical validation
            logger.info("Performing statistical validation...")
            statistical_report = self.validate_statistical_properties(train_df, test_df)
            self._save_report(statistical_report, self.data_validation_config.statistical_validation_path)
            
            # 4. Data drift detection
            logger.info("Performing drift detection...")
            drift_report = self.detect_data_drift(train_df, test_df)
            self._save_report(drift_report, self.data_validation_config.drift_report_path)
            
            # 5. Time series validation
            logger.info("Performing time series validation...")
            ts_report = self.validate_time_series(train_df, test_df)
            self._save_report(ts_report, self.data_validation_config.time_series_validation_path)
            
            # Generate overall validation status
            validation_status = {
                #"timestamp": datetime.now().isoformat(),
                "validation_passed": True,
                "summary": {
                    "schema_match": schema_report["schema_match"],
                    "quality_status": quality_report["overall_status"],
                    "drift_detected": drift_report["drift_detected"],
                    "drift_score": drift_report["drift_score"]
                },
                "recommendations": []
            }
            
            # Add recommendations based on findings
            if not schema_report["schema_match"]:
                validation_status["validation_passed"] = False
                validation_status["recommendations"].append(
                    "Schema mismatch detected. Review schema validation report."
                )
            
            if quality_report["overall_status"] == "FAILED":
                validation_status["validation_passed"] = False
                validation_status["recommendations"].append(
                    "Data quality issues detected. Review quality report."
                )
            
            if drift_report["drift_detected"]:
                validation_status["recommendations"].append(
                    f"Data drift detected in {len(drift_report['drifted_features'])} features. "
                    "Consider retraining or investigating data sources."
                )
            
            self._save_report(validation_status, self.data_validation_config.validation_status_path)
            
            # Create validation artifact
            validation_artifact = ValidationArtifact(
                drift_report_file=self.data_validation_config.drift_report_path,
                validation_status_file=self.data_validation_config.validation_status_path,
                quality_report_file=self.data_validation_config.quality_report_path,
                schema_validation_file=self.data_validation_config.schema_validation_path,
                statistical_validation_file=self.data_validation_config.statistical_validation_path,
                time_series_validation_file=self.data_validation_config.time_series_validation_path
            )
            
            logger.info("=" * 70)
            logger.info("DATA VALIDATION COMPLETED SUCCESSFULLY")
            logger.info(f"Validation Status: {'PASSED' if validation_status['validation_passed'] else 'FAILED'}")
            logger.info("=" * 70)
            
            return validation_artifact
            
        except Exception as e:
            raise ProjectException(e, sys)