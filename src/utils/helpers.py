"""
Utility Functions for Data Ingestion
File: src/utils/data_utils.py
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Tuple
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
import sys

logger = get_logger(__name__)


def validate_dataset_path(dataset_path: str) -> None:
    """
    Validate if the dataset path exists and is accessible.
    
    Args:
        dataset_path: Path to the dataset file
        
    Raises:
        DataIngestionException: If path doesn't exist or is not accessible
    """
    try:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at path: {dataset_path}")
        
        if not os.path.isfile(dataset_path):
            raise ValueError(f"Path is not a file: {dataset_path}")
        
        logger.info(f"Dataset path validated: {dataset_path}")
        
    except Exception as e:
        raise ProjectException(e, sys)


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        DataIngestionException: If loading fails
    """
    try:
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        raise ProjectException(e, sys)


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Destination file path
        
    Raises:
        DataIngestionException: If saving fails
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df.to_csv(file_path, index=False)
        logger.info(f"DataFrame saved to: {file_path}")
        
    except Exception as e:
        raise ProjectException(e, sys)


def generate_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate metadata information for the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing metadata
    """
    try:
        metadata = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
        }
        
        logger.info("Dataset metadata generated successfully")
        return metadata
        
    except Exception as e:
        raise ProjectException(e, sys)


def generate_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate schema information for the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing schema information
    """
    try:
        schema = {}
        
        for column in df.columns:
            col_info = {
                "dtype": str(df[column].dtype),
                "unique_values": int(df[column].nunique()),
                "null_count": int(df[column].isnull().sum()),
                "null_percentage": round(df[column].isnull().sum() / len(df) * 100, 2)
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[column]):
                col_info.update({
                    "min": float(df[column].min()) if not df[column].isnull().all() else None,
                    "max": float(df[column].max()) if not df[column].isnull().all() else None,
                    "mean": float(df[column].mean()) if not df[column].isnull().all() else None,
                    "median": float(df[column].median()) if not df[column].isnull().all() else None,
                    "std": float(df[column].std()) if not df[column].isnull().all() else None
                })
            
            # Add sample values for categorical columns
            elif pd.api.types.is_object_dtype(df[column]):
                col_info["sample_values"] = df[column].dropna().unique()[:5].tolist()
            
            schema[column] = col_info
        
        logger.info("Dataset schema generated successfully")
        return schema
        
    except Exception as e:
        raise ProjectException(e, sys)
    

def save_json_new(data: Any, file_path: str) -> None:
    """
    Universal JSON-safe serializer and saver.
    Converts all non-serializable objects (Timestamp, numpy types,
    datetime, UUID, sets, tuples, custom classes) into JSON-safe formats.
    
    Guarantees zero serialization errors.
    """

    import json
    import os
    import numpy as np
    import pandas as pd
    import datetime
    from uuid import UUID

    try:
        def to_serializable(obj):
            """
            Recursively convert non-serializable objects into JSON-safe formats.
            """

            # --- Basic Python types ---
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj

            # --- Pandas Timestamp / datetime ---
            if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
                return obj.isoformat()

            # --- Numpy types ---
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()

            # --- UUID ---
            if isinstance(obj, UUID):
                return str(obj)

            # --- Dict (ensure keys become str) ---
            if isinstance(obj, dict):
                return {str(k): to_serializable(v) for k, v in obj.items()}

            # --- List / Tuple / Set ---
            if isinstance(obj, (list, tuple, set)):
                return [to_serializable(i) for i in obj]

            # --- Pandas Series ---
            if isinstance(obj, pd.Series):
                return obj.apply(to_serializable).to_dict()

            # --- Pandas DataFrame ---
            if isinstance(obj, pd.DataFrame):
                return obj.applymap(to_serializable).to_dict(orient="records")

            # --- Any other object ---
            return str(obj)

        # Convert all data
        safe_data = to_serializable(data)

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save JSON
        with open(file_path, "w") as f:
            json.dump(safe_data, f, indent=4)

        logger.info(f"JSON saved successfully: {file_path}")

    except Exception as e:
        raise ProjectException(e, sys)

    


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary data to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Destination file path
        
    Raises:
        DataIngestionException: If saving fails
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        logger.info(f"JSON data saved to: {file_path}")
        
    except Exception as e:
        raise ProjectException(e, sys)
    

import json
import os
from typing import Any, Dict
from datetime import datetime

class SaveJsonException(Exception):
    """Custom exception for JSON saving errors."""
    pass


def save_json_data(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary data to a JSON file with full reliability.
    
    Key Capabilities:
    -----------------
    • Creates directories automatically  
    • Supports UTF-8 encoding  
    • Atomic write operation (prevents corruption)  
    • Graceful error handling  
    • Works for all valid JSON-serializable objects  
    """

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Temporary file for atomic operation
        temp_file = f"{file_path}.tmp_{datetime.now().timestamp()}"

        # Write JSON safely
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # Replace old file atomically
        os.replace(temp_file, file_path)

    except Exception as e:
        raise SaveJsonException(
            f"Failed to save JSON file at: {file_path}. Error: {str(e)}"
        ) from e


    

def split_train_test(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        DataIngestionException: If splitting fails
    """
    try:
        from sklearn.model_selection import train_test_split
        
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Data split - Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        return train_df, test_df
        
    except Exception as e:
        raise ProjectException(e, sys)