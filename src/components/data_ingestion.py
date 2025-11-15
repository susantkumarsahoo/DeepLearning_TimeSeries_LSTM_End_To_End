"""
Data Ingestion Component
File: src/components/data_ingestion.py
"""

import os
import sys
import shutil
from typing import Optional
from src.logging.logger import get_logger
from src.exceptions.exception import ProjectException
from src.entity.components_config_entity import DataIngestionConfig
from src.entity.artifact_entity import IngestionArtifact
from src.utils.helpers import (
    validate_dataset_path,
    load_csv_data,
    save_dataframe,
    generate_dataset_metadata,
    generate_dataset_schema,
    save_json,
    split_train_test
)

logger = get_logger(__name__)


class DataIngestion:
    """
    Data Ingestion Component for loading, validating, and splitting data.
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize Data Ingestion component.
        
        Args:
            config: DataIngestionConfig object containing all configurations
        """
        try:
            self.config = config
            logger.info("="*70)
            logger.info("DATA INGESTION INITIATED")
            logger.info("="*70)
            logger.info(f"Dataset Path: {self.config.dataset_path}")
            logger.info(f"Test Size: {self.config.test_size}")
            logger.info(f"Random State: {self.config.random_state}")
            
        except Exception as e:
            raise ProjectException(e, sys)
    
    def validate_dataset(self) -> None:
        """
        Validate the input dataset path and accessibility.
        
        Raises:
            DataIngestionException: If validation fails
        """
        try:
            logger.info("Validating dataset path...")
            validate_dataset_path(self.config.dataset_path)
            logger.info("Dataset validation completed successfully")
            
        except Exception as e:
            raise ProjectException(e, sys)
    
    def load_raw_data(self) -> None:
        """
        Load raw data and save to raw data directory.
        
        Raises:
            DataIngestionException: If loading or saving fails
        """
        try:
            logger.info("Loading raw data...")
            
            # Load dataset
            self.raw_df = load_csv_data(self.config.dataset_path)
            
            # Save raw data to artifacts
            save_dataframe(self.raw_df, self.config.raw_data_path)
            
            logger.info(f"Raw data loaded successfully with shape: {self.raw_df.shape}")
            
        except Exception as e:
            raise ProjectException(e, sys)
    
    def process_data(self) -> None:
        """
        Process the raw data (basic cleaning and validation).
        
        Raises:
            DataIngestionException: If processing fails
        """
        try:
            logger.info("Processing data...")
            
            # Create a copy for processing
            self.processed_df = self.raw_df.copy()
            
            # Basic processing steps
            initial_rows = len(self.processed_df)
            
            # Remove duplicate rows
            self.processed_df.drop_duplicates(inplace=True)
            duplicates_removed = initial_rows - len(self.processed_df)
            
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate rows")
            
            # Reset index
            self.processed_df.reset_index(drop=True, inplace=True)
            
            # Save processed data
            save_dataframe(self.processed_df, self.config.processed_data_path)
            
            logger.info(f"Data processing completed. Final shape: {self.processed_df.shape}")
            
        except Exception as e:
            raise ProjectException(e, sys)
    
    def split_data(self) -> None:
        """
        Split processed data into train and test sets.
        
        Raises:
            DataIngestionException: If splitting fails
        """
        try:
            logger.info("Splitting data into train and test sets...")
            
            # Split data
            self.train_df, self.test_df = split_train_test(
                self.processed_df,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            # Save train and test data
            save_dataframe(self.train_df, self.config.train_data_path)
            save_dataframe(self.test_df, self.config.test_data_path)
            
            logger.info(f"Train set shape: {self.train_df.shape}")
            logger.info(f"Test set shape: {self.test_df.shape}")
            
        except Exception as e:
            raise ProjectException(e, sys)
    
    def generate_metadata(self) -> None:
        """
        Generate and save metadata for the dataset.
        
        Raises:
            DataIngestionException: If metadata generation fails
        """
        try:
            logger.info("Generating dataset metadata...")
            
            # Generate metadata
            metadata = generate_dataset_metadata(self.processed_df)
            
            # Add additional information
            metadata.update({
                "original_dataset_path": self.config.dataset_path,
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
                "train_set_size": len(self.train_df),
                "test_set_size": len(self.test_df)
            })
            
            # Save metadata
            save_json(metadata, self.config.metadata_path)
            
            logger.info("Metadata generated and saved successfully")
            
        except Exception as e:
            raise ProjectException(e, sys)
    
    def generate_schema(self) -> None:
        """
        Generate and save schema for the dataset.
        
        Raises:
            DataIngestionException: If schema generation fails
        """
        try:
            logger.info("Generating dataset schema...")
            
            # Generate schema
            schema = generate_dataset_schema(self.processed_df)
            
            # Save schema
            save_json(schema, self.config.schema_path)
            
            logger.info("Schema generated and saved successfully")
            
        except Exception as e:
            raise ProjectException(e, sys)
    
    def initiate_data_ingestion(self) -> IngestionArtifact:
        """
        Execute the complete data ingestion pipeline.
        
        Returns:
            IngestionArtifact containing paths to all generated artifacts
            
        Raises:
            DataIngestionException: If any step in the pipeline fails
        """
        try:
            logger.info("Starting data ingestion pipeline...")
            
            # Step 1: Validate dataset
            self.validate_dataset()
            
            # Step 2: Load raw data
            self.load_raw_data()
            
            # Step 3: Process data
            self.process_data()
            
            # Step 4: Split data
            self.split_data()
            
            # Step 5: Generate metadata
            self.generate_metadata()
            
            # Step 6: Generate schema
            self.generate_schema()
            
            # Create artifact object
            ingestion_artifact = IngestionArtifact(
                raw_data_file=self.config.raw_data_path,
                processed_data_file=self.config.processed_data_path,
                train_file=self.config.train_data_path,
                test_file=self.config.test_data_path,
                metadata_file=self.config.metadata_path,
                schema_file=self.config.schema_path
            )
            
            logger.info("="*70)
            logger.info("DATA INGESTION COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"Raw data file: {ingestion_artifact.raw_data_file}")
            logger.info(f"Processed data file: {ingestion_artifact.processed_data_file}")
            logger.info(f"Train file: {ingestion_artifact.train_file}")
            logger.info(f"Test file: {ingestion_artifact.test_file}")
            logger.info(f"Metadata file: {ingestion_artifact.metadata_file}")
            logger.info(f"Schema file: {ingestion_artifact.schema_file}")
            logger.info("="*70)
            
            return ingestion_artifact
            
        except Exception as e:
            logger.error("Data ingestion failed")
            raise ProjectException(e, sys)