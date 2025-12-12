from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path

# Import your project modules
from src.pipelines.model_prediction_pipeline import run_prediction_pipeline
from src.entity.artifact_entity import DeploymentArtifact
from src.entity.model_config_entity import ModelDeploymentConfig
from src.logging.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Time Series Forecasting API",
    description="API for generating time series predictions using LSTM model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    
    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2024-01-01",
                "end_date": "2024-03-01"
            }
        }

class PredictionResponse(BaseModel):
    success: bool
    message: str
    total_predictions: int
    date_range: dict
    statistics: dict
    predictions: List[dict]
    file_path: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_status: str
    preprocessor_status: str

class ModelInfo(BaseModel):
    model_path: str
    preprocessor_path: str
    model_exists: bool
    preprocessor_exists: bool


# Global variables to cache deployment artifacts
deployment_artifact = None
deployment_config = None

def initialize_deployment():
    """Initialize deployment configuration and artifacts"""
    global deployment_artifact, deployment_config
    
    try:
        logger.info("Initializing deployment configuration...")
        
        # Initialize deployment config
        deployment_config = ModelDeploymentConfig()
        
        # Create deployment artifact
        deployment_artifact = DeploymentArtifact(
            deployed_model_file=deployment_config.deployed_model_path,
            deployed_preprocessor_file=deployment_config.deployed_preprocessor_path,
            deployment_report_file=deployment_config.deployment_report_path
        )
        
        # Verify files exist
        if not os.path.exists(deployment_artifact.deployed_model_file):
            logger.error(f"Model file not found: {deployment_artifact.deployed_model_file}")
            raise FileNotFoundError(f"Model file not found")
        
        if not os.path.exists(deployment_artifact.deployed_preprocessor_file):
            logger.error(f"Preprocessor file not found: {deployment_artifact.deployed_preprocessor_file}")
            raise FileNotFoundError(f"Preprocessor file not found")
        
        logger.info("Deployment configuration initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing deployment: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("=" * 80)
    logger.info("STARTING FASTAPI APPLICATION")
    logger.info("=" * 80)
    
    success = initialize_deployment()
    
    if success:
        logger.info("Application started successfully")
    else:
        logger.warning("Application started with warnings - model files may be missing")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Time Series Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "download_predictions": "/download/{filename}",
            "predictions_history": "/predictions/history"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model_exists = os.path.exists(deployment_artifact.deployed_model_file) if deployment_artifact else False
        preprocessor_exists = os.path.exists(deployment_artifact.deployed_preprocessor_file) if deployment_artifact else False
        
        return HealthResponse(
            status="healthy" if (model_exists and preprocessor_exists) else "unhealthy",
            timestamp=datetime.now().isoformat(),
            model_status="loaded" if model_exists else "not_found",
            preprocessor_status="loaded" if preprocessor_exists else "not_found"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about deployed model"""
    try:
        if not deployment_artifact:
            raise HTTPException(status_code=500, detail="Deployment not initialized")
        
        return ModelInfo(
            model_path=deployment_artifact.deployed_model_file,
            preprocessor_path=deployment_artifact.deployed_preprocessor_file,
            model_exists=os.path.exists(deployment_artifact.deployed_model_file),
            preprocessor_exists=os.path.exists(deployment_artifact.deployed_preprocessor_file)
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def generate_predictions(request: PredictionRequest):
    """Generate predictions for the specified date range"""
    try:
        logger.info("=" * 80)
        logger.info("PREDICTION REQUEST RECEIVED")
        logger.info("=" * 80)
        logger.info(f"Start Date: {request.start_date}")
        logger.info(f"End Date: {request.end_date}")
        
        # Validate deployment
        if not deployment_artifact:
            raise HTTPException(
                status_code=500, 
                detail="Deployment not initialized. Please restart the server."
            )
        
        # Parse and validate dates
        try:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}"
            )
        
        # Validate date range
        if start_date >= end_date:
            raise HTTPException(
                status_code=400,
                detail="start_date must be before end_date"
            )
        
        # Check if date range is too large (optional safety check)
        days_diff = (end_date - start_date).days
        if days_diff > 365:
            raise HTTPException(
                status_code=400,
                detail=f"Date range too large ({days_diff} days). Maximum allowed is 365 days."
            )
        
        # Run prediction pipeline
        logger.info("Running prediction pipeline...")
        predictions_artifact = run_prediction_pipeline(
            start_date=start_date,
            end_date=end_date,
            deployment_artifact=deployment_artifact
        )
        
        # Load predictions
        if not os.path.exists(predictions_artifact.predictor_file):
            raise HTTPException(
                status_code=500,
                detail="Predictions file was not created"
            )
        
        predictions_df = pd.read_csv(predictions_artifact.predictor_file)
        
        # Calculate statistics
        stats = {
            "mean": float(predictions_df['predicted_value'].mean()),
            "median": float(predictions_df['predicted_value'].median()),
            "std": float(predictions_df['predicted_value'].std()),
            "min": float(predictions_df['predicted_value'].min()),
            "max": float(predictions_df['predicted_value'].max()),
            "q25": float(predictions_df['predicted_value'].quantile(0.25)),
            "q75": float(predictions_df['predicted_value'].quantile(0.75))
        }
        
        # Prepare response
        predictions_list = predictions_df.to_dict('records')
        
        # Convert datetime to string for JSON serialization
        for pred in predictions_list:
            if 'ds' in pred:
                pred['ds'] = str(pred['ds'])
            if 'prediction_date' in pred:
                pred['prediction_date'] = str(pred['prediction_date'])
        
        logger.info(f"Successfully generated {len(predictions_df)} predictions")
        logger.info("=" * 80)
        
        return PredictionResponse(
            success=True,
            message="Predictions generated successfully",
            total_predictions=len(predictions_df),
            date_range={
                "start": str(predictions_df['ds'].min()),
                "end": str(predictions_df['ds'].max())
            },
            statistics=stats,
            predictions=predictions_list,
            file_path=predictions_artifact.predictor_file
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_predictions(filename: str):
    """Download predictions CSV file"""
    try:
        # Security: only allow downloading from predictions directory
        predictions_dir = Path("artifacts/predictions")
        file_path = predictions_dir / filename
        
        # Prevent directory traversal
        if not str(file_path.resolve()).startswith(str(predictions_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            media_type="text/csv",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/history")
async def get_prediction_history():
    """Get list of all prediction files"""
    try:
        predictions_dir = Path("artifacts/predictions")
        
        if not predictions_dir.exists():
            return {"files": [], "message": "No predictions directory found"}
        
        files = []
        for file_path in predictions_dir.glob("*.csv"):
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        # Sort by modified time, newest first
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return {
            "total_files": len(files),
            "files": files
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI application
    # FIXED: Changed from 172.0.0.1 to 0.0.0.0 (correct host)
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# To run this application:
# uvicorn app:app --reload --host 0.0.0.0 --port 8000