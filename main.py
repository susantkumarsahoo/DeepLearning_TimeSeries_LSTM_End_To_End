from src.pipelines.model_prediction_pipeline import run_prediction_pipeline
from src.entity.artifact_entity import DeploymentArtifact
from src.entity.model_config_entity import ModelDeploymentConfig
from datetime import datetime, timedelta
import os


def main():
    """
    Main function to run the prediction pipeline.
    """

    print("\n" + "=" * 80)
    print("PREDICTION PIPELINE EXECUTION")
    print("=" * 80)

    # Initialize deployment config
    deployment_config = ModelDeploymentConfig()

    # Create deployment artifact with paths to deployed model and scaler
    deployment_artifact = DeploymentArtifact(
        deployed_model_file=deployment_config.deployed_model_path,
        deployed_preprocessor_file=deployment_config.deployed_preprocessor_path,
        deployment_report_file=deployment_config.deployment_report_path
    )

    # Verify that model and scaler files exist
    if not os.path.exists(deployment_artifact.deployed_model_file):
        raise FileNotFoundError(
            f"Deployed model not found at: {deployment_artifact.deployed_model_file}"
        )

    if not os.path.exists(deployment_artifact.deployed_preprocessor_file):
        raise FileNotFoundError(
            f"Deployed preprocessor not found at: {deployment_artifact.deployed_preprocessor_file}"
        )

    # Define prediction date range
    start_date = datetime.now()
    end_date = start_date + timedelta(days=60)

    # Run prediction pipeline
    predictions_artifact = run_prediction_pipeline(
        start_date=start_date,
        end_date=end_date,
        deployment_artifact=deployment_artifact
    )

    return predictions_artifact


if __name__ == "__main__":
    main()

# python main.py