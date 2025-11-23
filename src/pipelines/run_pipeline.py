from src.pipelines.data_ingestion_pipeline import run_ingestion_pipeline
from src.pipelines.data_validation_pipeline import run_validation_pipeline
from src.pipelines.data_preprocessing_pipeline import run_preprocessing_pipeline
from src.constants.paths import dataset_path


def run_full_pipeline(dataset_path: str):
    ingestion_artifact = run_ingestion_pipeline(dataset_path)
    validation_artifact = run_validation_pipeline(ingestion_artifact)
    preprocessing_artifact = run_preprocessing_pipeline(ingestion_artifact, validation_artifact)
    return preprocessing_artifact


if __name__ == "__main__":
    dataset_path = dataset_path
    output = run_full_pipeline(dataset_path)
    print(output)


# python src/pipelines/run_pipeline.py
