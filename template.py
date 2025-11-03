import os
from pathlib import Path

# -------------------------
# Define project name (main package)
# -------------------------
project_name = "src"  # More descriptive than "src"

# -------------------------
# Define additional folders
# -------------------------
cicd_folder       = "Github"
configs_folder    = "configs"
data_folder       = "data"
notebooks_folder  = "notebooks"
static_css_folder = "static/css"
templates_folder  = "templates"
tests_folder      = "tests"
scripts_folder    = "scripts"
docs_folder       = "docs"
logs_folder       = "logs"  # For runtime logs

# -------------------------
# List of files & folders to create
# -------------------------
list_of_files = [
    # Main package
    f"{project_name}/__init__.py",

    # Config
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/config_reader.py",
    f"{project_name}/config/hyperparams.yaml",
    f"{project_name}/config/model_architecture.yaml",  # Central logging config

    # Integrations for external services
    f"{project_name}/integrations/__init__.py",
    f"{project_name}/integrations/aws_integration.py",
    f"{project_name}/integrations/database_integration.py",

    # Constants
    f"{project_name}/constants/__init__.py",
    f"{project_name}/constants/paths.py",

    # Data access
    f"{project_name}/data_access/__init__.py",
    f"{project_name}/data_access/data_loader.py",
    f"{project_name}/data_access/data_augmentation.py",

    # Entities
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/components_config_entity.py",
    f"{project_name}/entity/model_config_entity.py",
    f"{project_name}/entity/pipelines_config_entity.py",

    # Project-specific components
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_preprocessing.py",
    f"{project_name}/components/feature_engineering.py",
    f"{project_name}/components/feature_transformer.py",

    # Models
    f"{project_name}/models/__init__.py",
    f"{project_name}/models/trainer.py",
    f"{project_name}/models/predictor.py",
    f"{project_name}/models/evaluator.py",
    f"{project_name}/models/model_registry.py",
    f"{project_name}/models/deployer.py",

    # Pipelines
    f"{project_name}/pipelines/__init__.py",
    f"{project_name}/pipelines/training_pipeline.py",
    f"{project_name}/pipelines/inference_pipeline.py",
    f"{project_name}/pipelines/prediction_pipeline.py",
    f"{project_name}/pipelines/deployment_pipeline.py",

    # Logging
    f"{project_name}/logging/__init__.py",
    f"{project_name}/logging/logger.py",

    # Exceptions
    f"{project_name}/exceptions/__init__.py",
    f"{project_name}/exceptions/exception.py",

    # Utilities
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/handler.py",
    f"{project_name}/utils/metrics.py",
    f"{project_name}/utils/helpers.py",

    # Visualization
    f"{project_name}/visualization/__init__.py",
    f"{project_name}/visualization/plots.py",
    f"{project_name}/visualization/dashboard.py",

    # Cloud
    f"{project_name}/cloud/__init__.py",
    f"{project_name}/cloud/aws_storage.py",

    # API
    f"{project_name}/api/__init__.py",
    f"{project_name}/api/routes.py",

    # Monitoring
    f"{project_name}/monitoring/__init__.py",
    f"{project_name}/monitoring/drift_detector.py",
    f"{project_name}/monitoring/data_drift.py",
    f"{project_name}/monitoring/model_drift.py",

    # Outside project_name
    f"{cicd_folder}/'pipeline.yaml'",
    f"{configs_folder}/project_configs.yaml",
    f"{configs_folder}/model_configs.yaml",
    f"{data_folder}/raw/.gitkeep",
    f"{notebooks_folder}/README.md",  # Explains notebooks workflow
    f"{notebooks_folder}/01_note.ipynb",
    f"{notebooks_folder}/02_note.ipynb",
    f"{templates_folder}/project.html",
    f"{static_css_folder}/style.css",

    # Scripts
    f"{scripts_folder}/data_automation.py",
    f"{scripts_folder}/run_pipeline.sh",
    f"{scripts_folder}/run_pipeline.bat",  # Windows
    f"{scripts_folder}/db_migrations.py",

    # Tests
    f"{tests_folder}/__init__.py",
    f"{tests_folder}/conftest.py",  # For pytest fixtures
    f"{tests_folder}/test_models.py",
    f"{tests_folder}/test_api.py",
    f"{tests_folder}/test_data_pipeline.py",

    # Docs
    f"{docs_folder}/data_dictionary.md",
    f"{docs_folder}/architecture.md",
    f"{docs_folder}/pipeline_flow.md",

    # Logs folder
    f"{logs_folder}/.gitkeep",

    # Root-level files
    "requirements.txt",
    "requirements-dev.txt",  # For dev dependencies
#   "README.md",
    ".env",
    "setup.py",
#   ".gitignore",
    ".dockerignore",
    "Dockerfile",
    "docker-compose.yml",
    "main.py",
    "demo.py",
    "app.py"
]

# -------------------------
# Create files and directories
# -------------------------
for filepath in list_of_files:
    file_path = Path(filepath)
    dir_path = file_path.parent
    os.makedirs(dir_path, exist_ok=True)
    if not file_path.exists():
        file_path.touch()
        print(f"Created: {file_path}")
    else:
        print(f"Already exists: {file_path}")