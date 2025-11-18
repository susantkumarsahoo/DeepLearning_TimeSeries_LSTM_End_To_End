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
    f"{project_name}/config/model_architecture.yaml",  # Central logging config

    # Integrations for external services
    f"{project_name}/integrations/__init__.py",
    f"{project_name}/integrations/aws_integration.py",
    f"{project_name}/integrations/llms_integration.py"
    f"{project_name}/integrations/database_integration.py",

    # Constants
    f"{project_name}/constants/__init__.py",
    f"{project_name}/constants/paths.py",

    # Data access
    f"{project_name}/data_loaders/__init__.py",
    f"{project_name}/data_loaders/data_loader.py",
  
    # Entities
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/components_entity.py",

    # Project-specific components
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/processors_chunker.py",


    # Models
    f"{project_name}/llm_models/__init__.py",
    f"{project_name}/llm_models/llm_model.py",

    f"{project_name}/agents/__init__.py",
    f"{project_name}/agents/agent_registry.py",



    f"{project_name}/services/__init__.py",
    f"{project_name}/services/llm_service.py",
    f"{project_name}/services/embedding_service.py",
    f"{project_name}/services/retriever_service.py",


    f"{project_name}/langchain/__init__.py",
    f"{project_name}/langchain/llms_models.py",
    f"{project_name}/langchain/prompt.py",
    f"{project_name}/langchain/output_parsers.py",
    f"{project_name}/langchain/memory.py",
    f"{project_name}/langchain/chains.py",
    f"{project_name}/langchain/messages.py",
    f"{project_name}/langchain/agents.py",
    f"{project_name}/langchain/retrievers_rag.py",
    f"{project_name}/langchain/embedding.py",
    f"{project_name}/langchain/retrievers.py",
    f"{project_name}/langchain/runnables.py",       
    f"{project_name}/langchain/callbacks.py",
    f"{project_name}/langchain/schema.py",
    f"{project_name}/langchain/text_splitter.py",
    f"{project_name}/langchain/tools.py",
    f"{project_name}/langchain/document_loader.py",
    f"{project_name}/langchain/streaming.py",
    f"{project_name}/langchain/vector_stores.py",
    f"{project_name}/langchain/MCP.py",
    f"{project_name}/langchain/middleware.py",    



    f"{project_name}/langgraph /__init__.py",
    f"{project_name}/langgraph/state_graph.py",
    f"{project_name}/langgraph/graph.py",
    f"{project_name}/langgraph/state.py",
    f"{project_name}/langgraph/node.py",
    f"{project_name}/langgraph/edge.py",
    f"{project_name}/langgraph/control_flow.py",
    f"{project_name}/langgraph/conditional_flow.py",
    f"{project_name}/langgraph/retrievers.py",
    f"{project_name}/langgraph/checkpointer.py",
    f"{project_name}/langgraph/multi_agent.py",  

    # # RDB / vectorstore db
    f"{project_name}/stores/__init__.py",
    f"{project_name}/stores/vectorstore_db.py",
    f"{project_name}/stores/sqlite_db.py",

    # Pipelines
    f"{project_name}/pipelines/__init__.py",
    f"{project_name}/pipelines/training_pipeline.py",
    f"{project_name}/pipelines/inference_pipeline.py",
    f"{project_name}/pipelines/rag_pipeline.py",

    # Logging
    f"{project_name}/logging/__init__.py",
    f"{project_name}/logging/logger.py",

    # Exceptions
    f"{project_name}/exceptions/__init__.py",
    f"{project_name}/exceptions/exception.py",

    # Utilities
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/handler.py",
    f"{project_name}/utils/helpers.py",

    # Cloud
    f"{project_name}/cloud/__init__.py",
    f"{project_name}/cloud/aws_storage.py",

    # API
    f"{project_name}/api/__init__.py",
    f"{project_name}/api/routes.py",
    f"{project_name}/api/endpoints.py",

    # evaluation / validation
    f"{project_name}/eval/__init__.py",
    f"{project_name}/eval/evaluator.py",

    # Monitoring
    f"{project_name}/monitoring/__init__.py",
    f"{project_name}/monitoring/langSmith_observability.py",


    # Outside project_name
    f"{cicd_folder}/'pipeline.yaml'",
    f"{configs_folder}/llms_configs.yaml",
    f"{configs_folder}/project_configs.yaml",
    f"{data_folder}/raw/.gitkeep",
    f"{notebooks_folder}/README.md",  # Explains notebooks workflow
    f"{notebooks_folder}/01_note.ipynb",
    f"{notebooks_folder}/02_note.ipynb",
    f"{templates_folder}/project.html",
    f"{static_css_folder}/style.css",

    # Scripts
    f"{scripts_folder}/automation.sh",
    f"{scripts_folder}/run_pipeline.sh",
    f"{scripts_folder}/run_pipeline.bat",  # Windows


    # Tests
    f"{tests_folder}/__init__.py",
    f"{tests_folder}/conftest.py",  # For pytest fixtures
    f"{tests_folder}/test.py",


    # Docs
    f"{docs_folder}/architecture.md",

    # Logs folder
    f"{logs_folder}/.gitkeep",

    # Root-level files
    "requireme  nts.txt",
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