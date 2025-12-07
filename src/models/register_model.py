# register model

import json
import mlflow
import mlflow.sklearn
from pathlib import Path
import dagshub
import os

# *********
# dagshub.init(repo_owner='Mohdmuzaffar307', repo_name='Loan-Approval-end-to-end', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/Mohdmuzaffar307/Loan-Approval-end-to-end.mlflow")
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Mohdmuzaffar307"
repo_name = "Loan-Approval-end-to-end"

# ********


# mlflow.set_tracking_uri("http://127.0.0.1:5000/")


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    return model_info
    

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
   
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    print("Model URI:", model_uri)
        
    # Register model
    model_version = mlflow.register_model(model_uri, model_name)
    print("Registered model version:", model_version)
        
    # Transition the model to "Staging" stage
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )
    print("Model moved to Staging.")


def main():
    model_info_path = Path('reports/experiment_info.json')
    model_info = load_model_info(model_info_path)
    
    model_name = "my_model"
    register_model(model_name, model_info)
   

if __name__ == '__main__':
    main()
