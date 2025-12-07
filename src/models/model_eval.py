import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
import dagshub
import os
import json
from pathlib import Path


# dagshub.init(
#     repo_owner='Mohdmuzaffar307',
#     repo_name='Loan-Approval-end-to-end',
#     mlflow=True
# )
# mlflow.set_tracking_uri("https://dagshub.com/Mohdmuzaffar307/Loan-Approval-end-to-end.mlflow")



import os

dagshub_username = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_username or not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_USERNAME or DAGSHUB_TOKEN environment variable is not set"
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


dagshub_url = "https://dagshub.com"
repo_owner = "Mohdmuzaffar307"
repo_name = "Loan-Approval-end-to-end"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")



# Increase timeout for all httpx requests (global patch)



def load_model(model_path :Path):
    with open(model_path, 'rb') as file:
            model = joblib.load(file)
            
    return model


def predict_model(model,x_test_url:str,y_test_url:str):
    x_test=pd.read_csv(x_test_url)
    y_test=pd.read_csv(y_test_url)

    y_pred=model.predict(x_test)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    
    metrics_dict={"accurecy":acc,
                  "recall":rec,
                  "f1-score":f1,
                  "precesion":pre
                  }
    
    return metrics_dict

def save_metrics(metrics:dict,file_path:str):
    # os.makedirs(os.path.dirname(file_path),exist_ok=True)
    file_path.parent.mkdir(parents=True,exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
        print('metrics saved')
        

def save_model_info(run_id: str, model_path: str, file_path: Path) -> None:
    """Save the model run ID and path to a JSON file."""
    model_info = {'run_id': run_id, 'model_path': model_path}
    # os.makedirs(os.path.dirname(file_path),exist_ok=True)
    file_path.parent.mkdir(parents=True,exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)
        
   


def main():
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:
        # model loading
        model_path=Path("artifacts/model/model.joblib")
        # preprocessor_path=Path("artifacts/model/preprocessor.joblib")
        model=load_model(model_path=model_path)
       
        
        # data loading
        x_test_url=Path("data/interim/x_test_final.csv")
        y_test_url=Path("data/interim/y_test.csv")
        metrics_dict=predict_model(model=model,x_test_url=x_test_url,y_test_url=y_test_url)
        print("metrics dict returened")
        
        #save metrics
        metrics_path=Path("reports/metrics.json")
        save_metrics(metrics=metrics_dict,file_path=metrics_path)
        
        
        
         # Log metrics to MLflow
        for metric_name, metric_value in metrics_dict.items():
            mlflow.log_metric(metric_name, metric_value)
        print('metrics logged to mlflow')
            
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
                
        # ❗ CORRECT MODEL LOGGING for MLflow Registry
        mlflow.sklearn.log_model(model, artifact_path="model_path")
        print("Model logged successfully!")
        
        
        save_model_info(run.info.run_id, "model_path", Path('reports/experiment_info.json'))
        print('save_model_info')
            
         # Log the metrics file to MLflow
        mlflow.log_artifact('reports/metrics.json')
        print("metrics.json saved")

        # Log the model info file to MLflow
        mlflow.log_artifact('reports/experiment_info.json')

    
        
if __name__ =="__main__":
    main()

        
        
        