import pandas as pd
import joblib
import streamlit as st
import mlflow
import os
import json
import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPORT_PATH = os.path.join(BASE_DIR, "reports", "experiment_info.json")



dagshub_username = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_PAT")

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


# @st.cache_resource
def load_model_and_preprocessor():
    # Load experiment info
    with open(REPORT_PATH) as file:
        data = json.load(file)

    run_id = data["run_id"]
    model_path = data["model_path"]

    # Load MLflow model from RUN
    model_uri = f"runs:/{run_id}/{model_path}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load preprocessor
    preprocessor_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="preprocessor/preprocessor.joblib"
    )
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor

#Streamlit app code
model,preprocessor=load_model_and_preprocessor()
st.sidebar.title("Loan Approval")
st.sidebar.header("Parameters")
st.sidebar.markdown("Adjust the parameters below:")

# preprocessor = joblib.load(open(r"artifacts\model\preprocessor.joblib", "rb"))
# model = joblib.load(open(r"artifacts\model\model.joblib", "rb"))


st.title('Loan Approval App')
city = st.sidebar.text_input('Enter City', "")

income = st.sidebar.number_input("Enter income", value=0)
credit_score = st.sidebar.number_input("Creadit Score", value=0)
loan_amount = st.sidebar.number_input("loan_amount", value=0)
years_employed = st.sidebar.number_input("years_employed", value=0)
points = st.sidebar.number_input("point", value=0)

value_predict = pd.DataFrame(
    {
        "points": [points],
        "income": [income],
        "credit_score": [credit_score],
        "loan_amount": [loan_amount],
        "years_employed": [years_employed],
        "city": [city]
    }
)

predict_btn = st.sidebar.button('Predict')

if predict_btn:
    value_predict = preprocessor.transform(value_predict)
    print(value_predict.head(1))
    value = model.predict(value_predict)
    st.write(value)
