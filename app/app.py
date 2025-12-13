import pandas as pd
import joblib
import streamlit as st
import mlflow
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

# Load model as a PyFuncModel.

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)



#Streamlit app code

st.sidebar.title("Loan Approval")
st.sidebar.header("Parameters")
st.sidebar.markdown("Adjust the parameters below:")

preprocessor = joblib.load(open(r"artifacts\model\preprocessor.joblib", "rb"))
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
