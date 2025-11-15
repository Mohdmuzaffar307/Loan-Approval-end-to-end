import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.sidebar.title("Loan Approval")
st.sidebar.header("Parameters")
st.sidebar.markdown("Adjust the parameters below:")

preprocessor=joblib.load(open(r"E:\MLOPS-Projects\loan-approval-end-to-end\artifacts\model\preprocessor.joblib","rb"))
model=joblib.load(open(r"E:\MLOPS-Projects\loan-approval-end-to-end\artifacts\model\model.joblib","rb"))

st.title('Loan Approval App')
# df=pd.read_csv(r'E:\MLOPS-Projects\loan-approval-end-to-end\data\raw\raw_df.csv')
city=st.sidebar.text_input('Enter City',"")

income=st.sidebar.number_input("Enter income", value=0)
credit_score=st.sidebar.number_input("Creadit Score", value=0)
loan_amount=st.sidebar.number_input("loan_amount", value=0)
years_employed=st.sidebar.number_input("years_employed", value=0)
points=st.sidebar.number_input("point", value=0)

value_predict=pd.DataFrame({"points":[points],
                            "income":[income],
                            "credit_score":[credit_score],
                            "loan_amount":[loan_amount],
                            "years_employed":[years_employed],
                            "city":[city]  
                            })

# print(value_predict.head(1))
predict_btn=st.sidebar.button('Predict')


if predict_btn:
    value_predict=preprocessor.transform(value_predict)
    print(value_predict.head(1))
    value=model.predict(value_predict)
    st.write(value)