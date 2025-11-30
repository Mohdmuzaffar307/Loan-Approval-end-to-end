import pandas as pd
import numpy
from sklearn.preprocessing import OneHotEncoder,Normalizer,StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import os
from pathlib import Path

def load_data(x_train_url :str,x_test_url :str)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    x_train=pd.read_csv(x_train_url)
    x_test=pd.read_csv(x_test_url)
    y_train=x_train.iloc[:,-1]
    y_test=x_test.iloc[:,-1]
    x_train=x_train.drop(columns='loan_approved')
    x_test=x_test.drop(columns='loan_approved')
    print('load_data')
    return x_train,x_test,y_train,y_test

def normalization(x_train:pd.DataFrame,x_test:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    num_cols = x_train.select_dtypes(include=['int32', 'int64', 'float64']).columns
    cat_cols = x_train.select_dtypes(include='object').columns

# Preprocessing steps
    preprocessor = ColumnTransformer(transformers=[
        ("normalization", StandardScaler(), num_cols),
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
         ],remainder='passthrough')
    
    preprocessor.set_output(transform="pandas")
    
    preprocessor.fit(x_train)
    model_path=Path("artifacts/model/preprocessor.joblib")
    model_path.parent.mkdir(parents=True,exist_ok=True)
    # os.makedirs(os.path.dirname(model_path),exist_ok=True)
    joblib.dump(preprocessor, open(model_path,'wb'))
    
    x_train_transformed=preprocessor.transform(x_train)
    x_test_transformed=preprocessor.transform(x_test)
    print("normalization")
    
    return x_train_transformed,x_test_transformed


def save_data(x_train_path:Path,x_test_path:Path,x_train:pd.DataFrame,x_test:pd.DataFrame,y_train:pd.DataFrame,y_train_path:Path,y_test_path:str,y_test:pd.DataFrame)-> None:
    # os.makedirs(os.path.dirname(x_train_path),exist_ok=True)
    x_train_path.parent.mkdir(parents=True,exist_ok=True)
    
    # os.makedirs(os.path.dirname(x_test_path),exist_ok=True)
    x_test_path.parent.mkdir(parents=True,exist_ok=True)
    
    # os.makedirs(os.path.dirname(y_train_path),exist_ok=True)
    y_train_path.parent.mkdir(parents=True,exist_ok=True)
    
    # os.makedirs(os.path.dirname(y_test_path),exist_ok=True)
    y_test_path.parent.mkdir(parents=True,exist_ok=True)
    
    x_train.to_csv(x_train_path,index=False)
    x_test.to_csv(x_test_path,index=False)
    y_test.to_csv(y_test_path,index=False)
    y_train.to_csv(y_train_path,index=False)
    print('save_data')
    
    
def main():
    x_train_url=Path("data/processed/x_train_process.csv")
    x_test_url=Path("data/processed/x_test_process.csv")
    y_train_url=Path("data/interim/y_train.csv")
    y_test_url=Path("data/interim/y_test.csv")
    x_train,x_test,y_train,y_test=load_data(x_train_url=x_train_url,x_test_url=x_test_url)
    
    x_train_final,x_test_final=normalization(x_train=x_train,x_test=x_test)
    
    x_train_final_path=Path("data/interim/x_train_final.csv")
    x_test_final_path=Path("data/interim/x_test_final.csv")
    save_data(x_train_path=x_train_final_path,x_test_path=x_test_final_path,y_train_path=y_train_url,y_test_path=y_test_url,x_train=x_train_final,x_test=x_test_final,y_train=y_train,y_test=y_test)
    
    
if __name__ =="__main__":
    main()
    