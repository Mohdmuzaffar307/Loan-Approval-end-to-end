import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def load_data(x_train_url:str,y_train_url:str)->tuple[pd.DataFrame,pd.DataFrame]:
    x_train=pd.read_csv(x_train_url)
    y_train=pd.read_csv(y_train_url)
    return x_train,y_train
    
def model_building(x_train:pd.DataFrame,y_train:pd.DataFrame):
    model=RandomForestClassifier(n_estimators=120,max_depth=20)
    model.fit(x_train,y_train)
    model_path=r"artifacts\model\model.joblib"
    os.makedirs(os.path.dirname(model_path),exist_ok=True)
    joblib.dump(model,open(model_path,'wb'))
    print('model dumped')
    
def main():
    x_train_url=r"data\interim\x_train_final.csv"
    y_train_url=r"data\interim\y_train.csv"
    x_train,y_train=load_data(x_train_url=x_train_url,y_train_url=y_train_url)
    print('dataset loaded')
    
    model_building(x_train,y_train)

if __name__ =="__main__":
    main()
    

    