import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data(path : str)->pd.DataFrame:
    df=pd.read_csv(path)
    return df

def data_preprocessing(df:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    df=df.drop(columns='name')
    df['points']=df['points'].astype('int32')
    df['loan_approved']=df['loan_approved'].map({True:1,False:0})
    x_train,x_test=train_test_split(df,random_state=42,test_size=0.2)
    return x_train ,x_test

def save_data(x_test_url:Path,x_train_url:Path,x_train:pd.DataFrame,x_test:pd.DataFrame)->None:
    # os.makedirs(os.path.dirname(x_train_url),exist_ok=True)
    x_train_url.parent.mkdir(parents=True, exist_ok=True) 
    
    # os.makedirs(os.path.dirname(x_test_url),exist_ok=True)
    x_test_url.parent.mkdir(parents=True, exist_ok=True) 
    
    x_train.to_csv(x_train_url,index=False)
    x_test.to_csv(x_test_url,index=False)


def main():
    df=load_data(Path("data/raw/raw_df.csv"))
    print('dataset_loaded')
    
    x_train,x_test=data_preprocessing(df)
    print('train_test done')
    
    x_train_url=Path("data/processed/x_train_process.csv")
    x_test_url=Path("data/processed/x_test_process.csv")
    
    save_data(x_train=x_train,x_test=x_test,x_test_url=x_test_url,x_train_url=x_train_url)
    
if __name__ =="__main__":
    main()


    