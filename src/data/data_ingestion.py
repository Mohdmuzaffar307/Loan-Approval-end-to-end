import numpy as np
import pandas as pd
from pathlib import Path

def load_data(url:str)->pd.DataFrame:
    df=pd.read_csv(url)
    return df

def save_data(path : Path, df : pd.DataFrame)->None:
    path.parent.mkdir(parents=True, exist_ok=True) 
    df.to_csv(path,index=False)
    
    
def main():
    url=r"C:/Users/mdmuz/.cache/kagglehub/datasets/anishdevedward/loan-approval-dataset\versions/1/loan_approval.csv"
    df=load_data(url)
    path= Path("data/raw/raw_df.csv")
    save_data(path,df)
    
if __name__ =="__main__":
    main()
