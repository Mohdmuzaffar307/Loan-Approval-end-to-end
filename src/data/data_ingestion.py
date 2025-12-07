import numpy as np
import pandas as pd
from pathlib import Path
import kagglehub

def load_data()->pd.DataFrame:
    path = kagglehub.dataset_download("anishdevedward/loan-approval-dataset")
    df = pd.read_csv(f"{path}/loan_approval.csv")
    return df

def save_data(path : Path, df : pd.DataFrame)->None:
    path.parent.mkdir(parents=True, exist_ok=True) 
    df.to_csv(path,index=False)
    
    
def main():
    
    df=load_data()
    path= Path("data/raw/raw_df.csv")
    save_data(path,df)
    
if __name__ =="__main__":
    main()
