import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
TRAINPATH="/home/miki/Desktop/Deployment/bacillus-anthracis/data/traincsv/train.csv" 
TESTPATH="/home/miki/Desktop/Deployment/bacillus-anthracis/data/testcsv/test.csv" 

# create a function that import data and preprocess

def data_load(path = "."):
    """a function to import the data into
    working environment
    ============================================== 
    ARGUMENTS: uses the path as an argument
    RETURNS:pandas dataframe
    ==============================================
    """
    if os.path.exists(path = "."):
        df = pd.read_csv(path,header= 0, parse_dates=True, squeeze = True)
        df['date'] = pd.to_datetime(df['date'])
        df
        return df
    else:
        print("File doesn't exist")    
def summary_statistics(x):
    """"""
    if x.dtypes == 'int64'or x.dtypes == 'float64':
        return pd.DataFrame([[x.name, np.mean(x), np.std(x), np.median(x), np.var(x), np.min(x), np.max(x)]], 
                            columns = ["Variable", "Mean", "Standard Deviation", "Median", "Variance", "Minimum", "Maximum"]).set_index("Variable")
    else:
        print("None")
def check_index(x):
    dups = ["date", "case"]
    for dups in (x[x.index.duplicated()]):
        print('The duplications are found')
        x = x[~x.index.duplicated()]
        return x
    else:
        print('No duplication is found')
    return x
def data_normalize(x):
    
    scaler = RobustScaler()
    x = scaler.fit_transform(np.array(x).reshape(-1, 1)) 
    return x  
def create_trtst(data):
    train_dfs=[]
    val_dfs = []
    train_size = int(len(data) * 0.8)
    df_train = data.iloc[0:train_size], 
    df_test = data.iloc[train_size:len(data)]
    train_dfs.append(df_train)
    val_dfs.append(df_test)
    tr = pd.DataFrame([train_dfs])
    te = pd.DataFrame([val_dfs])
    tr.to_csv('TRAINPATH', index=False)
    te.to_csv('TESTPATH', index=False)