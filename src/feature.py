import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tsmoothie.utils_func import create_windows

# create a function that import data and preprocess

def data_load(path):
    """a function to import the data into
    working environment 
    ARGUMENTS: uses the path as an argument
    RETURNS:
    """
    if os.path.exists(path):
        df = pd.read_csv("path",header= 0, index_col= 0,  parse_dates=True, squeeze = True)
        df= df.replace('[./d.]', '', regex = True).astype(int)
    else:
        print("File doesn't exist")
    return df.dropna()
    
def evenly_spaced(data):
    df = data.pivot(index="date", columns="case", values="value")
    final_df = df.ffill()
    return final_df


def data_normalize(df, x, y):
    
    scaler = RobustScaler()
    df = scaler.fit_transform(np.array(df[["x", "y"]]).reshape(-1, 1))
    return df


def data_split(data):
    """funtion to split the dataset into train and 
    test set using the argument
    ==============================================
    ARGUMENT: data = df"""
    train, test = data[:670], data[670:]
    test = test.reset_index(drop=True, inplace=True)
    return train, test

def window_split(X= None, window_size = 12, end= None):
    """"""
    X = create_windows(X, window_shape = window_size, end_id= end)
    return X.shape
