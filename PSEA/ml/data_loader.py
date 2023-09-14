
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


data_path = "data.xlsx"
def get_data(path=data_path):
    data  = pd.read_excel(path)
    group = data.pop("group")
    X = np.array(data)
    names = X[:,:1]
    X = X[:,1:].astype(np.float32)
    y = np.array(group)
    return names, X, y


def get_split_data(path=data_path,test_size=0.2,seed=0):
    names, X, y = get_data(path)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


if __name__ == "__main__":
    pass