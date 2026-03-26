import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def pred_power(col:str, target:str, df):
    """ Function that creates a logistic regression and returns the auc of that feature

    Args:
        col (str): Name of the feature
        target (str): Name of the target feature
        df: pandas dataframe
    """
    lr = LogisticRegression()
    X = df[[col]]
    y = df[target]
    lr.fit(X, y)
    y_pred = lr.predict_proba(X)[:, 0]
    return roc_auc_score(y, y_pred)
    