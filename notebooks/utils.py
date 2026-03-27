import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def pred_power(col:str, target:str, df):
    """ Function that creates a logistic regression and returns the auc of that feature

    Args:
        col (str): Name of the feature
        target (str): Name of the target feature
        df: pandas dataframe with the features and target
    """
    lr = LogisticRegression()
    X = df[[col]]
    y = df[target]
    lr.fit(X, y)
    y_pred = lr.predict_proba(X)[:, 0]
    return roc_auc_score(y, y_pred)


def filter_corr(df, df_power, col="feature", col2="auc", thr=0.8):
    """
    Filter features by correlation

    Args:
        df (pandas dataframe): pandas dataframe with the features and target
        df_power (pandas dataframe): pandas dataframe with the predictive power of the features
        col (str, optional) : Name of the column with the feature names
        thr (float, optional): Threshold. Defaults to 0.8.
    """
    
    df_power = df_power.sort_values(by=col2, ascending=False)
    sorted_features = df_power.loc[:, col].tolist()
    
    features_to_keep = []
    features_to_drop = set()
    df_corr = df.loc[:, sorted_features].corr()
    
    for feature in sorted_features:
        if feature not in features_to_drop:
            features_to_keep.append(feature)
            correlations = df_corr[feature].abs()
            feats_correlated = correlations[correlations > thr].index.tolist()
            
            for corr_feat in feats_correlated:
                if corr_feat != feature:
                    features_to_drop.add(corr_feat)
    
    return features_to_keep