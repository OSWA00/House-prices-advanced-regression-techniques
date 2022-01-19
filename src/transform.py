import pandas as pd
import numpy as np


def log_transformation(df, features):
    for feature in features:
        df[feature] = np.log(df[feature])
    return df


def transform_totalBsmtSF(df):
    feature = 'TotalBsmtSF'
    df['HasBsmt'] = np.where(df[feature] > 0, 0, 1)

    def log(x):
        if x > 0:
            return np.log(x)
        else:
            return x
    df[feature] = df[feature].apply(log)
    return df