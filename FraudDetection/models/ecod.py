"""
This module contains the implementation of the ECOD algorithm.
"""
import pandas as pd
from pyod.models.ecod import ECOD


def ecod_anomaly_detection(data, contamination=0.055):
    """
    Args:
        data: Pandas DataFrame containing the data for fraud detection.
        contamination : The proportion of outliers in the data set
    Returns:
        pandas.Series: A binary vector with 'True' for outliers and 'False' for inliers.
    """
    ecod = ECOD(contamination=contamination)
    y_pred = ecod.fit_predict(data)
    return pd.Series(y_pred == 1, index=data.index)
