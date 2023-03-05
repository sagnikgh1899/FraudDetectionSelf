"""
This module contains the implementation of the Angle-based Outlier Detection (ABOD) algorithm.
"""
import pandas as pd
from pyod.models.abod import ABOD


def abod_anomaly_detection(data, contamination=0.055):
    """
    Args:
        data : Pandas DataFrame containing the data for fraud detection.
        contamination : The proportion of outliers in the data set
    Returns:
        pandas.Series: A binary vector with 'True' for outliers and 'False' for inliers.
    """
    abod = ABOD(contamination=contamination)
    y_pred = abod.fit_predict(data)
    return pd.Series(y_pred == 1, index=data.index)
