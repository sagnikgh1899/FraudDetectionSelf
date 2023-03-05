"""
This module implements the LOF (Local Outlier Factor) using PyOD library to detect anomalies.
"""

import pandas as pd
from pyod.models.lof import LOF


def lof_anomaly_detection(data: pd.DataFrame, contamination: float = 0.055) -> pd.Series:
    """
    Args:
        data : Pandas DataFrame containing the data for fraud detection.
        contamination : The proportion of outliers in the data set
    Returns:
        pandas.Series: A binary vector with 'True' for outliers and 'False' for inliers.
    """
    lof = LOF(contamination=contamination)
    y_pred = lof.fit_predict(data)
    return pd.Series(y_pred == 1, index=data.index)
