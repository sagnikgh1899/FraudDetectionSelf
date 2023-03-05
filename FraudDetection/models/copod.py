"""
COPOD: COPOD (COunt-based POD) algorithm for outlier detection.
COPOD is a powerful algorithm that utilizes the count-based approach for outlier detection.
"""

import pandas as pd
from pyod.models.copod import COPOD


def copod_anomaly_detection(data, contamination=0.055):
    """
    Args:
        data : Pandas DataFrame containing the data for fraud detection.
        contamination : The proportion of outliers in the data set.
    Returns:
        pandas.Series: A binary vector with 'True' for outliers and 'False' for inliers.
    """
    copod = COPOD(contamination=contamination)
    y_pred = copod.fit_predict(data)
    return pd.Series(y_pred == 1, index=data.index)
