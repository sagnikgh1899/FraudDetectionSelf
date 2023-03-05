"""
This module contains the implementation of k-Nearest Neighbors (knn) to detect anomalies
in the input data.
"""

import pandas as pd
from pyod.models.knn import KNN


def knn_anomaly_detection(data, n_neighbors=10, contamination=0.055):
    """
    Args:
        data : Pandas DataFrame containing the data for fraud detection.
        n_neighbors : Number of neighbors to consider for fraud detection.
        contamination : The proportion of outliers in the data set.
    Returns:
        pandas.Series: A binary vector with 'True' for outliers and 'False' for inliers.
    """
    knn = KNN(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = knn.fit_predict(data)
    return pd.Series(y_pred == 1, index=data.index)
