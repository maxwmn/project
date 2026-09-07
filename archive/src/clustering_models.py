"""
clustering_models.py

Functions for scaling the features and running K-Means.
We keep one job per function so the workflow stays clear.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def scale_features(features):
    """
    Standardize the features so every column is on a comparable scale.

    K-Means uses distances between points. If one feature has very large
    numbers (like flow, around 35000) and another has small numbers (like
    pH, around 8), the large feature would dominate the distance and the
    small feature would be ignored. Standardizing fixes this: each feature
    is centered at 0 and given a similar spread.

    Parameters
    ----------
    features : pandas.DataFrame
        Numeric features with no missing values.

    Returns
    -------
    scaled_values : numpy.ndarray
        The standardized features, ready for K-Means.
    scaler : StandardScaler
        The fitted scaler, kept in case you want to reverse the scaling
        later for interpretation.
    """
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features)
    return scaled_values, scaler


def run_kmeans(scaled_values, n_clusters, random_state=42):
    """
    Run the K-Means algorithm on already scaled data.

    We set n_init=10, which means scikit-learn runs K-Means 10 times with
    different random starts and keeps the best result (lowest distortion).
    This protects us from one unlucky random start, exactly as discussed
    in the lecture.

    Parameters
    ----------
    scaled_values : numpy.ndarray
        The standardized features.
    n_clusters : int
        The number of clusters (K) to find.
    random_state : int
        A fixed seed so results are repeatable in class.

    Returns
    -------
    model : KMeans
        The fitted K-Means model. Use model.labels_ for the cluster of
        each point and model.cluster_centers_ for the centroids.
    """
    model = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
    )
    model.fit(scaled_values)
    return model
