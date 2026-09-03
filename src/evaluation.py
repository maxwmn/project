"""
evaluation.py

Functions that help us choose a sensible number of clusters (K).
We use two simple tools: the elbow method and the silhouette score.
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_inertia_for_range(scaled_values, k_values, random_state=42):
    """
    Run K-Means for several values of K and record the inertia each time.

    "Inertia" is scikit-learn's name for the distortion cost J from the
    lecture: the total squared distance from each point to its own
    centroid. Lower inertia means tighter groups. We plot these values
    later to look for an elbow.

    Parameters
    ----------
    scaled_values : numpy.ndarray
        The standardized features.
    k_values : list of int
        The values of K to try, for example range(1, 9).
    random_state : int
        A fixed seed so results are repeatable.

    Returns
    -------
    list of float
        The inertia for each K, in the same order as k_values.
    """
    inertia_list = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        model.fit(scaled_values)
        inertia_list.append(model.inertia_)
    return inertia_list


def compute_silhouette_for_range(scaled_values, k_values, random_state=42):
    """
    Run K-Means for several values of K and record the silhouette score.

    The silhouette score asks, for each point, whether it sits well inside
    its own group or near the border with another group. The average score
    ranges from about -1 to 1, where higher is better. We skip K=1 because
    the silhouette score needs at least 2 clusters to be defined.

    Parameters
    ----------
    scaled_values : numpy.ndarray
        The standardized features.
    k_values : list of int
        The values of K to try. Values below 2 are ignored.
    random_state : int
        A fixed seed so results are repeatable.

    Returns
    -------
    tested_k : list of int
        The K values that were actually scored (2 and above).
    scores : list of float
        The average silhouette score for each tested K.
    """
    tested_k = []
    scores = []
    for k in k_values:
        if k < 2:
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(scaled_values)
        score = silhouette_score(scaled_values, labels)
        tested_k.append(k)
        scores.append(score)
    return tested_k, scores
