"""
visualization.py

Plotting helpers using matplotlib only.
Each function makes one clear figure and returns it so the notebook
can save it if needed.
"""

import matplotlib.pyplot as plt


def plot_missing_values(missing_counts):
    """
    Draw a bar chart of how many values are missing per column.

    Parameters
    ----------
    missing_counts : pandas.Series
        Missing value counts, for example from count_missing_values().
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(missing_counts.index, missing_counts.values)
    ax.set_title("Missing values per column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Number of missing values")
    plt.xticks(rotation=90)
    plt.tight_layout()
    return fig


def plot_feature_distributions(features):
    """
    Draw a histogram for each feature, so we can see its spread and any
    extreme values before clustering.

    Parameters
    ----------
    features : pandas.DataFrame
        The numeric features.
    """
    n_features = features.shape[1]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3 * n_rows))
    axes = axes.flatten()
    for i, column in enumerate(features.columns):
        axes[i].hist(features[column].dropna(), bins=30)
        axes[i].set_title(column)
    # Hide any empty subplots
    for j in range(n_features, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    return fig


def plot_elbow(k_values, inertia_list):
    """
    Draw the elbow plot: inertia (distortion) against the number of clusters.

    We look for the point where the curve stops dropping quickly. That bend
    is a hint for a good value of K.

    Parameters
    ----------
    k_values : list of int
    inertia_list : list of float
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_values), inertia_list, marker="o")
    ax.set_title("Elbow method: distortion versus number of clusters")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia (distortion J)")
    plt.tight_layout()
    return fig


def plot_silhouette(tested_k, scores):
    """
    Draw the average silhouette score against the number of clusters.

    Higher is better. We use this together with the elbow plot.

    Parameters
    ----------
    tested_k : list of int
    scores : list of float
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tested_k, scores, marker="o")
    ax.set_title("Silhouette score versus number of clusters")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Average silhouette score")
    plt.tight_layout()
    return fig


def plot_clusters_two_features(features, labels, feature_x, feature_y):
    """
    Draw a scatter plot of two chosen features, colored by cluster.

    This is the simplest way to look at the groups. We pick two features
    that are easy to interpret, for example incoming organic load and
    incoming suspended solids.

    Parameters
    ----------
    features : pandas.DataFrame
        The original (unscaled) features, so the axes show real units.
    labels : array-like
        The cluster label for each row.
    feature_x : str
        Column name for the horizontal axis.
    feature_y : str
        Column name for the vertical axis.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        features[feature_x],
        features[feature_y],
        c=labels,
        cmap="viridis",
        alpha=0.7,
    )
    ax.set_title("Clusters shown with two features")
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    plt.tight_layout()
    return fig


def plot_clusters_pca(pca_values, labels):
    """
    Draw a scatter plot of the data after reducing it to 2 dimensions
    with PCA, colored by cluster.

    PCA is an optional view. It squeezes all 9 features into 2 summary
    axes so we can see the whole grouping in a single picture. The axes
    themselves are combinations of features and are not directly readable,
    so we use this only to get an overall sense of the clusters.

    Parameters
    ----------
    pca_values : numpy.ndarray
        The data projected onto 2 PCA components.
    labels : array-like
        The cluster label for each row.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pca_values[:, 0],
        pca_values[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
    )
    ax.set_title("Clusters shown with PCA (2 summary axes)")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    plt.tight_layout()
    return fig
