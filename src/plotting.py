"""Plotting functions for the regression project.

All plots use Matplotlib only. Each function can optionally save the
figure to a file. We keep the plots simple so they are easy to read.
"""

import matplotlib.pyplot as plt
import numpy as np


def _save_if_needed(fig, save_path):
    """Save a figure to disk if a save path is given."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved figure to:", save_path)


def plot_actual_vs_predicted(y_true, y_pred, title, save_path=None):
    """Plot the real target values against the predicted values.

    A perfect model would place every point on the diagonal line. The
    closer the points are to the line, the better the predictions.

    Parameters
    ----------
    y_true : array-like
        The real target values.
    y_pred : array-like
        The predicted values.
    title : str
        Title for the plot.
    save_path : str or Path, optional
        Where to save the figure. If None, the figure is not saved.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    # Data points as red crosses, like in the lecture slides.
    ax.scatter(y_true, y_pred, color="red", marker="x", alpha=0.4, label="predictions")

    # The diagonal line shows where prediction equals the real value.
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    ax.plot([low, high], [low, high], color="teal", linewidth=2, label="perfect line")

    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")
    ax.set_title(title)
    ax.legend()
    _save_if_needed(fig, save_path)
    return fig


def plot_residuals(y_true, y_pred, title, save_path=None):
    """Plot residuals against the predicted values.

    A residual is the real value minus the predicted value. Good models
    usually have residuals scattered around zero with no clear pattern.

    Parameters
    ----------
    y_true : array-like
        The real target values.
    y_pred : array-like
        The predicted values.
    title : str
        Title for the plot.
    save_path : str or Path, optional
        Where to save the figure. If None, the figure is not saved.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, color="red", marker="x", alpha=0.4)
    # A horizontal line at zero helps us see if errors are balanced.
    ax.axhline(0, color="orange", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual (actual minus predicted)")
    ax.set_title(title)
    _save_if_needed(fig, save_path)
    return fig


def plot_metric_comparison(metrics_df, metric, title, save_path=None):
    """Plot one metric for several models as a bar chart.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Table with a "Model" column and the metric column.
    metric : str
        Which metric to plot, for example "RMSE".
    title : str
        Title for the plot.
    save_path : str or Path, optional
        Where to save the figure. If None, the figure is not saved.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(metrics_df["Model"], metrics_df[metric], color="teal")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_title(title)
    # Rotate the model names so long names stay readable.
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_if_needed(fig, save_path)
    return fig


def plot_feature_importance(feature_names, importances, title, save_path=None):
    """Plot feature importance values as a horizontal bar chart.

    Parameters
    ----------
    feature_names : list of str
        Names of the features.
    importances : array-like
        Importance value for each feature.
    title : str
        Title for the plot.
    save_path : str or Path, optional
        Where to save the figure. If None, the figure is not saved.
    """
    feature_names = np.asarray(feature_names)
    importances = np.asarray(importances)

    # Sort so the most important feature is at the top.
    order = np.argsort(importances)
    sorted_names = feature_names[order]
    sorted_values = importances[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(sorted_names, sorted_values, color="teal")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    _save_if_needed(fig, save_path)
    return fig
