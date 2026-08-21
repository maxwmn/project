"""Functions for evaluating regression models.

These functions calculate metrics, evaluate a single model, and
compare several models on the same training and test data.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, and R2 for regression predictions.

    MAE is the average absolute error.
    RMSE punishes large errors more strongly than MAE.
    R2 shows how much variation the model explains compared with always
    predicting the average value.

    Parameters
    ----------
    y_true : array-like
        The real target values.
    y_pred : array-like
        The values predicted by the model.

    Returns
    -------
    dict
        Keys: MAE, RMSE, R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    # We compute RMSE as the square root of the mean squared error.
    # This works the same way across all scikit-learn versions.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def evaluate_regression_model(name, model, X_train, X_test, y_train, y_test):
    """Fit a model, predict on the test set, and return its metrics.

    Parameters
    ----------
    name : str
        A short name for the model, used in the results.
    model : estimator
        A scikit-learn model or pipeline.
    X_train, X_test : array-like
        Training and test features.
    y_train, y_test : array-like
        Training and test targets.

    Returns
    -------
    tuple
        (metrics_dict, y_pred). metrics_dict also contains the model
        name under the key "Model".
    """
    # The model only ever learns from the training data.
    model.fit(X_train, y_train)
    # We make predictions on the test data the model has not seen.
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    metrics["Model"] = name
    return metrics, y_pred


def compare_regression_models(models, X_train, X_test, y_train, y_test):
    """Train several models and collect their metrics in one table.

    Parameters
    ----------
    models : dict
        A dictionary mapping a model name to a scikit-learn model.
    X_train, X_test : array-like
        Training and test features.
    y_train, y_test : array-like
        Training and test targets.

    Returns
    -------
    pandas.DataFrame
        One row per model with columns Model, MAE, RMSE, and R2,
        sorted from lowest RMSE to highest RMSE.
    """
    rows = []
    for name, model in models.items():
        metrics, _ = evaluate_regression_model(
            name, model, X_train, X_test, y_train, y_test
        )
        rows.append(metrics)

    results = pd.DataFrame(rows)
    # Put the model name first, then the metrics.
    results = results[["Model", "MAE", "RMSE", "R2"]]
    # Lower RMSE is better, so we sort by RMSE.
    results = results.sort_values("RMSE").reset_index(drop=True)
    return results
