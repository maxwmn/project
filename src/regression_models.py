"""Functions that build the regression models used in the project.

Each function returns a scikit-learn model or pipeline that is ready to
be trained. Models that need feature scaling are wrapped in a Pipeline
with StandardScaler. Tree based models do not need scaling.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


def make_linear_regression_pipeline():
    """Build a pipeline with StandardScaler and LinearRegression.

    Returns
    -------
    Pipeline
        Scales the features, then fits linear regression.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def make_ridge_pipeline(alpha=1.0):
    """Build a pipeline with StandardScaler and Ridge regression.

    Ridge is linear regression with a penalty on large weights. The
    penalty strength is controlled by alpha. A larger alpha means a
    stronger penalty.

    Parameters
    ----------
    alpha : float
        Strength of the regularization penalty.

    Returns
    -------
    Pipeline
        Scales the features, then fits Ridge regression.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_polynomial_ridge_pipeline(degree=2, alpha=1.0):
    """Build a pipeline with polynomial features, scaling, and Ridge.

    Polynomial features let a linear model fit curved relationships.
    Higher degrees can fit more complex curves but can also overfit.

    Parameters
    ----------
    degree : int
        Degree of the polynomial features.
    alpha : float
        Strength of the Ridge penalty.

    Returns
    -------
    Pipeline
        Builds polynomial features, scales them, then fits Ridge.
    """
    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_random_forest_regressor():
    """Build a Random Forest Regressor with beginner friendly settings.

    A random forest trains many decision trees and averages them. It
    does not need feature scaling.

    Returns
    -------
    RandomForestRegressor
        A forest with a fixed random_state so results are repeatable.
    """
    return RandomForestRegressor(
        n_estimators=100, # build 100 decision trees
        random_state=42, # make results reproducible
        n_jobs=-1, # use all available CPU cores for faster training
    )


def make_svr_pipeline():
    """Build a pipeline with StandardScaler and SVR.

    SVR is sensitive to feature scale, so scaling is required.
    Note for students: SVR can be slow on large datasets. Use it on a
    smaller sample, for example 10000 rows, when the dataset is big.

    Returns
    -------
    Pipeline
        Scales the features, then fits SVR.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", SVR()),
        ]
    )


def make_xgboost_or_fallback_regressor():
    """Return an XGBoost regressor, or a scikit-learn fallback.

    If the xgboost package is installed, we use XGBRegressor. If it is
    not installed, we use HistGradientBoostingRegressor from
    scikit-learn, which is a similar boosted tree model. This way the
    notebook runs even without xgboost.

    Returns
    -------
    estimator
        An XGBRegressor or a HistGradientBoostingRegressor.
    """
    try:
        from xgboost import XGBRegressor

        print("Using XGBoost regressor.")
        return XGBRegressor(
            n_estimators=200, # build 200 trees
            random_state=42,  # make results reproducible
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        print("xgboost not found. Using HistGradientBoostingRegressor instead.")
        return HistGradientBoostingRegressor(random_state=42)
