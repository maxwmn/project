"""Functions for preparing the data before training.

These functions select the columns we need, remove missing values,
split the data into training and test sets, and create a smaller
teaching sample for very large datasets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def select_regression_data(df, target, features):
    """Keep only the target column and the selected feature columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset.
    target : str
        Name of the target column we want to predict.
    features : list of str
        Names of the input feature columns.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with only the selected columns.
    """
    selected_columns = [target] + list(features)
    return df[selected_columns].copy()


def drop_missing_selected(df):
    """Drop rows that have missing values in the selected columns.

    This function assumes the DataFrame already contains only the
    columns we care about (see select_regression_data). It prints the
    shape before and after so students can see how many rows were
    removed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the selected columns.

    Returns
    -------
    pandas.DataFrame
        The data with no missing values in the selected columns.
    """
    rows_before = df.shape[0]
    cleaned = df.dropna()
    rows_after = cleaned.shape[0]
    print("Rows before dropping missing values:", rows_before)
    print("Rows after dropping missing values:", rows_after)
    print("Rows removed:", rows_before - rows_after)
    return cleaned


def train_test_split_regression(X, y, test_size=0.2, random_state=42):
    """Split features and target into training and test sets.

    This is a thin wrapper around scikit-learn train_test_split so the
    notebooks always use the same settings.

    Parameters
    ----------
    X : pandas.DataFrame
        The input features.
    y : pandas.Series
        The target values.
    test_size : float
        Fraction of the data used for testing.
    random_state : int
        Seed so the split is the same every time.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def make_teaching_sample(df, n_rows=80000, random_state=42):
    """Return a random sample if the dataset is large.

    Large datasets can make training slow during a lecture. This
    function returns a smaller random sample so the code runs quickly.
    If the dataset already has fewer rows than n_rows, it is returned
    unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset.
    n_rows : int
        Maximum number of rows to keep.
    random_state : int
        Seed so the sample is the same every time.

    Returns
    -------
    pandas.DataFrame
        The sampled data (or the original data if it is small).
    """
    if df.shape[0] <= n_rows:
        return df.copy()
    sample = df.sample(n=n_rows, random_state=random_state)
    print("Original rows:", df.shape[0], "Sampled rows:", sample.shape[0])
    return sample


def add_datetime_column(df, column="Datetime"):
    """Convert a text column into a real datetime column safely.

    If the column does not exist, the DataFrame is returned unchanged.
    Values that cannot be parsed become missing (NaT) instead of
    raising an error.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset.
    column : str
        Name of the column to convert.

    Returns
    -------
    pandas.DataFrame
        A copy of the data with the column converted to datetime.
    """
    result = df.copy()
    if column in result.columns:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    return result
