"""
data_cleaning.py

Simple cleaning helpers for the water treatment plant dataset.
We keep things easy to read so you can follow every step.
"""

import pandas as pd


def clean_column_names(data):
    """
    Tidy up the column names by removing extra spaces.

    The original headers are short codes like "Q-E" and "DBO-E".
    We do not rename them, because the codes carry meaning, but we
    strip any accidental spaces so lookups are reliable.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        A copy with cleaned column names.
    """
    data = data.copy()
    data.columns = [str(col).strip() for col in data.columns]
    return data


def select_features(data, feature_columns):
    """
    Keep only the columns we want to cluster on.

    Parameters
    ----------
    data : pandas.DataFrame
    feature_columns : list of str
        The columns to keep.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with only the selected feature columns.
    """
    features = data[feature_columns].copy()
    return features


def count_missing_values(data):
    """
    Count how many values are missing in each column.

    Returns
    -------
    pandas.Series
        The number of missing values per column, only for columns
        that actually have at least one missing value.
    """
    missing_counts = data.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    missing_counts = missing_counts.sort_values(ascending=False)
    return missing_counts


def fill_missing_with_median(data):
    """
    Fill missing values in each numeric column with that column's median.

    We use the median (the middle value) instead of the mean because the
    median is not pulled around by a few very large outlier days. This is
    simple, keeps every row, and is safe before scaling.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric features only.

    Returns
    -------
    pandas.DataFrame
        A copy with no missing values.
    """
    data = data.copy()
    for column in data.columns:
        median_value = data[column].median()
        data[column] = data[column].fillna(median_value)
    return data

def fill_missing_with_mean(data):
    """
    Fill missing values in each numeric column with that column's mean.

    The mean is the average value of the column. It is simple and keeps
    every row, but it can be affected by very large or unusual values.
    We use this function mainly to compare with median filling and see
    whether the clustering result changes.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric features only.

    Returns
    -------
    pandas.DataFrame
        A copy with no missing values.
    """
    data = data.copy()
    for column in data.columns:
        mean_value = data[column].mean()
        data[column] = data[column].fillna(mean_value)
    return data
