"""Functions for loading the datasets.

These functions load the CSV files and give a quick overview of the
data. They keep the loading logic in one place so the notebooks stay
short.
"""

import pandas as pd

from src import config


def load_csv(path):
    """Load a CSV file and print basic information about it.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        The loaded data.
    """
    df = pd.read_csv(path)
    print("Loaded file:", path)
    print("Rows:", df.shape[0], "Columns:", df.shape[1])
    return df


def load_air_quality():
    """Load the air quality dataset from data/raw.

    Returns
    -------
    pandas.DataFrame
        The air quality data.
    """
    return load_csv(config.AIR_QUALITY_FILE)


def load_power():
    """Load the household power dataset from data/raw.

    Returns
    -------
    pandas.DataFrame
        The power data.
    """
    return load_csv(config.POWER_FILE)


def quick_check(df):
    """Return a small dictionary that describes the data.

    The dictionary contains the shape, the column names, the number of
    missing values per column, and the first rows. This is useful for a
    fast first look at any dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The data to check.

    Returns
    -------
    dict
        Keys: shape, columns, missing_values, head.
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isna().sum(),
        "head": df.head(),
    }
    return info
