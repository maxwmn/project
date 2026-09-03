"""
data_loading.py

Simple functions for loading the water treatment plant dataset.
Each function does one clear job.
"""

import pandas as pd


def load_data(csv_path):
    """
    Load the water treatment plant CSV file into a pandas DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file, for example "../data/water_treatment_plant.csv".

    Returns
    -------
    pandas.DataFrame
        The full dataset, one row per day.
    """
    data = pd.read_csv(csv_path)
    return data


def get_input_feature_columns():
    """
    Return the list of input feature columns we use for clustering.

    These columns all end in "-E", meaning they describe the water
    arriving at the plant (the plant input). We cluster on the incoming
    water because that is the challenge the plant has to deal with, and
    it is the easiest part to interpret.

    Returns
    -------
    list of str
        The names of the 9 input feature columns.
    """
    input_features = [
        "Q-E",     # incoming flow
        "ZN-E",    # incoming zinc
        "PH-E",    # incoming pH
        "DBO-E",   # incoming biochemical oxygen demand (organic load)
        "DQO-E",   # incoming chemical oxygen demand (organic load)
        "SS-E",    # incoming suspended solids
        "SSV-E",   # incoming volatile suspended solids (percent)
        "SED-E",   # incoming sediments
        "COND-E",  # incoming conductivity
    ]
    return input_features
