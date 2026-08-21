"""Project paths and dataset file locations.

This file defines all the folder paths used in the project.
We use pathlib so the project works on Windows, Mac, and Linux
without changing anything. We never hard-code absolute paths.
"""

from pathlib import Path

# PROJECT_ROOT is the main project folder (week9_regression_project).
# This config.py file lives in src, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folders.
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Output folders for figures and tables.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Dataset files inside data/raw.
AIR_QUALITY_FILE = DATA_RAW_DIR / "air_quality_features.csv"
POWER_FILE = DATA_RAW_DIR / "power_2007_features.csv"


def ensure_output_dirs():
    """Create the output folders if they do not exist yet.

    Calling this before saving makes sure that saving a figure or a
    table never fails just because a folder is missing.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
