"""
anomaly_detection.py

Trains a single Random Forest anomaly-detection model shared across all
wind turbines, using SCADA sensor readings (wind speed, temperatures,
power, electrical signals, etc.).

The anomaly label is derived from the actual fault window of each event
(event_start/event_end in Wind Farm A/event_info.csv), not from
status_type_id: a check against event_info.csv showed status_type_id
stays "Normal Operation" throughout almost the entire real fault lead-up,
and only switches to "Service" after the window ends (once a technician
has already arrived) - so a status-based label mostly taught a model to
recognize post-repair service activity rather than genuine early warning
signs.

Because each turbine's one labeled fault event falls entirely within its
own "prediction" rows, pooling every turbine's "train" rows leaves zero
positive examples to learn from. This script instead holds out whole
turbines for testing: some turbines (with their full timeline, including
their real fault event) are used for training, and the rest are held out
entirely, so the model learns from real fault examples on other turbines
and is evaluated on whether it recognizes an unseen turbine's fault. Train
and test turbines are loaded and processed separately, so the ~1.15M-row,
22-turbine dataset is never held in memory all at once - this container
has limited RAM for a dataset this size.

In addition to each sensor's raw reading, the model also gets two trend
features per sensor: its deviation from that turbine's own trailing 24h
average, and its rate of change over the trailing 6 hours. A real fault
precursor is more likely to show up as a drift away from a turbine's own
recent baseline than as a single unusual point-in-time reading.
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 30)

CLEANED_DATA_DIR = Path("data_cleaning/cleaned_data")
EVENT_INFO_PATH = Path("Wind Farm A/event_info.csv")
OUTPUT_DIR = Path("analysis_output")

TIME_COL = "time_stamp"
ASSET_COL = "asset_id"
STATUS_COL = "status_type_id"
SPLIT_COL = "train_test"
SOURCE_COL = "source_file"
NON_FEATURE_COLS = [TIME_COL, ASSET_COL, "id", SPLIT_COL, STATUS_COL, SOURCE_COL]

# Fraction of turbines (not rows) held out entirely for testing.
TEST_TURBINE_FRACTION = 0.3

N_ESTIMATORS = 150
# Each tree is trained on a bootstrap sample of only this fraction of the
# pooled training rows. With ~800k rows across 15 turbines, letting every
# tree see the full dataset would be slow and memory-hungry for little
# accuracy benefit, since the trees would become highly correlated.
MAX_SAMPLES = 0.3
RANDOM_STATE = 42

# The data is sampled every 10 minutes, so 144 rows = 24 hours and 36 rows = 6 hours.
TREND_WINDOW_24H = 144
TREND_WINDOW_SLOPE_HOURS = 36


def add_trend_features(df, feature_cols):
    """Add trend features for one turbine's own time series: deviation from
    its trailing 24h average, and its rate of change over the trailing 6 hours.
    """
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    for col in feature_cols:
        rolling_mean_24h = df[col].rolling(TREND_WINDOW_24H, min_periods=1).mean()
        df[f"{col}_dev24h"] = (df[col] - rolling_mean_24h).astype("float32")
        df[f"{col}_slope6h"] = (df[col].diff(TREND_WINDOW_SLOPE_HOURS) / TREND_WINDOW_SLOPE_HOURS).astype("float32")
    return df


def load_and_label_turbine(csv_path, events):
    """Load one turbine's cleaned CSV, add its trend features, and label its
    rows using the actual fault window from event_info.csv (dropping the
    ambiguous post-repair rows for anomaly events).
    """
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    # Downcast sensor readings to float32: halves memory use for a dataset
    # this size, with no meaningful precision loss for SCADA measurements.
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    # Remember which file each row came from, for the per-turbine breakdown later.
    df[SOURCE_COL] = csv_path.name

    # Trend features are computed only for the "_avg" sensor columns: their
    # "_min"/"_max"/"_std" siblings would give largely redundant trends, and
    # the bare sensor_44-51 columns are monotonic energy counters (Wh/VArh)
    # for which "deviation from own average" isn't meaningful. This also
    # keeps the final column count (and memory use) manageable.
    trend_source_cols = [c for c in df.columns if c.endswith("_avg")]
    df = add_trend_features(df, trend_source_cols)

    # event_id in event_info.csv matches the numeric filename, e.g. "0_cleaned.csv" -> 0
    event_id = int(csv_path.name.split("_")[0])
    event = events.loc[event_id]
    is_anomaly_event = event["event_label"] == "anomaly"
    in_window = (df[TIME_COL] >= event["event_start"]) & (df[TIME_COL] <= event["event_end"])

    # 1 = inside an anomaly event's fault window, 0 = normal event or before the window.
    df["true_anomaly"] = int(is_anomaly_event) * in_window.astype(int)

    if is_anomaly_event:
        # Rows after event_end are the post-repair service buffer, not part
        # of the labeled fault window - drop them, they are neither a clean
        # "normal" baseline nor part of the fault lead-up.
        after_window = df[TIME_COL] > event["event_end"]
        df = df[~after_window]

    return df


def load_turbine_group(csv_paths, events):
    """Load, label, and pool a group of turbines (e.g. just the train
    turbines, or just the test turbines) into one DataFrame.
    """
    frames = [load_and_label_turbine(csv_path, events) for csv_path in csv_paths]
    return pd.concat(frames, ignore_index=True)


def split_turbine_files(folder, event_info_path, test_fraction, random_state):
    """Decide which whole turbine files go to train vs. test, stratified by
    normal/anomaly event - based only on event_info.csv, before loading any
    of the (much larger) sensor data.
    """
    events = pd.read_csv(event_info_path, sep=";")
    events["event_start"] = pd.to_datetime(events["event_start"])
    events["event_end"] = pd.to_datetime(events["event_end"])
    events = events.set_index("event_id")

    csv_paths = sorted(folder.glob("*_cleaned.csv"))
    event_ids = [int(p.name.split("_")[0]) for p in csv_paths]
    labels = events.loc[event_ids, "event_label"]

    train_paths, test_paths = train_test_split(
        csv_paths, test_size=test_fraction, stratify=labels.values, random_state=random_state
    )
    return train_paths, test_paths, events


def evaluate(y_true, y_pred):
    # Compute standard classification metrics comparing predictions to the truth.
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_rows": len(y_true),
        "n_anomalies": int(y_true.sum()),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: hold out whole turbines for testing (not individual rows), so the
    # model can only learn fault patterns from turbines it never sees at test
    # time. Decided from event_info.csv alone, before loading any sensor data.
    print("Splitting whole turbines into train/test groups (not individual rows)...")
    train_paths, test_paths, events = split_turbine_files(
        CLEANED_DATA_DIR, EVENT_INFO_PATH, TEST_TURBINE_FRACTION, RANDOM_STATE
    )
    print(f"  Train turbines ({len(train_paths)}): {sorted(p.name for p in train_paths)}")
    print(f"  Test turbines ({len(test_paths)}): {sorted(p.name for p in test_paths)}")

    # Step 2: load, add trend features, and label the train and test turbines
    # separately - so all 22 turbines are never pooled into one dataframe at once.
    print("Loading and labeling train turbines...")
    train_data = load_turbine_group(train_paths, events)
    print("Loading and labeling test turbines...")
    test_data = load_turbine_group(test_paths, events)

    # Step 3: pick which columns are actual sensor features (exclude ids/labels/metadata).
    # This includes each sensor's raw reading plus its `_dev24h`/`_slope6h` trend features.
    feature_cols = [c for c in train_data.columns if c not in NON_FEATURE_COLS + ["true_anomaly"]]

    X_train = train_data[feature_cols]
    y_train = train_data["true_anomaly"]
    X_test = test_data[feature_cols]
    y_test = test_data["true_anomaly"]
    test_source_files = test_data[SOURCE_COL].reset_index(drop=True)
    del train_data, test_data

    # Step 4: fill missing values using medians from the pooled training data only,
    # so no information from the test turbines leaks into training. This also covers
    # the trend features' NaNs at the start of each turbine's time series.
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    print(
        f"Training rows: {len(X_train)} ({int(y_train.sum())} anomalous), "
        f"test rows: {len(X_test)} ({int(y_test.sum())} anomalous), "
        f"features: {len(feature_cols)}"
    )

    # Step 5: configure and train one Random Forest classifier on the pooled training data.
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Step 6: predict on the held-out turbines and print overall accuracy/precision/recall/f1.
    y_pred = model.predict(X_test)
    overall = evaluate(y_test, y_pred)
    print("\nOverall performance (turbine-holdout, event-window labels, trend features):")
    for key, value in overall.items():
        print(f"  {key}: {value}")

    # Step 7: break the same model's predictions down per held-out turbine.
    results = pd.DataFrame(
        {"source_file": test_source_files, "true_label": y_test.values, "prediction": y_pred}
    )

    per_turbine = []
    for source_file, group in results.groupby("source_file"):
        metrics = evaluate(group["true_label"], group["prediction"])
        metrics["turbine_file"] = source_file
        per_turbine.append(metrics)

    per_turbine_df = pd.DataFrame(per_turbine)[
        ["turbine_file", "n_rows", "n_anomalies", "accuracy", "precision", "recall", "f1"]
    ]
    # Step 8: save the per-turbine results to a CSV and print them.
    per_turbine_df.to_csv(OUTPUT_DIR / "shared_model_per_turbine.csv", index=False)
    print("\nPer-turbine breakdown (held-out test turbines only):")
    print(per_turbine_df)

    # Step 9: show which sensor features mattered most to the model's decisions.
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=False).head(15)
    print("\nTop 15 features by importance:")
    print(top_features)

    # Step 10: save the overall metrics to a CSV as well.
    pd.DataFrame([overall]).to_csv(OUTPUT_DIR / "shared_model_overall.csv", index=False)


if __name__ == "__main__":
    main()
