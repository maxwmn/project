"""
anomaly_detection.py

Trains a single Random Forest anomaly-detection model shared across all
wind turbines, using SCADA sensor readings (wind speed, temperatures,
power, electrical signals, etc.). All "train" rows from every cleaned
turbine file in data_cleaning/cleaned_data/ are pooled to fit one model,
which is then evaluated on the pooled "prediction" rows - both overall
and broken down per turbine, to see how well the single model generalizes
across turbines.
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 30)

CLEANED_DATA_DIR = Path("data_cleaning/cleaned_data")
OUTPUT_DIR = Path("analysis_output")

TIME_COL = "time_stamp"
ASSET_COL = "asset_id"
STATUS_COL = "status_type_id"
SPLIT_COL = "train_test"
SOURCE_COL = "source_file"
NON_FEATURE_COLS = [TIME_COL, ASSET_COL, "id", SPLIT_COL, STATUS_COL, SOURCE_COL]

# Status 0 (normal operation) and 2 (idling) both count as normal
# behaviour; every other status is treated as an anomaly.
NORMAL_STATUS_IDS = {0, 2}

# Set to e.g. 10 during development to keep only every 10th row (across
# both train and test), so the script runs in seconds instead of minutes
# and won't exhaust memory on a laptop while you're iterating. Set back to
# 1 to use the full dataset for a real run.
SAMPLE_EVERY_NTH_ROW = 1

N_ESTIMATORS = 150
# Each tree is trained on a bootstrap sample of only this fraction of the
# pooled training rows. With ~1.2 million rows across 22 turbines, letting
# every tree see the full dataset would be slow and memory-hungry for
# little accuracy benefit, since the trees would become highly correlated.
MAX_SAMPLES = 0.3
RANDOM_STATE = 42


def load_all_turbines(folder):
    """Load and concatenate every cleaned turbine CSV into one DataFrame."""
    frames = []
    for csv_path in sorted(folder.glob("*_cleaned.csv")):
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.dropna(subset=[TIME_COL])
        # Downcast sensor readings to float32: halves memory use for a
        # dataset this size, with no meaningful precision loss for SCADA
        # measurements.
        float_cols = df.select_dtypes(include="float64").columns
        df[float_cols] = df[float_cols].astype("float32")
        df[SOURCE_COL] = csv_path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_labels(df):
    """1 = anomalous status, 0 = normal operation or idling."""
    return (~df[STATUS_COL].isin(NORMAL_STATUS_IDS)).astype(int)


def build_feature_matrix(df, feature_cols):
    X = df[feature_cols]
    y = make_labels(df)
    return X, y


def evaluate(y_true, y_pred):
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

    print("Loading and pooling all turbine files...")
    data = load_all_turbines(CLEANED_DATA_DIR)

    if SAMPLE_EVERY_NTH_ROW > 1:
        data = data.iloc[::SAMPLE_EVERY_NTH_ROW].reset_index(drop=True)
        print(f"Dev mode: keeping every {SAMPLE_EVERY_NTH_ROW}th row -> {len(data)} rows")

    feature_cols = [c for c in data.columns if c not in NON_FEATURE_COLS]

    train_df = data[data[SPLIT_COL] == "train"]
    test_df = data[data[SPLIT_COL] == "prediction"].copy()

    X_train, y_train = build_feature_matrix(train_df, feature_cols)
    X_test, y_test = build_feature_matrix(test_df, feature_cols)

    # Fill missing values using medians from the pooled training data only,
    # so no information from the test rows leaks into training.
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    print(
        f"Training rows: {len(X_train)}, test rows: {len(X_test)}, "
        f"turbines: {data[SOURCE_COL].nunique()}"
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    overall = evaluate(y_test, y_pred)
    print("\nOverall performance (one shared model, pooled across all turbines):")
    for key, value in overall.items():
        print(f"  {key}: {value}")

    # Break the same model's predictions down per turbine, to see how well
    # one shared model generalizes to each individual turbine.
    test_df["prediction"] = y_pred
    test_df["true_label"] = y_test.values

    per_turbine = []
    for source_file, group in test_df.groupby(SOURCE_COL):
        metrics = evaluate(group["true_label"], group["prediction"])
        metrics["turbine_file"] = source_file
        per_turbine.append(metrics)

    per_turbine_df = pd.DataFrame(per_turbine)[
        ["turbine_file", "n_rows", "n_anomalies", "accuracy", "precision", "recall", "f1"]
    ]
    per_turbine_df.to_csv(OUTPUT_DIR / "shared_model_per_turbine.csv", index=False)
    print("\nPer-turbine breakdown (same shared model used for every turbine):")
    print(per_turbine_df)

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importances.sort_values(ascending=False).head(15)
    print("\nTop 15 features by importance:")
    print(top_features)

    pd.DataFrame([overall]).to_csv(OUTPUT_DIR / "shared_model_overall.csv", index=False)


if __name__ == "__main__":
    main()
