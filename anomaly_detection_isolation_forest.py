"""
anomaly_detection_isolation_forest.py

Unsupervised anomaly detection: for each wind turbine individually, fits an
Isolation Forest on that turbine's own historical "train" rows (assumed to
represent normal operation), then scores its "prediction" rows for how
unusual they are - no anomaly labels are used for training at all.

This is much closer to the CARE-to-Compare benchmark's own design (each
file's train/prediction split, exactly one event per file) than the
supervised turbine-holdout approach in archive/anomaly_detection.py /
archive/anomaly_detection_2.0.py (superseded, kept for reference): it
sidesteps the problem of one shared model having to generalize fault
signatures ACROSS turbines with different sensor baselines/calibrations,
since each turbine here gets its own dedicated model that only has to
recognize a deviation from ITS OWN normal behaviour.

The anomaly label (used only to evaluate the model afterwards, never for
training) is the actual fault window from event_info.csv, same as those
scripts - see archive/anomaly_detection.py's docstring for why
status_type_id is not used for this.

CONTAMINATION_VALUES sweeps several contamination settings (Isolation
Forest's expected-anomaly-rate knob) to show the precision/recall trade-off:
each turbine is loaded and feature-engineered only once and cached, then
re-fit for every contamination value, since reloading all 22 CSVs per
value would be far slower for no benefit.
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 30)

CLEANED_DATA_DIR = Path("data_cleaning/cleaned_data")
EVENT_INFO_PATH = Path("Wind Farm A/event_info.csv")
FEATURE_DESCRIPTION_PATH = Path("Wind Farm A/feature_description.csv")
OUTPUT_DIR = Path("analysis_output/isolation_forest")

# Restrict the model to only the temperature sensors (plus power_curve_deviation
# and the trend features derived from them). Every supervised run so far has
# ranked temperature sensors as the most important features by a wide margin;
# this tests whether dropping the other ~25 avg sensors (wind, electrical,
# angles, ...) - noise for Isolation Forest's random axis selection - helps
# it find the real signal instead of splitting on irrelevant columns.
FEATURE_SET = "temperature"  # "all", "temperature", "temperature_wind_gust", or "temperature_mech"

# If True, standardize each turbine's features to its own z-score (mean 0,
# std 1, computed from that turbine's own train rows) before fitting. This
# puts every feature on the same scale, so Isolation Forest's random
# feature/threshold selection isn't implicitly biased towards features with
# wider raw numeric ranges (e.g. a 0-2000 kW power reading vs. a -5..5 trend
# deviation).
Z_SCORE_NORMALIZE = True

TIME_COL = "time_stamp"
ASSET_COL = "asset_id"
STATUS_COL = "status_type_id"
SPLIT_COL = "train_test"
SOURCE_COL = "source_file"
NON_FEATURE_COLS = [TIME_COL, ASSET_COL, "id", SPLIT_COL, STATUS_COL, SOURCE_COL]

WIND_SPEED_COL = "wind_speed_3_avg"
POWER_COL = "power_30_avg"
ROTOR_RPM_COL = "sensor_52_avg"
GENERATOR_RPM_COL = "sensor_18_avg"
PITCH_AVG_COL = "sensor_5_avg"
PITCH_STD_COL = "sensor_5_std"

# The data is sampled every 10 minutes, so 144 rows = 24 hours and 36 rows = 6 hours.
TREND_WINDOW_24H = 144
TREND_WINDOW_SLOPE_HOURS = 36

N_ESTIMATORS = 150
RANDOM_STATE = 42

# Contamination = the fraction of rows Isolation Forest expects to be
# anomalous. "auto" (the earlier default) picks a threshold from the
# isolation-score distribution itself; the numeric values below force it to
# flag roughly that fraction of each turbine's prediction rows as anomalous,
# trading precision for recall as the value increases.
CONTAMINATION_VALUES = ["auto", 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# F-beta weight: beta=2 (F2) weights recall 4x as much as precision in the
# score, reflecting that missing a real fault (false negative) - leaving a
# turbine running when something is wrong - is worse than an unnecessary
# false alarm (false positive), which just means pitching it out of the
# wind when it wasn't strictly necessary.
FBETA_WEIGHT = 2


def get_temperature_avg_columns(feature_description_path):
    """Return the "_avg" column names for every sensor whose description
    mentions "temperature", read from the dataset's own feature_description.csv
    rather than hand-picked, so it stays correct if that file changes.
    """
    descriptions = pd.read_csv(feature_description_path, sep=";")
    is_temperature = descriptions["description"].str.contains("temperature", case=False, na=False)
    return [f"{name}_avg" for name in descriptions.loc[is_temperature, "sensor_name"]]


def add_power_curve_deviation(df):
    """Add a `power_curve_deviation` feature: how far a turbine's actual
    power output is from what a turbine at this wind speed would normally
    produce, based on its own "train" rows.

    A raw power reading confounds two very different things: low wind (a
    turbine simply can't produce power) and a fault (the turbine can't
    produce the power it should for the wind it's getting). This feature
    isolates the second case - a standard technique in wind turbine fault
    detection, usually called a "power curve" model. An isotonic (monotonic,
    non-decreasing) fit is used because a real power curve rises with wind
    speed up to the turbine's rated power and then plateaus, but is never
    expected to decrease as wind speed increases.
    """
    train_mask = df[SPLIT_COL] == "train"
    curve = IsotonicRegression(out_of_bounds="clip", increasing=True)
    curve.fit(df.loc[train_mask, WIND_SPEED_COL], df.loc[train_mask, POWER_COL])
    expected_power = curve.predict(df[WIND_SPEED_COL])
    df["power_curve_deviation"] = (df[POWER_COL] - expected_power).astype("float32")
    return df


def add_gear_ratio_deviation(df):
    """Add a `gear_ratio_deviation` feature: how far a turbine's actual
    generator speed is from what its own gearbox ratio would normally
    produce at this rotor speed, based on its own "train" rows.

    A gearbox has a fixed mechanical ratio between rotor and generator
    speed; slip or wear in the gearbox shows up as a deviation from that
    otherwise-constant ratio. Relevant here since "Gearbox failure" and
    "Generator bearing failure" are two of the fault types in this dataset.
    """
    train_mask = df[SPLIT_COL] == "train"
    curve = IsotonicRegression(out_of_bounds="clip", increasing=True)
    curve.fit(df.loc[train_mask, ROTOR_RPM_COL], df.loc[train_mask, GENERATOR_RPM_COL])
    expected_generator_rpm = curve.predict(df[ROTOR_RPM_COL])
    df["gear_ratio_deviation"] = (df[GENERATOR_RPM_COL] - expected_generator_rpm).astype("float32")
    return df


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
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    df[SOURCE_COL] = csv_path.name

    df = add_power_curve_deviation(df)
    df = add_gear_ratio_deviation(df)

    trend_source_cols = [c for c in df.columns if c.endswith("_avg")]
    df = add_trend_features(df, trend_source_cols)

    event_id = int(csv_path.name.split("_")[0])
    event = events.loc[event_id]
    is_anomaly_event = event["event_label"] == "anomaly"
    in_window = (df[TIME_COL] >= event["event_start"]) & (df[TIME_COL] <= event["event_end"])
    df["true_anomaly"] = int(is_anomaly_event) * in_window.astype(int)

    if is_anomaly_event:
        after_window = df[TIME_COL] > event["event_end"]
        df = df[~after_window]

    return df


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # f2 weights recall higher than precision (see FBETA_WEIGHT above) -
        # a better fit than f1 when missing a fault matters more than a false alarm.
        "f2": fbeta_score(y_true, y_pred, beta=FBETA_WEIGHT, zero_division=0),
        "n_rows": len(y_true),
        "n_anomalies": int(y_true.sum()),
    }


def prepare_turbine_data(csv_paths, events, temperature_avg_cols=None, extra_cols=()):
    """Load, feature-engineer, and median-fill every turbine once, caching
    just the small (X_train, X_test, y_test, name, test_timestamps,
    event_start, is_anomaly_event) tuples needed to fit and score a model
    repeatedly without re-reading the CSVs each time.

    test_timestamps/event_start/is_anomaly_event are kept alongside the
    feature matrices so lead-time (see compute_lead_time_minutes) can be
    computed later without re-loading the CSV.

    If `temperature_avg_cols` is given, features are restricted to those
    columns, `power_curve_deviation`, `gear_ratio_deviation`, `extra_cols`,
    and the `_dev24h`/`_slope6h` trend variants of any "_avg" column among
    them - everything else (wind, electrical, angle sensors, ...) is
    dropped from the model input.
    """
    prepared = []
    for csv_path in csv_paths:
        df = load_and_label_turbine(csv_path, events)
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS + ["true_anomaly"]]

        if temperature_avg_cols is not None:
            base_avg_cols = set(temperature_avg_cols) | {c for c in extra_cols if c.endswith("_avg")}
            allowed = base_avg_cols | {"power_curve_deviation", "gear_ratio_deviation"} | set(extra_cols)
            feature_cols = [
                c for c in feature_cols if c in allowed or c.rsplit("_", 1)[0] in base_avg_cols
            ]

        train_df = df[df[SPLIT_COL] == "train"]
        test_df = df[df[SPLIT_COL] == "prediction"]

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_test = test_df["true_anomaly"]
        test_timestamps = test_df[TIME_COL].reset_index(drop=True)

        event_id = int(csv_path.name.split("_")[0])
        event = events.loc[event_id]
        is_anomaly_event = event["event_label"] == "anomaly"
        event_start = event["event_start"] if is_anomaly_event else None

        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        if Z_SCORE_NORMALIZE:
            # Fit on this turbine's own train rows only, so no information
            # about its test/prediction window leaks into the scaling.
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

        prepared.append((csv_path.name, X_train, X_test, y_test, test_timestamps, event_start, is_anomaly_event))
    return prepared


def compute_lead_time_minutes(y_pred, test_timestamps, event_start):
    """Minutes between the model's first flagged anomaly and the official
    fault window start (event_start): positive = a genuine early warning,
    flagged before the labeled window even began. None if this turbine has
    no fault window (a normal event) or the model never flagged anything.
    """
    if event_start is None:
        return None
    flagged_times = test_timestamps[y_pred == 1]
    if flagged_times.empty:
        return None
    first_flag_time = flagged_times.min()
    return (event_start - first_flag_time).total_seconds() / 60


def run_sweep(prepared, contamination):
    per_turbine = []
    all_true = []
    all_pred = []

    for name, X_train, X_test, y_test, test_timestamps, event_start, is_anomaly_event in prepared:
        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train)
        y_pred = (model.predict(X_test) == -1).astype(int)

        metrics = evaluate(y_test, y_pred)
        metrics["turbine_file"] = name
        metrics["lead_time_minutes"] = compute_lead_time_minutes(y_pred, test_timestamps, event_start)
        per_turbine.append(metrics)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    overall = evaluate(pd.Series(all_true), pd.Series(all_pred))
    per_turbine_df = pd.DataFrame(per_turbine)

    # Early-warning summary: among turbines with a real fault window, how many
    # got flagged before event_start at all, and how much lead time did that buy.
    anomaly_turbines = per_turbine_df[per_turbine_df["lead_time_minutes"].notna() | (per_turbine_df["n_anomalies"] > 0)]
    early_warnings = per_turbine_df[per_turbine_df["lead_time_minutes"] > 0]
    overall["anomaly_turbines"] = len(anomaly_turbines)
    overall["early_warning_turbines"] = len(early_warnings)
    overall["median_lead_time_minutes"] = (
        early_warnings["lead_time_minutes"].median() if len(early_warnings) else None
    )

    return overall, per_turbine_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(EVENT_INFO_PATH, sep=";")
    events["event_start"] = pd.to_datetime(events["event_start"])
    events["event_end"] = pd.to_datetime(events["event_end"])
    events = events.set_index("event_id")

    csv_paths = sorted(CLEANED_DATA_DIR.glob("*_cleaned.csv"))

    temperature_avg_cols = None
    extra_cols = ()
    if FEATURE_SET == "temperature":
        temperature_avg_cols = get_temperature_avg_columns(FEATURE_DESCRIPTION_PATH)
        print(f"Restricting to {len(temperature_avg_cols)} temperature sensors: {temperature_avg_cols}")
    elif FEATURE_SET == "temperature_wind_gust":
        temperature_avg_cols = get_temperature_avg_columns(FEATURE_DESCRIPTION_PATH)
        # wind_speed_3_std = how much wind speed varied within each 10-minute
        # window, i.e. a gustiness/turbulence proxy - not used in any run so far.
        extra_cols = ("wind_speed_3_std",)
        print(
            f"Restricting to {len(temperature_avg_cols)} temperature sensors "
            f"plus wind gustiness ({extra_cols})"
        )
    elif FEATURE_SET == "temperature_mech":
        temperature_avg_cols = get_temperature_avg_columns(FEATURE_DESCRIPTION_PATH)
        # Pitch angle (average + within-window std) as a proxy for erratic
        # hydraulic pitch control behaviour, on top of gear_ratio_deviation
        # (added in load_and_label_turbine) as a gearbox-slip proxy.
        extra_cols = (PITCH_AVG_COL, PITCH_STD_COL)
        print(
            f"Restricting to {len(temperature_avg_cols)} temperature sensors "
            f"plus pitch behaviour ({extra_cols}) and gear_ratio_deviation"
        )

    print("Loading and feature-engineering all turbines once...")
    prepared = prepare_turbine_data(csv_paths, events, temperature_avg_cols, extra_cols)
    print(f"Using {prepared[0][1].shape[1]} features per turbine.")

    sweep_results = []
    for contamination in CONTAMINATION_VALUES:
        print(f"\n=== contamination={contamination} ===")
        overall, per_turbine_df = run_sweep(prepared, contamination)
        for key, value in overall.items():
            print(f"  {key}: {value}")
        overall["contamination"] = contamination
        sweep_results.append(overall)

        per_turbine_df = per_turbine_df[
            ["turbine_file", "n_rows", "n_anomalies", "accuracy", "precision", "recall", "f1", "f2", "lead_time_minutes"]
        ]
        per_turbine_df.to_csv(
            OUTPUT_DIR / f"isolation_forest_per_turbine_contamination_{contamination}.csv", index=False
        )

    # F2 (recall weighted 4x precision, see FBETA_WEIGHT) is the main criterion
    # for picking a contamination value - a missed fault matters more than a
    # false alarm here (see the F2 discussion in the journal).
    sweep_df = pd.DataFrame(sweep_results)[
        [
            "contamination", "f2", "precision", "recall", "f1", "accuracy",
            "anomaly_turbines", "early_warning_turbines", "median_lead_time_minutes",
            "n_rows", "n_anomalies",
        ]
    ].sort_values("f2", ascending=False)
    print("\nPrecision/recall trade-off across contamination values (sorted by F2, the main criterion):")
    print(sweep_df)
    sweep_df.to_csv(OUTPUT_DIR / "isolation_forest_contamination_sweep.csv", index=False)

    best = sweep_df.iloc[0]
    print(
        f"\nBest by F2: contamination={best['contamination']} -> F2={best['f2']:.3f} "
        f"(precision={best['precision']:.3f}, recall={best['recall']:.3f}), "
        f"{int(best['early_warning_turbines'])}/{int(best['anomaly_turbines'])} fault turbines flagged "
        f"before event_start (median lead time: {best['median_lead_time_minutes']:.0f} min)"
        if pd.notna(best["median_lead_time_minutes"])
        else f"\nBest by F2: contamination={best['contamination']} -> F2={best['f2']:.3f}"
    )


if __name__ == "__main__":
    main()
