"""
anomaly_detection_isolation_forest.py

Unsupervised anomaly detection: for each wind turbine individually, fits an
Isolation Forest on that turbine's own historical "train" rows (assumed
normal operation), then scores its "prediction" rows for how unusual they
are - no anomaly labels are used for training, only for evaluation
afterwards (event_info.csv fault windows).

Closer to the CARE-to-Compare benchmark's own per-file train/prediction
design than the supervised turbine-holdout approach in
archive/anomaly_detection*.py (superseded, kept for reference): each turbine
gets its own model, so there's no need for one shared model to generalize
fault signatures across turbines with different sensor baselines.

See anomaly_detection_journal.md for the full rationale behind each design
decision below (why per-turbine models, why these features, why Fbeta, ...).
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 30)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)

CLEANED_DATA_DIR = Path("data_cleaning/cleaned_data")
EVENT_INFO_PATH = Path("Wind Farm A/event_info.csv")
FEATURE_DESCRIPTION_PATH = Path("Wind Farm A/feature_description.csv")
OUTPUT_DIR = Path("isolation_forest_output")

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

# Which sensors the model sees. "all" = every sensor; the others restrict to
# temperature sensors (consistently the top features in prior supervised
# runs) plus the given extra columns, to cut noise from Isolation Forest's
# random axis selection.
FEATURE_SET = "temperature"  # "all", "temperature", "temperature_wind_gust", or "temperature_mech"
FEATURE_SET_EXTRA_COLS = {
    "temperature": (),
    "temperature_wind_gust": ("wind_speed_3_std",),  # gustiness/turbulence proxy
    "temperature_mech": (PITCH_AVG_COL, PITCH_STD_COL),  # erratic pitch control proxy
}

# Standardize each turbine's features to its own z-score (fit on its train
# rows only) before fitting, so raw numeric range doesn't bias feature choice.
Z_SCORE_NORMALIZE = True

# Isolation Forest's expected-anomaly-rate knob. "auto" derives a fixed
# generic threshold from the isolation-score formula; the numeric values
# instead calibrate the threshold from each turbine's own train-score
# distribution, trading precision for recall as the value increases.
CONTAMINATION_VALUES = ["auto", 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# beta=sqrt(2) (beta^2=2) weights recall 2x as much as precision: a missed
# fault (turbine keeps running) is worse than an unnecessary false alarm.
FBETA_WEIGHT = 2**0.5

# How many of each turbine's most extreme features to report per alarm row.
TOP_N_ALARM_FEATURES = 3

# How many features to keep in the overall (pooled-across-turbines) feature
# importance ranking used for the results visualization.
TOP_N_GLOBAL_FEATURES = 15

# Statuses considered "normal operation" per Wind Farm A/README.md (0 and 2;
# the others - derated, service, downtime, other - are already flagged by
# the SCADA system itself, so the CARE-score excludes them from Coverage and
# Accuracy to avoid trivially inflating those sub-scores).
CARE_NORMAL_STATUS_IDS = (0, 2)

# beta=0.5 weights precision over recall, per the CARE-score paper (Gueck et
# al.), to penalize excessive false positives more than Fbeta above does.
CARE_BETA = 0.5

# Net consecutive alarm rows (on normal-status timestamps) needed before an
# event counts as "detected" for the Reliability sub-score - 72 = 12h at the
# 10-minute sampling rate, the value the paper found worked across its 95
# datasets.
CARE_CRITICALITY_THRESHOLD = 72

# Sub-score weights for the final CARE score (formula 4 in the paper):
# Accuracy counts double so normal-only turbines matter as much as
# anomaly-event turbines overall.
CARE_SUBSCORE_WEIGHTS = {"coverage": 1, "earliness": 1, "reliability": 1, "accuracy": 2}


class TurbineData(NamedTuple):
    name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    test_timestamps: pd.Series
    status_test: pd.Series
    event_start: "pd.Timestamp | None"  # None for turbines with no fault window


def get_temperature_avg_columns(feature_description_path):
    """Return the "_avg" columns for every sensor described as "temperature"
    in feature_description.csv (not hand-picked, so it tracks that file)."""
    descriptions = pd.read_csv(feature_description_path, sep=";")
    is_temperature = descriptions["description"].str.contains("temperature", case=False, na=False)
    return [f"{name}_avg" for name in descriptions.loc[is_temperature, "sensor_name"]]


def add_power_curve_deviation(df):
    """Add `power_curve_deviation`: actual power minus the power an isotonic
    fit (wind speed -> power, fit on train rows) would expect at that wind
    speed - separates "too little wind" from "too little power for the wind"."""
    train_mask = df[SPLIT_COL] == "train"
    curve = IsotonicRegression(out_of_bounds="clip", increasing=True)
    curve.fit(df.loc[train_mask, WIND_SPEED_COL], df.loc[train_mask, POWER_COL])
    df["power_curve_deviation"] = (df[POWER_COL] - curve.predict(df[WIND_SPEED_COL])).astype("float32")
    return df


def add_gear_ratio_deviation(df):
    """Add `gear_ratio_deviation`: actual generator speed minus the speed an
    isotonic fit (rotor rpm -> generator rpm, fit on train rows) would expect
    - gearbox slip/wear shows up as a deviation from the fixed gear ratio."""
    train_mask = df[SPLIT_COL] == "train"
    curve = IsotonicRegression(out_of_bounds="clip", increasing=True)
    curve.fit(df.loc[train_mask, ROTOR_RPM_COL], df.loc[train_mask, GENERATOR_RPM_COL])
    df["gear_ratio_deviation"] = (df[GENERATOR_RPM_COL] - curve.predict(df[ROTOR_RPM_COL])).astype("float32")
    return df


def add_trend_features(df, feature_cols):
    """Add, per sensor, deviation from the trailing 24h average and the
    slope over the trailing 6 hours - real precursors tend to drift rather
    than show up as a single unusual moment."""
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    for col in feature_cols:
        rolling_mean_24h = df[col].rolling(TREND_WINDOW_24H, min_periods=1).mean()
        df[f"{col}_dev24h"] = (df[col] - rolling_mean_24h).astype("float32")
        df[f"{col}_slope6h"] = (df[col].diff(TREND_WINDOW_SLOPE_HOURS) / TREND_WINDOW_SLOPE_HOURS).astype("float32")
    return df


def load_and_label_turbine(csv_path, events):
    """Load one turbine's cleaned CSV, add engineered/trend features, and
    label rows from the fault window in event_info.csv (dropping the
    ambiguous post-repair rows for anomaly events; status_type_id is not
    used for labeling, see archive/anomaly_detection.py's docstring)."""
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

    event = events.loc[int(csv_path.name.split("_")[0])]
    is_anomaly_event = event["event_label"] == "anomaly"
    in_window = (df[TIME_COL] >= event["event_start"]) & (df[TIME_COL] <= event["event_end"])
    df["true_anomaly"] = int(is_anomaly_event) * in_window.astype(int)

    if is_anomaly_event:
        df = df[~(df[TIME_COL] > event["event_end"])]

    return df


def evaluate(y_true, y_pred):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    n_anomalies = int(y_true.sum())

    if n_anomalies == 0:
        # No real fault window in this slice: there is nothing to recall, so
        # recall is trivially perfect, and precision is perfect too unless the
        # model raised any false alarm - a quiet turbine should score as well
        # as a turbine with a correctly caught anomaly, not as a failure.
        precision = 1.0 if y_pred.sum() == 0 else 0.0
        recall = 1.0
        f1 = fbeta = precision
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fbeta = fbeta_score(y_true, y_pred, beta=FBETA_WEIGHT, zero_division=0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fbeta": fbeta,
        "n_rows": len(y_true),
        "n_anomalies": n_anomalies,
    }


def prepare_turbine_data(csv_paths, events, temperature_avg_cols=None, extra_cols=()):
    """Load, feature-engineer, and median-fill every turbine once into a
    list of TurbineData, so a model can be refit per contamination value
    without re-reading the CSVs each time.

    If `temperature_avg_cols` is given, features are restricted to those
    columns, `power_curve_deviation`, `gear_ratio_deviation`, `extra_cols`,
    and the `_dev24h`/`_slope6h` trend variants of any "_avg" column among
    them.
    """
    prepared = []
    for csv_path in csv_paths:
        df = load_and_label_turbine(csv_path, events)
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS + ["true_anomaly"]]

        if temperature_avg_cols is not None:
            base_avg_cols = set(temperature_avg_cols) | {c for c in extra_cols if c.endswith("_avg")}
            allowed = base_avg_cols | {"power_curve_deviation", "gear_ratio_deviation"} | set(extra_cols)
            feature_cols = [c for c in feature_cols if c in allowed or c.rsplit("_", 1)[0] in base_avg_cols]

        train_df = df[df[SPLIT_COL] == "train"]
        test_df = df[df[SPLIT_COL] == "prediction"]

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        if Z_SCORE_NORMALIZE:
            # Fit on this turbine's own train rows only, so no test/prediction
            # information leaks into the scaling.
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

        event = events.loc[int(csv_path.name.split("_")[0])]
        event_start = event["event_start"] if event["event_label"] == "anomaly" else None

        prepared.append(TurbineData(
            csv_path.name, X_train, X_test, test_df["true_anomaly"],
            test_df[TIME_COL].reset_index(drop=True),
            test_df[STATUS_COL].reset_index(drop=True), event_start,
        ))
    return prepared


def compute_lead_time_minutes(y_pred, test_timestamps, event_start):
    """Minutes between the model's first flagged anomaly and the official
    fault window start (event_start): positive = flagged before the window
    even began. None if there's no fault window or nothing was flagged."""
    if event_start is None:
        return None
    flagged_times = test_timestamps[y_pred == 1]
    if flagged_times.empty:
        return None
    return (event_start - flagged_times.min()).total_seconds() / 60


def care_masked_confusion(status, y_true, y_pred):
    """tp/fp/fn/tn restricted to rows with a normal status-ID
    (CARE_NORMAL_STATUS_IDS), as required by the CARE-score paper's Coverage
    and Accuracy sub-scores."""
    mask = np.isin(np.asarray(status), CARE_NORMAL_STATUS_IDS)
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def care_fbeta(tp, fp, fn, beta=CARE_BETA):
    """Fbeta from raw counts (paper formula 1) - reused for both the
    row-level Coverage sub-score and the event-level Reliability sub-score."""
    denom = (1 + beta**2) * tp + beta**2 * fn + fp
    return (1 + beta**2) * tp / denom if denom else 0.0


def care_event_detected(status, y_pred, threshold=CARE_CRITICALITY_THRESHOLD):
    """Turn a row-level prediction sequence into a single event-level
    detected/not-detected label (paper's criticality Algorithm 1): a
    running counter rises by 1 on each predicted-anomaly row with a normal
    status, decays by 1 (floored at 0) on each predicted-normal row with a
    normal status, and holds steady on abnormal-status rows (already
    flagged by SCADA on its own). Reaching `threshold` anywhere along the
    way counts the event as detected - an isolated false positive decays
    away, but a sustained run of alarms does not."""
    is_normal_status = np.isin(np.asarray(status), CARE_NORMAL_STATUS_IDS)
    y_pred = np.asarray(y_pred)
    criticality = 0
    max_criticality = 0
    for normal, pred in zip(is_normal_status, y_pred):
        if normal:
            criticality = criticality + 1 if pred == 1 else max(criticality - 1, 0)
            max_criticality = max(max_criticality, criticality)
    return max_criticality >= threshold


def care_earliness_weighted_score(y_pred_in_event):
    """Earliness sub-score (WS) for one anomaly event: each in-event
    timestamp is weighted 1.0 in the first half of the event, decreasing
    linearly to 0.0 by the end, and the score is the weighted share of
    timestamps flagged as an anomaly - rewarding a model that raises the
    alarm early over one that only catches the tail of the fault."""
    y_pred_in_event = np.asarray(y_pred_in_event)
    n = len(y_pred_in_event)
    relative_position = (np.arange(n) + 0.5) / n  # midpoint of each step, in [0, 1]
    weights = np.clip(2 * (1 - relative_position), 0, 1)
    return float((weights * y_pred_in_event).sum() / weights.sum())


def compute_care_score(coverage_scores, earliness_scores, accuracy_scores, event_true, event_pred, coverage_tp_total):
    """Combine the four CARE sub-scores (Gueck et al., "CARE to Compare",
    section 4.1) into the final CARE score, for comparison against the
    plain Fbeta/precision/recall sweep above. Unlike Fbeta, CARE separately
    rewards early detection (earliness), correctly staying quiet on
    normal-only turbines (accuracy), and not raising a sustained false
    alarm event on any turbine (reliability), rather than pooling every row
    across every turbine into one confusion matrix."""
    coverage = float(np.mean(coverage_scores)) if coverage_scores else float("nan")
    earliness = float(np.mean(earliness_scores)) if earliness_scores else float("nan")
    accuracy = float(np.mean(accuracy_scores)) if accuracy_scores else float("nan")

    event_true = np.asarray(event_true)
    event_pred = np.asarray(event_pred)
    event_tp = int(((event_true == 1) & (event_pred == 1)).sum())
    event_fp = int(((event_true == 0) & (event_pred == 1)).sum())
    event_fn = int(((event_true == 1) & (event_pred == 0)).sum())
    reliability = care_fbeta(event_tp, event_fp, event_fn)

    if coverage_tp_total == 0:
        # No real anomaly was ever correctly flagged: matches the paper's
        # "all normal" trivial-strategy special case.
        care = 0.0
    elif accuracy < 0.5:
        # Worse at recognizing normal behavior than a coin flip: the paper
        # caps the score at Accuracy regardless of the other sub-scores.
        care = accuracy
    else:
        w = CARE_SUBSCORE_WEIGHTS
        care = (
            w["coverage"] * coverage + w["earliness"] * earliness
            + w["reliability"] * reliability + w["accuracy"] * accuracy
        ) / sum(w.values())

    return {
        "care_score": care,
        "care_coverage_fbeta": coverage,
        "care_earliness_ws": earliness,
        "care_reliability_efbeta": reliability,
        "care_accuracy": accuracy,
    }


def fit_predict(X_train, X_test, contamination):
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    return (model.predict(X_test) == -1).astype(int)


def compute_alarm_rows(prepared, contamination):
    """For one contamination setting, refit each turbine's model and return
    its flagged alarm rows (the X_test rows predicted as anomalies), keyed
    by turbine file name - the shared input for explain_alarm_features and
    aggregate_alarm_feature_importance, so each turbine is only fit once."""
    return {t.name: t.X_test[fit_predict(t.X_train, t.X_test, contamination) == 1] for t in prepared}


def explain_alarm_features(alarm_rows_by_turbine, top_n=TOP_N_ALARM_FEATURES):
    """Per turbine, report the top_n features with the most extreme mean
    |z-score| among its flagged alarm rows.

    Individual trees split on a random feature/threshold each, so no single
    tree explains an alarm - it only exists in the aggregate path length.
    Averaging each feature's signed z-score across a turbine's alarm rows
    (relies on Z_SCORE_NORMALIZE) instead gives a direct, comparable answer
    to which features were unusually high/low when the alarm fired.
    """
    if not Z_SCORE_NORMALIZE:
        raise ValueError("explain_alarm_features requires Z_SCORE_NORMALIZE=True")

    rows = []
    for name, alarm_rows in alarm_rows_by_turbine.items():
        row = {"turbine_file": name, "n_alarms": len(alarm_rows)}

        if len(alarm_rows):
            ranked_features = alarm_rows.abs().mean().sort_values(ascending=False)
            for rank, feature in enumerate(ranked_features.index[:top_n]):
                row[f"top_feature_{rank + 1}"] = f"{feature} ({alarm_rows[feature].mean():+.2f}σ)"

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_alarm_feature_importance(alarm_rows_by_turbine, top_n=TOP_N_GLOBAL_FEATURES):
    """Pool every turbine's alarm rows together and rank features by mean
    |z-score| across all of them - the single most influential features
    driving alarms overall, as opposed to explain_alarm_features's
    per-turbine top_n breakdown."""
    if not Z_SCORE_NORMALIZE:
        raise ValueError("aggregate_alarm_feature_importance requires Z_SCORE_NORMALIZE=True")

    non_empty = [rows for rows in alarm_rows_by_turbine.values() if len(rows)]
    if not non_empty:
        return pd.Series(dtype=float, name="mean_abs_zscore")

    pooled = pd.concat(non_empty, ignore_index=True)
    importance = pooled.abs().mean().sort_values(ascending=False).head(top_n)
    importance.name = "mean_abs_zscore"
    return importance


def run_sweep(prepared, contamination):
    per_turbine = []
    all_true = []
    all_pred = []

    coverage_scores = []
    earliness_scores = []
    accuracy_scores = []
    event_true = []
    event_pred = []
    coverage_tp_total = 0

    for t in prepared:
        y_pred = fit_predict(t.X_train, t.X_test, contamination)

        metrics = evaluate(t.y_test, y_pred)
        metrics["turbine_file"] = t.name
        metrics["lead_time_minutes"] = compute_lead_time_minutes(y_pred, t.test_timestamps, t.event_start)
        per_turbine.append(metrics)

        all_true.extend(t.y_test.tolist())
        all_pred.extend(y_pred.tolist())

        is_anomaly_turbine = t.event_start is not None
        tp, fp, fn, tn = care_masked_confusion(t.status_test, t.y_test, y_pred)
        if is_anomaly_turbine:
            coverage_scores.append(care_fbeta(tp, fp, fn))
            in_event_pred = y_pred[t.y_test.to_numpy() == 1]
            if len(in_event_pred):
                earliness_scores.append(care_earliness_weighted_score(in_event_pred))
            coverage_tp_total += tp
        else:
            accuracy_scores.append(tn / (fp + tn) if (fp + tn) else float("nan"))

        event_true.append(int(is_anomaly_turbine))
        event_pred.append(int(care_event_detected(t.status_test, y_pred)))

    overall = evaluate(pd.Series(all_true), pd.Series(all_pred))
    per_turbine_df = pd.DataFrame(per_turbine)

    # Early-warning summary: among turbines with a real fault window, how many
    # got flagged before event_start at all, and how much lead time did that buy.
    anomaly_turbines = per_turbine_df[per_turbine_df["lead_time_minutes"].notna() | (per_turbine_df["n_anomalies"] > 0)]
    early_warnings = per_turbine_df[per_turbine_df["lead_time_minutes"] > 0]
    overall["anomaly_turbines"] = len(anomaly_turbines)
    overall["early_warning_turbines"] = len(early_warnings)
    overall["median_lead_time_minutes"] = early_warnings["lead_time_minutes"].median() if len(early_warnings) else None

    overall.update(compute_care_score(
        coverage_scores, earliness_scores, accuracy_scores, event_true, event_pred, coverage_tp_total
    ))

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
    if FEATURE_SET != "all":
        temperature_avg_cols = get_temperature_avg_columns(FEATURE_DESCRIPTION_PATH)
        extra_cols = FEATURE_SET_EXTRA_COLS[FEATURE_SET]
        extra_note = f" plus {extra_cols}" if extra_cols else ""
        print(f"Restricting to {len(temperature_avg_cols)} temperature sensors{extra_note}")

    print("Loading and feature-engineering all turbines once...")
    prepared = prepare_turbine_data(csv_paths, events, temperature_avg_cols, extra_cols)
    print(f"Using {prepared[0].X_train.shape[1]} features per turbine.")

    sweep_results = []
    for contamination in CONTAMINATION_VALUES:
        print(f"\n=== contamination={contamination} ===")
        overall, per_turbine_df = run_sweep(prepared, contamination)
        for key, value in overall.items():
            print(f"  {key}: {value}")
        overall["contamination"] = contamination
        sweep_results.append(overall)

        per_turbine_df = per_turbine_df[
            ["turbine_file", "n_rows", "n_anomalies", "accuracy", "precision", "recall", "f1", "fbeta", "lead_time_minutes"]
        ]
        per_turbine_df.to_csv(
            OUTPUT_DIR / f"isolation_forest_per_turbine_contamination_{contamination}.csv", index=False
        )

    # Fbeta (recall weighted 2x precision, see FBETA_WEIGHT) is the main criterion
    # for picking a contamination value - a missed fault matters more than a
    # false alarm here (see the Fbeta discussion in the journal). The CARE
    # score (see care_* functions above) is reported alongside it purely for
    # comparison, since it scores coverage, earliness, reliability, and
    # normal-behavior accuracy separately instead of pooling every row into
    # one confusion matrix.
    sweep_df = pd.DataFrame(sweep_results)[
        [
            "contamination", "fbeta", "precision", "recall", "f1", "accuracy",
            "care_score", "care_coverage_fbeta", "care_earliness_ws",
            "care_reliability_efbeta", "care_accuracy",
            "anomaly_turbines", "early_warning_turbines", "median_lead_time_minutes",
            "n_rows", "n_anomalies",
        ]
    ].sort_values("fbeta", ascending=False)
    print("\nPrecision/recall trade-off across contamination values (sorted by Fbeta, the main criterion):")
    print(sweep_df)
    sweep_df.to_csv(OUTPUT_DIR / "isolation_forest_contamination_sweep.csv", index=False)

    best = sweep_df.iloc[0]
    lead_time_note = (
        f", {int(best['early_warning_turbines'])}/{int(best['anomaly_turbines'])} fault turbines flagged "
        f"before event_start (median lead time: {best['median_lead_time_minutes']:.0f} min)"
        if pd.notna(best["median_lead_time_minutes"])
        else ""
    )
    print(
        f"\nBest by Fbeta: contamination={best['contamination']} -> Fbeta={best['fbeta']:.3f} "
        f"(precision={best['precision']:.3f}, recall={best['recall']:.3f}){lead_time_note}"
    )
    print(f"  Its CARE score: {best['care_score']:.3f} (coverage={best['care_coverage_fbeta']:.3f}, "
          f"earliness={best['care_earliness_ws']:.3f}, reliability={best['care_reliability_efbeta']:.3f}, "
          f"accuracy={best['care_accuracy']:.3f})")

    best_by_care = sweep_df.sort_values("care_score", ascending=False).iloc[0]
    if best_by_care["contamination"] != best["contamination"]:
        print(
            f"  Best by CARE instead: contamination={best_by_care['contamination']} -> "
            f"CARE={best_by_care['care_score']:.3f} (Fbeta={best_by_care['fbeta']:.3f})"
        )

    print(f"\nExplaining alarms for the best contamination ({best['contamination']})...")
    alarm_rows_by_turbine = compute_alarm_rows(prepared, best["contamination"])

    alarm_features_df = explain_alarm_features(alarm_rows_by_turbine)
    alarm_features_df.to_csv(
        OUTPUT_DIR / f"isolation_forest_alarm_feature_explanation_contamination_{best['contamination']}.csv",
        index=False,
    )
    print(
        f"\nTop {TOP_N_ALARM_FEATURES} features by mean |z-score| among each turbine's alarm rows "
        f"(contamination={best['contamination']}):"
    )
    print(alarm_features_df.to_string(index=False))

    feature_importance = aggregate_alarm_feature_importance(alarm_rows_by_turbine)
    feature_importance.to_csv(
        OUTPUT_DIR / f"isolation_forest_feature_importance_contamination_{best['contamination']}.csv",
        index_label="feature", header=True,
    )
    print(
        f"\nOverall top {TOP_N_GLOBAL_FEATURES} most influential features (mean |z-score| pooled "
        f"across all turbines' alarm rows, contamination={best['contamination']}):"
    )
    print(feature_importance.to_string())


if __name__ == "__main__":
    main()
