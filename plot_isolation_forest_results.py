"""
plot_isolation_forest_results.py

Visualizes the two headline outputs of anomaly_detection_isolation_forest.py's
contamination sweep, as two standalone figures: the Fbeta (F2,
beta=FBETA_WEIGHT) score across contamination values, with the best one
highlighted, and the overall most influential features driving alarms at
that best contamination (mean |z-score| pooled across all turbines' alarm
rows), labeled with their actual sensor descriptions from
feature_description.csv instead of the raw anonymized column names.

Run anomaly_detection_isolation_forest.py first - this script only reads the
CSVs it writes to isolation_forest_output/, it does not refit any models.
"""

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from anomaly_detection_isolation_forest import CONTAMINATION_VALUES, FBETA_WEIGHT, FEATURE_DESCRIPTION_PATH, OUTPUT_DIR

SWEEP_PATH = OUTPUT_DIR / "isolation_forest_contamination_sweep.csv"

# Okabe-Ito colorblind-safe pair: the best contamination value stands out
# against the rest of the sweep.
BEST_COLOR = "#D55E00"
OTHER_COLOR = "#0072B2"

# Engineered features that have no entry in feature_description.csv, given a
# readable label directly instead.
FEATURE_LABEL_OVERRIDES = {
    "power_curve_deviation": "Power curve deviation (actual vs. expected power)",
    "gear_ratio_deviation": "Gear ratio deviation (actual vs. expected generator speed)",
}

# Suffixes add_trend_features() appends to a sensor's "_avg" column name in
# anomaly_detection_isolation_forest.py - kept as a readable tag rather than
# dropped, since they change what the feature actually measures.
TREND_SUFFIX_LABELS = {
    "_dev24h": " (24h deviation)",
    "_slope6h": " (6h slope)",
}


def load_sensor_descriptions(feature_description_path):
    """Map each raw sensor_name (e.g. "sensor_14") to its human-readable
    description from feature_description.csv."""
    descriptions = pd.read_csv(feature_description_path, sep=";")
    return dict(zip(descriptions["sensor_name"], descriptions["description"]))


def describe_feature(feature, sensor_descriptions):
    """Turn a raw feature column name (e.g. "sensor_14_avg_dev24h") into a
    human-readable label, using the sensor's actual description instead of
    its anonymized code, with any trend-feature suffix kept as a tag."""
    if feature in FEATURE_LABEL_OVERRIDES:
        return FEATURE_LABEL_OVERRIDES[feature]

    base = feature
    suffix_label = ""
    for suffix, label in TREND_SUFFIX_LABELS.items():
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            suffix_label = label
            break

    sensor_name = base.rsplit("_", 1)[0] if base.rsplit("_", 1)[-1] in ("avg", "min", "max", "std") else base
    description = sensor_descriptions.get(sensor_name, feature)
    return f"{description}{suffix_label}"


def plot_fbeta_sweep(sweep_df, best_contamination, out_path):
    """Bar chart of Fbeta (F2) per contamination value, in the same order as
    CONTAMINATION_VALUES, with the best-scoring value picked out in color."""
    order = [str(c) for c in CONTAMINATION_VALUES]
    sweep_df = sweep_df.set_index("contamination").reindex(order).reset_index()
    colors = [BEST_COLOR if c == str(best_contamination) else OTHER_COLOR for c in sweep_df["contamination"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(sweep_df["contamination"], sweep_df["fbeta"], color=colors)
    for bar, value in zip(bars, sweep_df["fbeta"]):
        if pd.notna(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)

    ax.set_title(f"F2 score (Fbeta, β={FBETA_WEIGHT:.2f}) by contamination\nbest: contamination={best_contamination}", fontsize=12)
    ax.set_xlabel("contamination")
    ax.set_ylabel("Fbeta (F2) score")
    ax.set_ylim(0, 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_feature_importance(importance_df, sensor_descriptions, out_path):
    """Horizontal bar chart of the overall most influential features, most
    important at the top, labeled with their actual sensor descriptions."""
    importance_df = importance_df.sort_values("mean_abs_zscore")
    labels = [
        "\n".join(textwrap.wrap(describe_feature(feature, sensor_descriptions), width=42))
        for feature in importance_df["feature"]
    ]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(labels, importance_df["mean_abs_zscore"], color=OTHER_COLOR)
    ax.set_title("Most influential features in flagged alarms", fontsize=12)
    ax.set_xlabel("mean |z-score| across pooled alarm rows")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    if not SWEEP_PATH.exists():
        raise FileNotFoundError(f"{SWEEP_PATH} not found - run anomaly_detection_isolation_forest.py first.")
    sweep_df = pd.read_csv(SWEEP_PATH, dtype={"contamination": str})
    best_contamination = sweep_df.sort_values("fbeta", ascending=False).iloc[0]["contamination"]

    importance_path = OUTPUT_DIR / f"isolation_forest_feature_importance_contamination_{best_contamination}.csv"
    if not importance_path.exists():
        raise FileNotFoundError(f"{importance_path} not found - run anomaly_detection_isolation_forest.py first.")
    importance_df = pd.read_csv(importance_path)
    sensor_descriptions = load_sensor_descriptions(FEATURE_DESCRIPTION_PATH)

    plot_fbeta_sweep(sweep_df, best_contamination, OUTPUT_DIR / "isolation_forest_f2_score.png")
    plot_feature_importance(importance_df, sensor_descriptions, OUTPUT_DIR / "isolation_forest_feature_importance.png")


if __name__ == "__main__":
    main()
