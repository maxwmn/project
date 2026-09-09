"""
plot_power_curve.py

Visualizes the `power_curve_deviation` feature added in
anomaly_detection_isolation_forest.py: for a few example turbines, plots
wind speed vs. actual power output, colored by whether each point falls
inside a labeled fault window, with the turbine's own expected power curve
(fit on its historical "train" rows only) overlaid. Points that sit well
below the curve during a fault window are exactly what the
power_curve_deviation feature captures numerically.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

from anomaly_detection_isolation_forest import (
    CLEANED_DATA_DIR,
    EVENT_INFO_PATH,
    WIND_SPEED_COL,
    POWER_COL,
    SPLIT_COL,
    load_and_label_turbine,
)

OUTPUT_DIR = Path("analysis_output")

# A mix of real fault events (different root causes) plus one normal turbine for contrast.
EXAMPLE_TURBINES = [
    ("0_cleaned.csv", "Generator bearing failure"),
    ("68_cleaned.csv", "Transformer failure"),
    ("45_cleaned.csv", "Hydraulic group"),
    ("13_cleaned.csv", "Normal (no fault)"),
]

# Okabe-Ito colorblind-safe pair, plus a distinct marker shape per series so
# identity never relies on color alone.
NORMAL_COLOR = "#0072B2"
ANOMALY_COLOR = "#D55E00"
CURVE_COLOR = "#444444"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    events = pd.read_csv(EVENT_INFO_PATH, sep=";")
    events["event_start"] = pd.to_datetime(events["event_start"])
    events["event_end"] = pd.to_datetime(events["event_end"])
    events = events.set_index("event_id")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, (filename, description) in zip(axes, EXAMPLE_TURBINES):
        df = load_and_label_turbine(CLEANED_DATA_DIR / filename, events)
        test_df = df[df[SPLIT_COL] == "prediction"]
        train_df = df[df[SPLIT_COL] == "train"]

        normal_rows = test_df[test_df["true_anomaly"] == 0]
        anomaly_rows = test_df[test_df["true_anomaly"] == 1]

        ax.scatter(
            normal_rows[WIND_SPEED_COL],
            normal_rows[POWER_COL],
            s=10,
            alpha=0.35,
            color=NORMAL_COLOR,
            marker="o",
            linewidths=0,
            label="Normal",
            zorder=2,
        )
        ax.scatter(
            anomaly_rows[WIND_SPEED_COL],
            anomaly_rows[POWER_COL],
            s=22,
            alpha=0.85,
            color=ANOMALY_COLOR,
            marker="x",
            label="Anomaly (fault window)",
            zorder=3,
        )

        # Redraw the same expected power curve used as the feature, for reference.
        curve = IsotonicRegression(out_of_bounds="clip", increasing=True)
        curve.fit(train_df[WIND_SPEED_COL], train_df[POWER_COL])
        x_line = pd.Series(sorted(df[WIND_SPEED_COL].dropna().unique()))
        ax.plot(
            x_line,
            curve.predict(x_line),
            color=CURVE_COLOR,
            linewidth=2,
            label="Expected power (from train)",
            zorder=4,
        )

        ax.set_title(f"{filename} — {description}", fontsize=11)
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Grid power (kW)")
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Power curve deviation: actual vs. expected power output", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    out_path = OUTPUT_DIR / "power_curve_deviation_examples.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
