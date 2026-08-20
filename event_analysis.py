# 02_event_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

cleaned_folder = "data_cleaning/cleaned_data"
event_info_path = "Wind Farm A/event_info.csv"
output_folder = "analysis_output"
os.makedirs(output_folder, exist_ok=True)

PRE_EVENT_WINDOW = pd.Timedelta(hours=48)
STATUS_COL = "status_type_id"
TIME_COL = "time_stamp"
ASSET_COL = "asset_id"


def load_event_info(path):
    with open(path, "r") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]
    df["event_start"] = pd.to_datetime(df["event_start"], errors="coerce")
    df["event_end"] = pd.to_datetime(df["event_end"], errors="coerce")
    return df


def load_all_cleaned_files(folder):
    """Load every cleaned CSV, tag it with its event_id (from filename) and asset_id (from data)."""
    files = glob.glob(os.path.join(folder, "*.csv"))
    data = {}
    for file_path in files:
        event_id = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, "r") as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","
        df = pd.read_csv(file_path, sep=sep)
        if TIME_COL in df.columns:
            df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        data[event_id] = df
    return data


def group_files_by_turbine(data, event_info):
    """Return dict: asset_id -> list of file entries (event_id, df, label, event_start, event_end)."""
    groups = {}
    for event_id, df in data.items():
        if ASSET_COL not in df.columns:
            continue
        asset_id = df[ASSET_COL].iloc[0]

        match = event_info[event_info["event_id"].astype(str) == str(event_id)]
        if match.empty:
            continue

        groups.setdefault(asset_id, []).append({
            "event_id": event_id,
            "df": df,
            "label": match["event_label"].values[0],
            "event_start": match["event_start"].values[0],
            "event_end": match["event_end"].values[0]
        })
    return groups


def get_pre_event_data(df, event_start, window=PRE_EVENT_WINDOW):
    event_start = pd.to_datetime(event_start)
    return df[(df[TIME_COL] >= event_start - window) & (df[TIME_COL] < event_start)]


def get_active_sensor_columns(df):
    """Exclude constant columns (no information) from comparisons."""
    candidate_cols = [c for c in df.columns if c not in [TIME_COL, ASSET_COL, STATUS_COL, "id", "train_test"]]
    return [c for c in candidate_cols if df[c].nunique(dropna=True) > 1]


def resample_status_hourly(df):
    """Aggregates status_type_id to hourly resolution (most frequent status per hour)."""
    df = df.set_index(TIME_COL)
    hourly_status = df[STATUS_COL].resample("1h").agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    return hourly_status.dropna().reset_index()


def plot_turbine_overview(asset_id, anomaly_entries, normal_entries, output_folder,
                           plot_window_before=pd.Timedelta(days=14),
                           plot_window_after=pd.Timedelta(days=2)):
    """
    Creates ONE combined figure per turbine: one subplot per anomaly
    event (status timeline around that event), plus one extra subplot
    showing a normal-condition reference timeline for visual comparison.
    """
    n_rows = len(anomaly_entries) + 1

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.2 * n_rows))
    if n_rows == 1:
        axes = [axes]

    hourly_cache = {}
    all_statuses = set()
    for entry in anomaly_entries:
        hourly_df = resample_status_hourly(entry["df"])
        hourly_cache[entry["event_id"]] = hourly_df
        all_statuses.update(hourly_df[STATUS_COL].unique())

    normal_hourly = resample_status_hourly(normal_entries[0]["df"])
    all_statuses.update(normal_hourly[STATUS_COL].unique())

    all_statuses = sorted(all_statuses)
    cmap = plt.get_cmap("tab10")
    color_map = {s: cmap(i % 10) for i, s in enumerate(all_statuses)}

    # one subplot per anomaly event
    for ax, entry in zip(axes[:-1], anomaly_entries):
        hourly_df = hourly_cache[entry["event_id"]]
        event_start_dt = pd.to_datetime(entry["event_start"])
        event_end_dt = pd.to_datetime(entry["event_end"]) if pd.notna(entry["event_end"]) else event_start_dt

        window_start = event_start_dt - plot_window_before
        window_end = event_end_dt + plot_window_after
        plot_df = hourly_df[(hourly_df[TIME_COL] >= window_start) & (hourly_df[TIME_COL] <= window_end)]

        for _, row in plot_df.iterrows():
            ax.axvspan(row[TIME_COL], row[TIME_COL] + pd.Timedelta(hours=1), color=color_map[row[STATUS_COL]])

        ax.axvspan(event_start_dt, event_end_dt, color="black", alpha=0.15)
        ax.axvline(event_start_dt, color="black", linestyle="--", linewidth=1)
        ax.axvline(event_end_dt, color="black", linestyle="--", linewidth=1)
        ax.set_yticks([])
        ax.set_title(f"Event {entry['event_id']}", fontsize=9, loc="left")

    # last subplot: normal reference
    ax_normal = axes[-1]
    window_length = plot_window_before + plot_window_after
    normal_start = normal_hourly[TIME_COL].min()
    normal_plot_df = normal_hourly[normal_hourly[TIME_COL] <= normal_start + window_length]

    for _, row in normal_plot_df.iterrows():
        ax_normal.axvspan(row[TIME_COL], row[TIME_COL] + pd.Timedelta(hours=1), color=color_map[row[STATUS_COL]])

    ax_normal.set_yticks([])
    ax_normal.set_title("Normal reference", fontsize=9, loc="left")
    ax_normal.set_xlabel("Time")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[s]) for s in all_statuses]
    labels = [f"Status {s}" for s in all_statuses]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))

    fig.suptitle(f"Turbine {asset_id} — All events vs. normal reference", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(output_folder, f"overview_asset{asset_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_turbine_status_table(asset_id, anomaly_entries, normal_df, output_folder):
    """
    Builds ONE combined table per turbine: rows are status_type_id values,
    columns are the normal baseline share plus the pre-event share for
    EVERY anomaly event of that turbine, side by side.
    """
    normal_dist = normal_df[STATUS_COL].value_counts(normalize=True).sort_index()
    table = pd.DataFrame({"normal_share": normal_dist})

    for entry in anomaly_entries:
        pre_event_df = get_pre_event_data(entry["df"], entry["event_start"])
        if pre_event_df.empty:
            continue
        dist = pre_event_df[STATUS_COL].value_counts(normalize=True).sort_index()
        table[f"event_{entry['event_id']}_pre_event_share"] = dist

    table = table.fillna(0)

    out_path = os.path.join(output_folder, f"status_table_asset{asset_id}.csv")
    table.to_csv(out_path, sep=";")
    return out_path


def analyze_turbine(asset_id, entries, output_folder):
    normal_entries = [e for e in entries if str(e["label"]).lower() in ["normal", "0"]]
    anomaly_entries = [e for e in entries if str(e["label"]).lower() not in ["normal", "0"]]

    if not normal_entries or not anomaly_entries:
        return None

    normal_df = pd.concat([e["df"] for e in normal_entries], ignore_index=True)

    plot_path = plot_turbine_overview(asset_id, anomaly_entries, normal_entries, output_folder)
    table_path = build_turbine_status_table(asset_id, anomaly_entries, normal_df, output_folder)

    return plot_path, table_path


if __name__ == "__main__":
    event_info = load_event_info(event_info_path)
    data = load_all_cleaned_files(cleaned_folder)
    groups = group_files_by_turbine(data, event_info)

    processed = 0
    skipped = 0

    for asset_id, entries in groups.items():
        result = analyze_turbine(asset_id, entries, output_folder)
        if result is None:
            print(f"Turbine {asset_id}: skipped (no normal/anomaly pair)")
            skipped += 1
        else:
            plot_path, table_path = result
            print(f"Turbine {asset_id}: done -> {os.path.basename(plot_path)}, {os.path.basename(table_path)}")
            processed += 1

    print(f"\nDone. Turbines processed: {processed}, skipped (no normal/anomaly pair): {skipped}")
    print(f"Output saved in: {output_folder}")