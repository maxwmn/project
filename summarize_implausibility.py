import pandas as pd
import glob
import os

# --- Paths ---
cleaned_folder = "data_cleaning\\cleaned_data_31"
report_folder = "data_cleaning\\reports_31"
output_path = "data_cleaning\\reports_31\\implausibility_summary.csv"


def get_event_id(file_name):
    # Extracts the event ID from a report filename, e.g. "flags_3.csv.txt" -> 3
    name = file_name.replace("flags_", "").replace("replaced_values_", "")
    name = name.replace(".csv.txt", "")
    try:
        return int(name)
    except ValueError:
        return name


def get_row_count(event_id):
    # Reads the corresponding cleaned file just to get its number of rows,
    # used to express affected rows as a percentage
    cleaned_path = os.path.join(cleaned_folder, f"{event_id}.csv")
    if not os.path.exists(cleaned_path):
        return None
    df = pd.read_csv(cleaned_path, sep=";", decimal=",")
    return len(df)


def summarize_event(event_id):
    # Reads this event's flags and replaced-values report (if they exist)
    # and splits the affected sensors into "min" and "max" issues
    flags_path = os.path.join(report_folder, f"flags_{event_id}.csv.txt")
    replaced_path = os.path.join(report_folder, f"replaced_values_{event_id}.csv.txt")

    min_flagged = 0
    max_flagged = 0

    if os.path.exists(replaced_path):
        replaced = pd.read_csv(replaced_path, index_col=0)
        for sensor_col, count in replaced.iloc[:, 0].items():
            if sensor_col.endswith("_min"):
                min_flagged += count
            elif sensor_col.endswith("_max"):
                max_flagged += count

    total_rows = get_row_count(event_id)
    total_flagged = min_flagged + max_flagged
    pct_affected = (total_flagged / total_rows * 100) if total_rows else None

    return {
        "event_id": event_id,
        "total_rows": total_rows,
        "min_flagged": min_flagged,
        "max_flagged": max_flagged,
        "total_flagged": total_flagged,
        "pct_rows_affected": round(pct_affected, 3) if pct_affected is not None else None,
    }


if __name__ == "__main__":
    # Use the cleaned files as the master list of events, so that events
    # WITHOUT any implausibility report (i.e. no issues found) are still
    # included in the summary with a count of 0
    cleaned_files = glob.glob(os.path.join(cleaned_folder, "*.csv"))
    event_ids = [os.path.splitext(os.path.basename(f))[0] for f in cleaned_files]

    rows = [summarize_event(event_id) for event_id in event_ids]
    summary_df = pd.DataFrame(rows).sort_values("event_id")

    # Add a final row with totals/averages across all events
    totals = {
        "event_id": "TOTAL / AVERAGE",
        "total_rows": summary_df["total_rows"].sum(),
        "min_flagged": summary_df["min_flagged"].sum(),
        "max_flagged": summary_df["max_flagged"].sum(),
        "total_flagged": summary_df["total_flagged"].sum(),
        "pct_rows_affected": round(summary_df["pct_rows_affected"].mean(), 3),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)

    summary_df.to_csv(output_path, sep=";", index=False, decimal=",")
    print(f"Summary saved to: {output_path}")
    print(summary_df.to_string(index=False))