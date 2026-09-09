import pandas as pd
import glob
import os
import re
 
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
 
# --- Paths and settings ---
input_folder = "Wind Farm A\\datasets"
output_folder = "data_cleaning\\cleaned_data"
report_folder = "data_cleaning\\reports"
event_info_path = "Wind Farm A\\event_info.csv"
 
# Columns to remove entirely, regardless of what the implausibility check finds
# sensor_31 was removed because more than 75% of its values were implausible
# sensors 46 and 49 were removed because they were constant (no variation) across all events
columns_to_drop = ["sensor_31_avg", "sensor_31_min", "sensor_31_max", "sensor_46", "sensor_49"]



os.makedirs(output_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)
 
 
def clean_column_name(col):
    # Standardizes a column name: lowercase, no special characters, underscores instead of spaces
    col = col.strip().lower()
    col = re.sub(r"[^\w\s]", "", col)
    col = re.sub(r"\s+", "_", col)
    return col
 
 
def get_event_id(file_name):
    # Extracts the event ID from the filename (e.g. "3.csv" -> 3), used to match event_info.csv
    event_id = os.path.splitext(file_name)[0]
    try:
        return int(event_id)
    except ValueError:
        return event_id
 
 
def flag_and_replace_implausible_values(df, avg_cols, min_cols, max_cols):
    # For each sensor, flags rows where Min > Avg or Max < Avg (physically impossible),
    # and replaces only the implausible Min/Max value with NaN (Avg is left untouched,
    # since Avg values are documented as generally reliable)
    issue_flags = pd.DataFrame(index=df.index)
    replaced_count = {}
 
    for avg_col in avg_cols:
        prefix = re.sub(r"_avg$", "", avg_col, flags=re.IGNORECASE)
        matching_min = [c for c in min_cols if c == f"{prefix}_min"]
        matching_max = [c for c in max_cols if c == f"{prefix}_max"]
 
        if matching_min:
            min_col = matching_min[0]
            mask_min = df[min_col] > df[avg_col]
            issue_flags[f"{prefix}_min_gt_avg"] = mask_min
            replaced_count[min_col] = mask_min.sum()
            df.loc[mask_min, min_col] = pd.NA
 
        if matching_max:
            max_col = matching_max[0]
            mask_max = df[max_col] < df[avg_col]
            issue_flags[f"{prefix}_max_lt_avg"] = mask_max
            replaced_count[max_col] = mask_max.sum()
            df.loc[mask_max, max_col] = pd.NA
 
    return df, issue_flags, replaced_count
 
 
def write_report(report_path, df, file_name, event_info, event_id,
                  avg_cols, min_cols, max_cols, std_cols, meta_cols):
    # Writes a per-file summary report: column overview, matched event metadata,
    # time range, status distribution, constant columns, descriptive statistics,
    # and missing values per column
    with open(report_path, "w") as f:
        f.write(f"REPORT FOR {file_name}\n{'='*50}\n\n")
 
        f.write(f"Total columns: {len(df.columns)}\n")
        f.write(f"Meta columns: {meta_cols}\n")
        f.write(f"Avg columns: {len(avg_cols)}\n")
        f.write(f"Min columns: {len(min_cols)}\n")
        f.write(f"Max columns: {len(max_cols)}\n")
        f.write(f"Std columns: {len(std_cols)}\n\n")
 
        event_row = event_info[event_info["event_id"] == event_id]
        if not event_row.empty:
            f.write(f"Event ID: {event_id}\n")
            f.write(f"Event label: {event_row['event_label'].values[0]}\n")
            f.write(f"Event start: {event_row['event_start'].values[0]}\n")
            f.write(f"Event end: {event_row['event_end'].values[0]}\n")
            f.write(f"Event description: {event_row['event_description'].values[0]}\n\n")
        else:
            f.write("Event info: no matching event_id found for this file\n\n")
 
        if "time_stamp" in df.columns:
            df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
            f.write(f"Time range: {df['time_stamp'].min()} to {df['time_stamp'].max()}\n")
            f.write(f"Duplicate timestamps: {df['time_stamp'].duplicated().sum()}\n\n")
 
        if "status_type_id" in df.columns:
            f.write("Status type distribution:\n")
            f.write(df["status_type_id"].value_counts().to_string())
            f.write("\n\n")
 
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        f.write(f"Constant columns (no variation): {constant_cols}\n\n")
 
        f.write("\n\nDescribe (transposed):\n")
        f.write(df.describe().T.to_string())
 
        f.write("\n\nMissing values per column (only columns with missing values):\n")
        missing = df.isnull().sum()
        f.write(missing[missing > 0].to_string())
 
 
def process_file(file_path, event_info):
    # Runs the full cleaning pipeline for a single file: load, standardize
    # column names, drop unwanted columns, write the report, remove empty
    # rows, flag/replace implausible values, save the cleaned file
    file_name = os.path.basename(file_path)
    print(f"\nProcessing: {file_name}")
 
    event_id = get_event_id(file_name)
 
    # Auto-detect the delimiter (handles both "," and ";")
    df = pd.read_csv(file_path, sep=None, engine="python")
    df.columns = [clean_column_name(c) for c in df.columns]
 
    cols_present = [c for c in columns_to_drop if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
 
    # Identify sensor column groups based on the CARE dataset naming pattern
    avg_cols = [c for c in df.columns if c.endswith("_avg")]
    min_cols = [c for c in df.columns if c.endswith("_min")]
    max_cols = [c for c in df.columns if c.endswith("_max")]
    std_cols = [c for c in df.columns if c.endswith("_std")]
    meta_cols = [c for c in df.columns if c in
                 ["id", "asset_id", "time_stamp", "status_type_id", "train_test"]]
 
    report_path = os.path.join(report_folder, f"report_{file_name}.txt")
    write_report(report_path, df, file_name, event_info, event_id,
                 avg_cols, min_cols, max_cols, std_cols, meta_cols)
    print(f"Report saved to: {report_path}")
 
    # Remove rows that are completely empty
    df = df.dropna(how="all")
 
    df, issue_flags, replaced_count = flag_and_replace_implausible_values(
        df, avg_cols, min_cols, max_cols
    )
 
    # Save the flag summary only if implausible values were actually found
    flag_sums = issue_flags.sum()
    flag_sums = flag_sums[flag_sums > 0]
    if not flag_sums.empty:
        flag_summary_path = os.path.join(report_folder, f"flags_{file_name}.txt")
        flag_sums.to_csv(flag_summary_path, header=["count_of_implausible_rows"])
        print(f"Implausibility flags saved to: {flag_summary_path}")
 
    replaced_series = pd.Series(replaced_count)
    replaced_series = replaced_series[replaced_series > 0]
    if not replaced_series.empty:
        replaced_path = os.path.join(report_folder, f"replaced_values_{file_name}.txt")
        replaced_series.to_csv(replaced_path, header=["rows_fully_wiped_per_sensor"])
        print(f"Wiped-row counts saved to: {replaced_path}")
 
    # Save the cleaned file, keeping the original delimiter (semicolon)
    output_path = os.path.join(output_folder, file_name)
    df.to_csv(output_path, index=False, sep=";", decimal=",")
    print(f"Cleaned file saved to: {output_path}")
 
 
if __name__ == "__main__":
    event_info = pd.read_csv(event_info_path, sep=None, engine="python")
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
 
    for file_path in csv_files:
        process_file(file_path, event_info)
 
    print("\nAll files processed.")
 
