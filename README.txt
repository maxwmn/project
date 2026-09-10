# Wind Turbine SCADA Analysis — Wind Farm A

This repository contains the code and outputs for a project analyzing wind turbine SCADA data from Wind Farm A, 
part of the CARE-to-Compare dataset (Gück et al.), available at https://zenodo.org/records/10958775. 
The goal is to investigate whether operational status patterns can serve as early indicators of turbine anomalies, 
and to what extent this is practically useful for preventive action. The project combines a descriptive status ID 
analysis with a complementary machine learning approach.

## Project structure

Wind Farm A/
    datasets/
        Raw SCADA CSV files (one per event), from the CARE-to-Compare dataset
    event_info.csv
        Event metadata: event_id, label (normal/anomaly), start, end, description

data_cleaning/
    cleaned_data/
        Cleaned CSV files, output of clean_data.py
    reports/
        Per-event diagnostic reports and summary tables

analysis_output/
    overview_asset<ID>.png
        Status timeline per turbine (all events + normal reference)
    status_table_asset<ID>.csv
        Status share comparison per turbine
    summary_all_events_<window>days.csv
        Combined summary table across all turbines/events, per window size
    summary_bars_averaged.png
        Averaged status share bar chart, all turbines
    summary_bars_asset<ID>.png
        Per-event status share bar chart, per turbine

isolation_forest_output/
    isolation_forest_contamination_sweep.csv
        Overall precision/recall/F-beta/lead-time results across all tested contamination values, sorted by F2
    isolation_forest_per_turbine_contamination_<value>.csv
        Per-turbine results (precision/recall/F-beta, lead time before event_start) for one contamination value
    isolation_forest_alarm_feature_explanation_contamination_<value>.csv
        Top contributing features behind each flagged anomaly, for the best (F2) contamination value
    isolation_forest_feature_importance_contamination_<value>.csv
        Overall feature ranking (mean |z-score| pooled across all turbines' alarm rows), for the best (F2) contamination value
    isolation_forest_f2_score.png
        F2 score across all tested contamination values, best one highlighted
    isolation_forest_feature_importance.png
        Overall most influential features (actual sensor names, not raw column codes)




clean_data.py
    Cleans the raw SCADA files (see below)
event_analysis.py
    Runs the status ID analysis
summarize_implausibility.py
    Aggregates implausibility flags across all events into one table
anomaly_detection_isolation_forest.py
    Per-turbine Isolation Forest anomaly detection on the cleaned SCADA data
plot_isolation_forest_results.py
    Visualizes the best F2 score and the most important features from the Isolation Forest results

README.txt

## Scripts and how they fit together

Run in this order:

1. clean_data.py
Reads the raw SCADA files from Wind Farm A/datasets/, standardizes column names, removes sensors with severe data quality issues (sensor_31, sensor_46, sensor_49), checks each sensor's Min/Max values against its Avg value for physical plausibility, replaces implausible values with NaN, and writes the cleaned files to data_cleaning/cleaned_data/. Also generates a diagnostic report per event in data_cleaning/reports/.

2. summarize_implausibility.py (optional, for reporting)
Aggregates the implausibility flags generated during cleaning into a single overview table (implausibility_summary.csv), showing the share of implausible values per event.

3. event_analysis.py
Loads the cleaned files and event_info.csv, groups them by turbine, and compares the distribution of status_type_id during normal operation, the pre-event window, and the event itself. Produces per-turbine status timelines, bar charts, and a combined summary table across all turbines and events. The pre-event window size (in days) can be adjusted via the PRE_EVENT_WINDOW variable at the top of the script.

4. anomaly_detection_isolation_forest.py
Loads the cleaned files, engineers features (power curve deviation, gear ratio deviation, rolling trend deviations), and fits a separate Isolation Forest per turbine on its own "train" rows to flag unusual "prediction" rows. Sweeps several contamination values, evaluates each against the fault windows in event_info.csv (precision/recall/F-beta, lead time before event_start), and reports per-turbine results plus the top contributing features for the best contamination value. Results are written to isolation_forest_output/.

5. plot_isolation_forest_results.py (optional, for visualization)
Reads the CSVs written by anomaly_detection_isolation_forest.py (no refitting) and produces two standalone charts: the F2 (Fbeta) score across all tested contamination values, with the best one highlighted, and the overall most influential features driving alarms at that contamination (mean |z-score| pooled across all turbines' alarm rows), labeled with their actual sensor descriptions from feature_description.csv rather than the raw anonymized column names. Saves isolation_forest_output/isolation_forest_f2_score.png and isolation_forest_output/isolation_forest_feature_importance.png.


## Requirements

- Python 3.10+
- pandas
- matplotlib

## Running the scripts

From the project's root folder:
python clean_data.py
python event_analysis.py

Adjust the folder paths at the top of each script if your local folder structure differs from the one shown above.

## Data source

Gück, S., Roelofs, C. et al. (2024). CARE to Compare: A real-world dataset for anomaly detection in wind turbine data [Data set]. Zenodo. https://zenodo.org/records/10958775

## Authors

Tim Nikolov & Max Wiemann