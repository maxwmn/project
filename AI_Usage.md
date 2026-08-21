# Documentation of AI-Assisted Development

This log documents the use of an AI assistant (Claude, Anthropic) in the development of the data cleaning and data analysis scripts used in this project. It lists the essential prompts that shaped the final code, along with a brief description of what was requested and what the AI contributed. Debugging exchanges (e.g. error messages and their fixes) are summarized rather than listed individually, unless they led to a meaningful change in the underlying logic.

The overall analytical concept, decisions on which methods to apply, and the interpretation of results were made by the author. The AI assistant was used to translate these decisions into working Python code, to suggest technical implementation options, and to help debug and refine the scripts iteratively.

---

## 1. Data Cleaning Script

| # | Prompt (paraphrased) | Purpose / AI contribution |
|---|---|---|
| 1 | Request for a Python script to clean multiple files in a folder using pandas commands specified by the author (`df.columns.tolist()`, `df.info()`, `df.describe()`, `df.isnull().sum()`, `df.dropna(how="all")`), later adapted to the specific structure of the CARE-to-Compare dataset (Avg/Min/Max/Std sensor groups) | Initial script structure: looping through CSV files, inspecting each with the specified commands, applying a basic cleaning step, and writing readable per-file diagnostic reports instead of console output |
| 2 | Request to check each sensor's Min and Max values against its Avg value for every timestamp, based on the known data quality issues documented for the dataset, and to replace implausible values | AI implemented the plausibility check (Min > Avg, Max < Avg) and replaced individual implausible Min/Max values with NaN, while keeping the Avg value unchanged, based on the dataset documentation noting Avg values as generally reliable |
| 3 | Request to add the option to remove entire columns (e.g. sensor_31, due to over 50% implausible values) | AI added a configurable list of columns to exclude from the dataset |
| 4 | Request to add further descriptive information to the report (time range, status_type_id distribution, constant columns) and to link event metadata (`event_info.csv`) to each report via filename/event ID | AI implemented the corresponding report sections and the event metadata lookup |


---

## 2. Data Analysis Script

| # | Prompt (paraphrased) | Purpose / AI contribution |
|---|---|---|
| 1 | Request for suggestions on how to compare the time shortly before an event with normal turbine behavior, based on the author's own idea of separating datasets by turbine and by normal/abnormal state, and visualizing `status_type_id` over time; the author specified that grouping should be done per turbine, using event labels to distinguish normal vs. anomalous datasets | AI proposed a defined pre-event time window, a normal-behavior baseline, a timeline visualization, and a quantitative frequency comparison, and implemented the corresponding functions for loading data, grouping files by turbine (`asset_id`), extracting the pre-event window, plotting a status timeline, and comparing status distributions |
| 2 | Feedback that the timeline plot was too cluttered with individual data points, with the author specifying that the most frequent status per hour should be used | AI implemented hourly aggregation of `status_type_id` |
| 3 | Feedback that the plot remained hard to read, and the observation that the event duration (start to end) was not visually represented, only the start | AI implemented a colored timeline (one color per status), limited to a window around the event, showing the full event duration as a shaded band |
| 4 | Request to combine all plots of one turbine into a single document, alongside a normal-condition reference, and to create a combined comparison table across all events | AI implemented `plot_turbine_overview()` and `build_turbine_status_table()`, combining multiple events and the normal baseline into shared figures/tables per turbine |
| 5 | Request for a summary table across all turbines and events, distinguishing normal, pre-event, and event periods, with a corresponding visualization; the author specified that both an averaged and a per-event version of the visualization should be produced | AI implemented `build_summary_table()` (one row per status/event/turbine, including differences from the normal baseline) and two visualization functions (`plot_summary_overview_averaged()` and `plot_summary_overview_per_event()`) |


