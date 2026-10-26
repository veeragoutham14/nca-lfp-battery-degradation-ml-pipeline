
from src.anomaly_detection import (
    get_all_anomalies,
    get_top_anomalies,
    run_isolation_forest,
)
from src.archive_utils import archive_run_outputs
from src.config import (
    ANOMALY_OUT_DIR,
    ARCHIVE_BASE_DIR,
    DATA_DIR,
    FEATURE_OUTPUT_FILENAME,
    OUT_DIR,
)
from src.feature_engineering import build_feature_table
from src.io_utils import (
    load_all_csvs,
    load_feature_table,
    save_anomaly_tables,
    save_feature_table,
)
from src.preprocessing import prepare_base_dataframe
from src.visualization import run_all_visual_checks


def extract() -> None:
    """Run the complete raw-data preparation up to the saved feature file.

    The flow covers reading all raw CSV files, preprocessing, the subsequent
    feature engineering and writing the final feature table to the processed
    area. This function is therefore the entry point whenever the underlying raw
    data has changed.

    Call context:
    Can be enabled directly from `main()` and serves as the upstream step before
    visualisation or anomaly detection.
    """
    print("\n--- Loading raw CSV logs ---")
    df_raw = load_all_csvs(DATA_DIR)

    print("\n--- Running preprocessing ---")
    df_base = prepare_base_dataframe(df_raw)

    print("\n--- Running feature engineering ---")
    df_feat, feature_cols = build_feature_table(df_base)

    print("\n--- Saving processed dataset ---")
    save_feature_table(df_feat, OUT_DIR)

    print("\nExtraction complete.")
    print("Feature count:", len(feature_cols))


def visualize() -> None:
    """Load the feature table and start the graphical plausibility checks.

    This function is not part of the actual model computation; it is for
    qualitative checking of the created features. Developers can use it to
    inspect time series, distributions and relationships before using the
    features for anomaly detection.

    Call context:
    Optional pipeline function that can be enabled from `main()`.
    """
    print("\n--- Loading processed feature table ---")
    df_feat = load_feature_table(OUT_DIR)

    print("\n--- Running visualization checks ---")
    run_all_visual_checks(df_feat)


def detect_anomalies() -> None:
    """Run the core anomaly-detection step on the feature table.

    First, the previously created feature file is loaded. The dataset is then
    scored with the Isolation Forest model, split into anomaly-only rows and a top
    selection, and finally written to several result files. Optionally,
    visualisations of individual anomaly features can be created afterwards.

    Call context:
    This is the central runtime step and is currently enabled by default in
    `main()`.
    """
    print("\n--- Loading processed feature table ---")
    df_feat = load_feature_table(OUT_DIR)

    print("\n--- Running Isolation Forest anomaly detection ---")
    df_scored = run_isolation_forest(df_feat)

    df_anomalies_only = get_all_anomalies(df_scored)
    df_top20 = get_top_anomalies(df_scored, n=20)

    print("\nTotal samples:", len(df_scored))
    print("Detected anomalies:", len(df_anomalies_only))

    print("\n--- Saving anomaly outputs ---")
    save_anomaly_tables(
        df_all=df_scored,
        df_anomalies_only=df_anomalies_only,
        df_top20=df_top20,
        out_dir=ANOMALY_OUT_DIR,
    )

    # Optional plotting
    """
    print("\n--- Plotting anomalies ---")
    plot_anomalies(df_scored, "temp_spread_C")
    if "cell_voltage_spread_V" in df_scored.columns:
        plot_anomalies(df_scored, "cell_voltage_spread_V")
    if "dT_max_dt_Cps" in df_scored.columns:
        plot_anomalies(df_scored, "dT_max_dt_Cps")
    """

    print("\nAnomaly detection complete.")


def archive_outputs(run_label: str = "iforest_pack_v1") -> None:
    """Archive the created run artefacts after a successful run.

    The function passes the relevant directories and files to the archiving logic
    and makes sure that temporary model results are stored safely, while the
    processed feature file stays in its usual place. This is especially helpful
    when several runs should be compared or documented over time.

    Call context:
    Usually called in `main()` after `detect_anomalies`.
    """
    print("\n--- Archiving temporary outputs ---")

    processed_feature_file = OUT_DIR / FEATURE_OUTPUT_FILENAME

    archive_dir = archive_run_outputs(
        models_dir=ANOMALY_OUT_DIR,
        processed_feature_file=processed_feature_file,
        archive_base_dir=ARCHIVE_BASE_DIR,
        run_label=run_label,
        extra_metadata={
            "pipeline_mode": "detect_only",
        },
    )

    print("Archived outputs to:", archive_dir)
    print("Cleared working folders:", ANOMALY_OUT_DIR)
    print("Kept processed features untouched in:", OUT_DIR)


def main() -> None:
    """Control which top-level pipeline steps are currently enabled.

    This function decides which phases of a run are actually executed, for
    example only anomaly detection, or extraction, visualisation and archiving as
    well. It also wraps the top-level error handling, so that the reason for an
    abort is reported clearly when something goes wrong.

    Call context:
    The `__main__` block at the end of the file calls this function when the
    module is started as a script.
    """
    try:
        #extract()   # run only when raw data changed
        detect_anomalies()
        # visualize()  # enable if you want visualization output before archive
        #archive_outputs(run_label="iforest_mode_aware_Peng")

    except Exception as exc:
        print("\nPipeline failed before archive reset.")
        print("Reason:", exc)
        raise


if __name__ == "__main__":
    main()
