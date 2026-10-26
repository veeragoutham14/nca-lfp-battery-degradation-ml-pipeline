from pathlib import Path

import pandas as pd

from src.config import (
    ANOMALY_ALL_OUTPUT_FILENAME,
    ANOMALY_ONLY_OUTPUT_FILENAME,
    ANOMALY_TOP20_OUTPUT_FILENAME,
    CSV_GLOB,
    FEATURE_OUTPUT_FILENAME,
    TS_COL,
)


def read_logger_csv(path: Path) -> pd.DataFrame:
    """Load a single raw logger CSV in the expected export format.

    The function handles the project-specific CSV structure with a header-row
    offset, semicolon separators and a comma as the decimal separator. It then
    cleans up the column names and stores the source file name in its own column,
    so it stays traceable which file a row came from.

    Call context:
    Called by `load_all_csvs` for every file found.
    """
    df = pd.read_csv(
        path,
        skiprows=6,
        sep=";",
        decimal=",",
        engine="python",
        on_bad_lines="skip",
    )

    df = df.rename(columns=lambda c: c.lstrip("# ").strip()).copy()

    # Add the source filename in one concat step to avoid DataFrame fragmentation
    # warnings on wide CSV exports.
    source_col = pd.Series(path.name, index=df.index, name="__source_file")
    df = pd.concat([df, source_col], axis=1, copy=False)
    return df


def load_all_csvs(data_dir: Path) -> pd.DataFrame:
    """Load all raw CSV files from a directory and combine them.

    Each file is read individually via `read_logger_csv`. Files that load
    successfully are collected; faulty files are only logged, so the overall run
    stays as robust as possible. Finally, all valid partial DataFrames are
    concatenated and returned as one combined raw dataset.

    Call context:
    This is the first data-loading step in `src.main.extract`.
    """
    files = sorted(data_dir.glob(CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    bad = []

    for f in files:
        try:
            d = read_logger_csv(f)
            dfs.append(d)
            print(f"Loaded {f.name}: {d.shape}")
        except Exception as e:
            bad.append((f.name, str(e)))

    if bad:
        print("\nFiles that failed to load:")
        for name, err in bad:
            print("-", name, ":", err)

    if not dfs:
        raise ValueError("No valid CSV files could be loaded.")

    df = pd.concat(dfs, ignore_index=True)
    print("\nCombined shape:", df.shape)
    return df


def save_feature_table(df: pd.DataFrame, out_dir: Path) -> Path:
    """Save the prepared feature data as a processed CSV file.

    The function writes the given DataFrame with the project-specific CSV
    settings into the target directory and returns the final file path. The file
    is then consistently available for visualisation, anomaly detection or
    archiving.

    Call context:
    Called in `src.main.extract` after `build_feature_table` has finished.
    """
    out_csv = out_dir / FEATURE_OUTPUT_FILENAME
    df.to_csv(
        out_csv,
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig"
        
    )
    print("Saved extracted dataset to:", out_csv)
    return out_csv


def load_feature_table(out_dir: Path) -> pd.DataFrame:
    """Load the previously created feature table back from the file system.

    In addition, the timestamp column, if present, is converted back to a
    datetime type. The function is therefore the standard loading interface for
    all pipeline steps that work with processed features instead of raw data.

    Call context:
    Used by `src.main.visualize` and `src.main.detect_anomalies`.
    """
    feature_path = out_dir / FEATURE_OUTPUT_FILENAME
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Processed feature file not found: {feature_path}\n"
            "Run extraction first."
        )

    df = pd.read_csv(feature_path, sep=";", decimal=",")

    if TS_COL in df.columns:
        df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")

    print("Loaded processed feature table:", feature_path)
    print("Shape:", df.shape)
    return df


def save_anomaly_tables(
    df_all: pd.DataFrame,
    df_anomalies_only: pd.DataFrame,
    df_top20: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write the three central result files of the anomaly detection to disk.

    It saves the complete scored dataset, the anomaly-only subset and a compact
    top-N selection of the most conspicuous cases. All files are written in the
    same CSV format, so they can easily be reused, compared or archived later.

    Call context:
    Run in `src.main.detect_anomalies` after model scoring and explanation
    enrichment.
    """
    all_path = out_dir / ANOMALY_ALL_OUTPUT_FILENAME
    only_path = out_dir / ANOMALY_ONLY_OUTPUT_FILENAME
    top20_path = out_dir / ANOMALY_TOP20_OUTPUT_FILENAME

    for df, path in [
        (df_all, all_path),
        (df_anomalies_only, only_path),
        (df_top20, top20_path),
    ]:
        df.to_csv(
            path,
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig"
        )

    print("Saved all anomaly results to:", all_path)
    print("Saved anomaly-only rows to:", only_path)
    print("Saved top 20 anomalies to:", top20_path)

    return all_path, only_path, top20_path


def load_anomaly_table(path: Path) -> pd.DataFrame:
    """Load a previously saved result file of the anomaly detection.

    This helper is meant for later analysis, notebook or visualisation work. If
    the timestamp column is present, it is converted to a datetime type while
    reading, so the dataset can be processed further without extra preparation.

    Call context:
    Meant as a general helper for downstream evaluations of the files created by
    `save_anomaly_tables`.
    """
    if not path.exists():
        raise FileNotFoundError(f"Anomaly file not found: {path}")

    df = pd.read_csv(path, sep=";", decimal=",")

    if TS_COL in df.columns:
        df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")

    print("Loaded anomaly table:", path)
    print("Shape:", df.shape)
    return df
