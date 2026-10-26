import pandas as pd

from src.config import (
    CURRENT_COL,
    MODULE_PACK_GROUPS,
    PACK_TEMP_GROUPS,
    TEMP_SENSOR_COLS,
    TIMESTAMP_FORMAT,
    TS_COL,
    VOLTAGE_COL,
    VOLTAGE_SPREAD_FEATURES,
    X_RAW,
)

MIN_VALID_TEMP_SENSORS_PER_PACK = 4
MIN_REQUIRED_MODULES = 1


def get_complete_pack_groups(columns: pd.Index | list[str]) -> dict[str, list[str]]:
    """
    Find all pack groups whose temperature sensors are fully present in the dataset.

    A pack only counts as available here if all expected temperature-sensor
    columns exist in the loaded dataset. This lets the pipeline handle different
    hardware layouts robustly without failing on modules or packs that are not
    present.
    """
    column_set = set(columns)
    complete_packs = {}

    for pack_name, sensor_cols in PACK_TEMP_GROUPS.items():
        if all(col in column_set for col in sensor_cols):
            complete_packs[pack_name] = sensor_cols

    return complete_packs


def get_complete_module_groups(complete_packs: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Find all modules for which both defined packs are fully present.

    A module is only considered available if both pack names from
    `MODULE_PACK_GROUPS` were detected in the dataset as complete pack groups.
    """
    complete_modules = {}

    for module_name, pack_names in MODULE_PACK_GROUPS.items():
        if all(pack_name in complete_packs for pack_name in pack_names):
            complete_modules[module_name] = pack_names

    return complete_modules


def select_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the raw dataset to the columns the pipeline needs.

    The function builds the target list from the timestamp, the defined raw
    features and the source-file column, and checks at the same time whether
    anything essential is missing. If required columns are missing, a clear error
    is raised early. If all columns are present, a clean slice of the DataFrame
    is returned.

    Call context:
    This is the first step in `prepare_base_dataframe`.
    """
    complete_packs = get_complete_pack_groups(df.columns)
    complete_modules = get_complete_module_groups(complete_packs)

    if len(complete_modules) < MIN_REQUIRED_MODULES:
        raise ValueError(
            "No supported battery module layout found in raw data.\n"
            f"Detected complete packs: {sorted(complete_packs.keys())}\n"
            "Need at least one full module with both packs present."
        )

    available_temp_cols = [
        col
        for pack_name in complete_packs
        for col in PACK_TEMP_GROUPS[pack_name]
    ]

    need = [TS_COL] + X_RAW[: len(X_RAW) - len(TEMP_SENSOR_COLS)] + available_temp_cols + ["__source_file"]
    missing = [c for c in need if c not in df.columns]

    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(missing))

    return df[need].copy()


def parse_and_sort_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the timestamp column to real datetime values and sort chronologically.

    Invalid timestamps are dropped, so subsequent steps only work with consistent
    time series. Sorting makes sure that differences, rolling windows and other
    time-based calculations are based on the correct order.

    Call context:
    Run in `prepare_base_dataframe` before any time-based calculation.
    """
    df[TS_COL] = pd.to_datetime(
        df[TS_COL],
        format=TIMESTAMP_FORMAT,
        errors="coerce",
    )
    df = df.dropna(subset=[TS_COL]).sort_values(TS_COL).reset_index(drop=True)
    return df


def get_numeric_columns() -> list[str]:
    """
    Return the set of raw columns that have to be interpreted as numeric.

    This includes in particular the temperature sensors, the voltage-spread
    inputs and the central electrical signals. The function keeps the definition
    in one place, so the actual type conversion does not have to maintain several
    scattered lists.

    Call context:
    Internal helper for `convert_numeric_columns`.
    """
    numeric_cols = (
        TEMP_SENSOR_COLS
        + VOLTAGE_SPREAD_FEATURES
        + [
            CURRENT_COL,
            VOLTAGE_COL,
        ]
    )

    seen = set()
    unique_numeric_cols = []
    for col in numeric_cols:
        if col not in seen:
            unique_numeric_cols.append(col)
            seen.add(col)

    return unique_numeric_cols


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all relevant raw signal columns to numeric values.

    Entries that cannot be interpreted are deliberately turned into `NaN`, so they
    can be detected cleanly in later validation or filter steps. This prevents
    faulty string or mixed formats from slipping unnoticed into the feature
    calculation.

    Call context:
    Run in `prepare_base_dataframe` after the column selection.
    """
    numeric_cols = get_numeric_columns()

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def get_sufficient_pack_temperature_mask(
    df: pd.DataFrame,
    pack_groups: dict[str, list[str]],
) -> pd.Series:
    """
    Check for all rows at once whether every available pack has enough valid temperature sensors.

    The logic matches the previous row-by-row check: for every pack, the number of
    non-missing sensor values is counted per row. A row is only kept if every
    pack taken into account has at least `MIN_VALID_TEMP_SENSORS_PER_PACK` valid
    values.

    The implementation is deliberately vectorised, so that very large datasets
    can also be processed efficiently.
    """
    mask = pd.Series(True, index=df.index)

    for cols in pack_groups.values():
        pack_valid_mask = df[cols].notna().sum(axis=1) >= MIN_VALID_TEMP_SENSORS_PER_PACK
        mask &= pack_valid_mask

    return mask


def drop_rows_missing_core_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all rows that are unsuitable for reliable feature engineering.

    First, rows without the central electrical signals are dropped. Then rows
    without the required voltage-spread inputs are excluded. Finally,
    `get_sufficient_pack_temperature_mask` makes sure that the temperature coverage
    per pack is high enough as well. The result is a much more robust input
    dataset for the subsequent engineering steps.

    Call context:
    Run in `prepare_base_dataframe` after the numeric type conversion.
    """
    pack_groups = get_complete_pack_groups(df.columns)
    module_groups = get_complete_module_groups(pack_groups)

    if len(module_groups) < MIN_REQUIRED_MODULES:
        raise ValueError(
            "No supported battery module layout remains after column selection.\n"
            f"Detected complete packs: {sorted(pack_groups.keys())}\n"
            "Need at least one full module with both packs present."
        )

    # Only keep what actually matters
    core = [CURRENT_COL, VOLTAGE_COL]

    df = df.dropna(subset=core).reset_index(drop=True)

    # Require voltage spread inputs
    df = df.dropna(subset=VOLTAGE_SPREAD_FEATURES).reset_index(drop=True)

    # Require proper temperature coverage (pack-aware)
    mask = get_sufficient_pack_temperature_mask(df, pack_groups=pack_groups)
    df = df[mask].reset_index(drop=True)

    return df


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete preprocessing for the combined raw dataset.

    This function wraps the individual preprocessing steps in the right order:
    select the required columns, parse timestamps, run the numeric conversion
    and remove unusable rows. The result is the stable starting point for all
    subsequent feature engineering.

    Call context:
    Called in `src.main.extract` immediately before `build_feature_table`.
    """
    df = select_required_columns(df)
    df = parse_and_sort_timestamp(df)
    df = convert_numeric_columns(df)
    df = drop_rows_missing_core_signals(df)
    return df
