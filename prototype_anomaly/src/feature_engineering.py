import numpy as np
import pandas as pd

from src.config import (
    CELL_VOLT_MAX_COL,
    CELL_VOLT_MIN_COL,
    CURRENT_COL,
    MODULE_PACK_GROUPS,
    PACK_NAMES,
    PACK_TEMP_GROUPS,
    POWER_COL,
    REST_A,
    ROLL_S,
    SOC_COL,
    SOH_COL,
    TS_COL,
    VOLTAGE_COL,
)


def add_dt_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the time step between two consecutive rows.

    The difference in seconds is computed from the timestamp column and stored as
    `dt_s`. Missing or non-positive values are then replaced with a robust
    median-based substitute. This matters because later derivatives and
    time-based normalisations only work reliably when `dt_s` is valid and
    positive.

    Call context:
    Run in `build_feature_table` as the first feature-engineering step.
    """
    df = df.copy()

    df["dt_s"] = df[TS_COL].diff().dt.total_seconds()
    med_dt = df["dt_s"].dropna().median()

    if pd.isna(med_dt) or med_dt <= 0:
        med_dt = 1.0

    df["dt_s"] = df["dt_s"].fillna(med_dt)
    df.loc[df["dt_s"] <= 0, "dt_s"] = med_dt
    return df


def add_current_voltage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive basic electrical context features from current and voltage.

    Among others, this produces absolute values, squares, and discrete and
    time-normalised changes of current and voltage. These features form the
    electrical context for the later anomaly analysis and are also needed by
    several subsequent steps such as mode classification, load normalisation and
    rolling features.

    Call context:
    Called in `build_feature_table` directly after `add_dt_feature`.
    """
    df = df.copy()

    df["current_abs_A"] = df[CURRENT_COL].abs()
    df["current_sq_A2"] = df[CURRENT_COL] ** 2

    df["dI_A"] = df[CURRENT_COL].diff()
    df["dV_V"] = df[VOLTAGE_COL].diff()

    df["dI_dt_Aps"] = df["dI_A"] / df["dt_s"]
    df["dV_dt_Vps"] = df["dV_V"] / df["dt_s"]

    if POWER_COL in df.columns:
        df["power_abs_W"] = df[POWER_COL].abs()

    return df


def add_operating_mode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label every row with an operating mode and mark mode changes.

    Based on the current, rows are classified as rest, charge or discharge. On top
    of that, the function detects when the operating mode changes and how many
    seconds have passed since the last change. This information is relevant both
    for describing features and for the later mode-aware explanation of
    anomalies.

    Call context:
    Run in `build_feature_table`; the mode columns it creates are reused later in
    the explainer module.
    """
    df = df.copy()

    df["is_rest"] = (df["current_abs_A"] < REST_A).astype(int)
    df["is_charging"] = (df[CURRENT_COL] > REST_A).astype(int)
    df["is_discharging"] = (df[CURRENT_COL] < -REST_A).astype(int)

    mode_now = np.select(
        [df["is_rest"] == 1, df["is_charging"] == 1, df["is_discharging"] == 1],
        ["rest", "charging", "discharging"],
        default="unknown",
    )
    df["mode_label"] = mode_now

    mode_change = df["mode_label"] != df["mode_label"].shift(1)
    df["mode_change_flag"] = mode_change.astype(int)

    # time since last mode change
    segment_id = mode_change.cumsum()
    df["time_since_mode_change_s"] = df.groupby(segment_id)["dt_s"].cumsum()

    return df


def add_voltage_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the cell-voltage spread from the minimum and maximum voltage.

    This feature is a direct indicator of electrical imbalance in the battery
    system. If both required input columns are present, their difference is
    written to the DataFrame as the new feature `cell_voltage_spread_V`.

    Call context:
    Run in `build_feature_table`, so the feature is available later both for the
    model and for explanations.
    """
    df = df.copy()

    if CELL_VOLT_MAX_COL in df.columns and CELL_VOLT_MIN_COL in df.columns:
        df["cell_voltage_spread_V"] = df[CELL_VOLT_MAX_COL] - df[CELL_VOLT_MIN_COL]

    return df


def add_pack_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build pack- and module-level temperature features from the individual sensors.

    For every pack, the mean, standard deviation, maximum, minimum, spread, index
    of the hottest sensor and hotspot delta are computed. On top of that,
    higher-level summaries across all packs are built, such as the temperature
    spread between packs, internal non-uniformity, or differences between the two
    packs of a module. The function thus condenses raw sensor data into
    physically interpretable structural features.

    Call context:
    This is one of the central steps in `build_feature_table`; many later
    features and practically the whole thermal anomaly explanation build on it.

    Why:
    - We no longer want global max/min over all 32 sensors.
    - We want to learn abnormal pack behaviour.
    """
    df = df.copy()

    pack_mean_cols = []
    pack_std_cols = []
    pack_spread_cols = []
    pack_hotspot_cols = []

    for pack_name, sensor_cols in PACK_TEMP_GROUPS.items():
        available = [c for c in sensor_cols if c in df.columns]
        if not available:
            continue

        mean_col = f"{pack_name}_temp_mean_C"
        std_col = f"{pack_name}_temp_std_C"
        max_col = f"{pack_name}_temp_max_C"
        min_col = f"{pack_name}_temp_min_C"
        spread_col = f"{pack_name}_temp_spread_C"
        hot_idx_col = f"{pack_name}_hot_sensor_idx"
        hotspot_delta_col = f"{pack_name}_hotspot_delta_C"

        df[mean_col] = df[available].mean(axis=1)
        df[std_col] = df[available].std(axis=1)
        df[max_col] = df[available].max(axis=1)
        df[min_col] = df[available].min(axis=1)
        df[spread_col] = df[max_col] - df[min_col]

        # sensor index 0..3 of hottest sensor within this pack
        df[hot_idx_col] = df[available].to_numpy().argmax(axis=1)
        df[hotspot_delta_col] = df[max_col] - df[mean_col]

        pack_mean_cols.append(mean_col)
        pack_std_cols.append(std_col)
        pack_spread_cols.append(spread_col)
        pack_hotspot_cols.append(hotspot_delta_col)

    if not pack_mean_cols:
        raise ValueError("No pack-level temperature features could be computed.")

    # Summary across pack means
    df["pack_temp_spread_C"] = df[pack_mean_cols].max(axis=1) - df[pack_mean_cols].min(axis=1)
    df["pack_temp_std_C"] = df[pack_mean_cols].std(axis=1)
    df["pack_temp_median_C"] = df[pack_mean_cols].median(axis=1)
    df["pack_max_dev_from_median_C"] = (
        df[pack_mean_cols].sub(df["pack_temp_median_C"], axis=0).abs().max(axis=1)
    )

    # Summary across pack internal spreads
    df["max_pack_internal_spread_C"] = df[pack_spread_cols].max(axis=1)
    df["mean_pack_internal_spread_C"] = df[pack_spread_cols].mean(axis=1)

    # Summary across pack std
    df["max_pack_internal_std_C"] = df[pack_std_cols].max(axis=1)
    df["mean_pack_internal_std_C"] = df[pack_std_cols].mean(axis=1)

    # Summary across pack hotspots
    df["max_pack_hotspot_delta_C"] = df[pack_hotspot_cols].max(axis=1)
    df["mean_pack_hotspot_delta_C"] = df[pack_hotspot_cols].mean(axis=1)

    # Module-level two-pack mismatch
    module_delta_cols = []
    for module_name, pack_pair in MODULE_PACK_GROUPS.items():
        if len(pack_pair) != 2:
            continue

        p1, p2 = pack_pair
        p1_mean = f"{p1}_temp_mean_C"
        p2_mean = f"{p2}_temp_mean_C"
        delta_col = f"{module_name}_pack_delta_C"

        if p1_mean in df.columns and p2_mean in df.columns:
            df[delta_col] = (df[p1_mean] - df[p2_mean]).abs()
            module_delta_cols.append(delta_col)

    if module_delta_cols:
        df["max_intra_module_pack_delta_C"] = df[module_delta_cols].max(axis=1)
        df["mean_intra_module_pack_delta_C"] = df[module_delta_cols].mean(axis=1)

    return df


def add_temperature_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rates of change for the thermal summary features created earlier.

    Instead of looking only at static temperature patterns, this function adds
    features that describe how quickly the spread, standard deviation or median
    deviation of the pack temperatures change. Such dynamic features can be
    especially informative for transient or building-up anomalies.

    Call context:
    Run in `build_feature_table` after `add_pack_temperature_features`.
    """
    df = df.copy()

    df["d_pack_temp_spread_C"] = df["pack_temp_spread_C"].diff()
    df["d_pack_temp_std_C"] = df["pack_temp_std_C"].diff()
    df["d_pack_max_dev_from_median_C"] = df["pack_max_dev_from_median_C"].diff()

    df["d_pack_temp_spread_dt_Cps"] = df["d_pack_temp_spread_C"] / df["dt_s"]
    df["d_pack_temp_std_dt_Cps"] = df["d_pack_temp_std_C"] / df["dt_s"]
    df["d_pack_max_dev_from_median_dt_Cps"] = df["d_pack_max_dev_from_median_C"] / df["dt_s"]

    return df


def add_load_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create current-normalised features for load-dependency analysis.

    Some temperature or voltage deviations have to be judged differently under
    high load than at rest. Existing core features are therefore divided by the
    absolute current, provided there is enough load at all. During rest phases
    the values are deliberately set to `NaN`, so later steps can handle these
    special cases in a controlled way.

    Call context:
    Run in `build_feature_table` before the final row filters and NaN handling.
    """
    df = df.copy()

    load_mask = df["current_abs_A"] >= REST_A

    if "pack_temp_spread_C" in df.columns:
        df["pack_temp_spread_per_current"] = np.where(
            load_mask,
            df["pack_temp_spread_C"] / df["current_abs_A"],
            np.nan,
        )

    if "pack_max_dev_from_median_C" in df.columns:
        df["pack_dev_from_median_per_current"] = np.where(
            load_mask,
            df["pack_max_dev_from_median_C"] / df["current_abs_A"],
            np.nan,
        )

    if "cell_voltage_spread_V" in df.columns:
        df["volt_spread_per_current"] = np.where(
            load_mask,
            df["cell_voltage_spread_V"] / df["current_abs_A"],
            np.nan,
        )

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling-window features based on the time index.

    Once the base features exist, smoothed means, standard deviations, maxima and
    persistence signals are computed over a seconds-based rolling window. This
    makes it easier to tell whether a conspicuous pattern appears only briefly or
    persists over a longer phase.

    Call context:
    Run towards the end of `build_feature_table`, once all required source
    features have been created.
    """
    df = df.copy()
    df = df.set_index(TS_COL)

    win = f"{ROLL_S}s"

    # Current / voltage rolling context
    df[f"I_mean_{ROLL_S}s"] = df[CURRENT_COL].rolling(win, min_periods=3).mean()
    df[f"I_abs_mean_{ROLL_S}s"] = df["current_abs_A"].rolling(win, min_periods=3).mean()
    df[f"I_std_{ROLL_S}s"] = df[CURRENT_COL].rolling(win, min_periods=3).std()
    df[f"I_max_abs_{ROLL_S}s"] = df["current_abs_A"].rolling(win, min_periods=3).max()

    df[f"V_mean_{ROLL_S}s"] = df[VOLTAGE_COL].rolling(win, min_periods=3).mean()

    # Thermal persistence
    df[f"pack_temp_spread_mean_{ROLL_S}s"] = (
        df["pack_temp_spread_C"].rolling(win, min_periods=3).mean()
    )
    df[f"pack_temp_std_mean_{ROLL_S}s"] = (
        df["pack_temp_std_C"].rolling(win, min_periods=3).mean()
    )
    df[f"pack_max_dev_from_median_mean_{ROLL_S}s"] = (
        df["pack_max_dev_from_median_C"].rolling(win, min_periods=3).mean()
    )

    if "max_intra_module_pack_delta_C" in df.columns:
        df[f"max_intra_module_pack_delta_mean_{ROLL_S}s"] = (
            df["max_intra_module_pack_delta_C"].rolling(win, min_periods=3).mean()
        )

    if "cell_voltage_spread_V" in df.columns:
        df[f"Vspread_mean_{ROLL_S}s"] = (
            df["cell_voltage_spread_V"].rolling(win, min_periods=3).mean()
        )

    df = df.reset_index()
    return df


def get_feature_columns() -> list[str]:
    """Return the ordered target list of all intended model features.

    This function defines in one place which of the created columns are treated
    as relevant features for training, scoring and analysis. The list contains
    context, thermal, voltage, dynamic, rolling and optional power features.
    Duplicate entries are removed at the end while the original order is kept.

    Call context:
    Used in `build_feature_table` to select, from the full DataFrame, the feature
    columns that are actually passed on.
    """
    cols = [
        # Context you may or may not train on later
        CURRENT_COL,
        VOLTAGE_COL,
        SOC_COL,
        SOH_COL,
        "dt_s",

        # Current / electrical context
        "current_abs_A",
        "current_sq_A2",
        "dI_dt_Aps",
        "dV_dt_Vps",
        "is_rest",
        "is_charging",
        "is_discharging",
        "mode_change_flag",
        "time_since_mode_change_s",

        # Electrical imbalance
        "cell_voltage_spread_V",
        "volt_spread_per_current",

        # Pack-based thermal summary
        "pack_temp_spread_C",
        "pack_temp_std_C",
        "pack_max_dev_from_median_C",
        "max_pack_internal_spread_C",
        "mean_pack_internal_spread_C",
        "max_pack_internal_std_C",
        "mean_pack_internal_std_C",
        "max_pack_hotspot_delta_C",
        "mean_pack_hotspot_delta_C",

        # Thermal dynamics
        "d_pack_temp_spread_dt_Cps",
        "d_pack_temp_std_dt_Cps",
        "d_pack_max_dev_from_median_dt_Cps",

        # Thermal / current ratios you can selectively ignore later
        "pack_temp_spread_per_current",
        "pack_dev_from_median_per_current",

        # Rolling current / voltage context
        f"I_mean_{ROLL_S}s",
        f"I_abs_mean_{ROLL_S}s",
        f"I_std_{ROLL_S}s",
        f"I_max_abs_{ROLL_S}s",
        f"V_mean_{ROLL_S}s",

        # Rolling thermal persistence
        f"pack_temp_spread_mean_{ROLL_S}s",
        f"pack_temp_std_mean_{ROLL_S}s",
        f"pack_max_dev_from_median_mean_{ROLL_S}s",

        # Rolling voltage persistence
        f"Vspread_mean_{ROLL_S}s",
    ]

    # Add module-level pack delta features
    for module_name in MODULE_PACK_GROUPS.keys():
        cols.append(f"{module_name}_pack_delta_C")

    cols.append(f"max_intra_module_pack_delta_mean_{ROLL_S}s")

    # Add individual pack features so you can inspect or selectively train
    for pack_name in PACK_NAMES:
        cols.extend(
            [
                f"{pack_name}_temp_mean_C",
                f"{pack_name}_temp_std_C",
                f"{pack_name}_temp_spread_C",
                f"{pack_name}_hot_sensor_idx",
                f"{pack_name}_hotspot_delta_C",
            ]
        )

    # Optional power features
    optional_cols = [
        POWER_COL,
        "power_abs_W",
    ]
    cols.extend(optional_cols)

    # remove duplicates while preserving order
    seen = set()
    unique_cols = []
    for col in cols:
        if col not in seen:
            unique_cols.append(col)
            seen.add(col)

    return unique_cols


def build_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Run the complete feature-engineering pipeline and clean up the result.

    The function calls all individual engineering steps in the intended order,
    then determines the relevant feature list and removes rows with gaps that are
    not acceptable for the core data. Special cases such as load-normalised
    features during rest phases are handled separately, so these features do not
    thin out the dataset unnecessarily.

    Returns:
    - engineered dataframe
    - feature column list

    Call context:
    Called in `src.main.extract` after preprocessing.
    """
    df = add_dt_feature(df)
    df = add_current_voltage_features(df)
    df = add_operating_mode_features(df)
    df = add_voltage_spread_features(df)
    df = add_pack_temperature_features(df)
    df = add_temperature_dynamics(df)
    df = add_load_normalized_features(df)
    df = add_rolling_features(df)

    feature_cols = [c for c in get_feature_columns() if c in df.columns]

    # These ratio features are allowed to be NaN during rest / near-zero load.
    ratio_like_cols = {
        "volt_spread_per_current",
        "pack_temp_spread_per_current",
        "pack_dev_from_median_per_current",
    }

    core_required_cols = [c for c in feature_cols if c not in ratio_like_cols]

    df_feat = df.dropna(subset=core_required_cols).copy()

    for col in ratio_like_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna(0.0)

    print("Rows after feature eng:", len(df_feat))
    print("Feature columns:", feature_cols)

    return df_feat, feature_cols
