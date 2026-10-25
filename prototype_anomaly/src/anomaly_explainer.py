import numpy as np
import pandas as pd

from src.config import ANOMALY_FEATURES 

# =========================================================
# EXPLAINER CONFIG
# =========================================================
MIN_MODE_ROWS = 30

# If MAD-derived scale is smaller than this, the feature is considered
# too stable / too close to constant for reliable explanation ranking.
MIN_EXPLAINER_SCALE = 1e-4

# Clip robust z so near-zero-scale features cannot dominate everything.
ROBUST_Z_CLIP = 10.0

# Combined ranking score weights
Z_WEIGHT = 0.7
RELATIVE_DELTA_WEIGHT = 0.3

# Prevent divide-by-zero in relative delta normalization
RELATIVE_CENTER_FLOOR = 1e-3


# =========================================================
# MODE HELPERS
# =========================================================
def get_row_mode(row: pd.Series) -> str:
    """
    Determine battery mode from one-hot mode columns.
    """
    if "is_rest" in row.index and row["is_rest"] == 1:
        return "rest"
    if "is_charging" in row.index and row["is_charging"] == 1:
        return "charging"
    if "is_discharging" in row.index and row["is_discharging"] == 1:
        return "discharging"
    return "unknown"


def get_battery_mode_series(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized battery mode inference from one-hot mode columns.
    """
    return pd.Series(
        np.select(
            [
                ("is_rest" in df.columns) & (df["is_rest"] == 1),
                ("is_charging" in df.columns) & (df["is_charging"] == 1),
                ("is_discharging" in df.columns) & (df["is_discharging"] == 1),
            ],
            ["rest", "charging", "discharging"],
            default="unknown",
        ),
        index=df.index,
    )


def get_mode_normal_subset(
    df_normal: pd.DataFrame,
    row: pd.Series,
    min_rows: int = MIN_MODE_ROWS,
) -> pd.DataFrame:
    """
    Return normal rows matching the anomaly row's mode.
    Falls back to all normal rows if mode-specific subset is too small.
    """
    mode = get_row_mode(row)

    if mode == "rest" and "is_rest" in df_normal.columns:
        subset = df_normal[df_normal["is_rest"] == 1]
    elif mode == "charging" and "is_charging" in df_normal.columns:
        subset = df_normal[df_normal["is_charging"] == 1]
    elif mode == "discharging" and "is_discharging" in df_normal.columns:
        subset = df_normal[df_normal["is_discharging"] == 1]
    else:
        subset = df_normal

    if len(subset) < min_rows:
        return df_normal

    return subset


def precompute_mode_baselines(
    df_normal: pd.DataFrame,
    feature_cols: list[str],
    min_mode_rows: int = MIN_MODE_ROWS,
    min_scale: float = MIN_EXPLAINER_SCALE,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """
    Precompute robust baselines once for all supported modes and a fallback.
    """
    fallback = compute_robust_baseline(
        df_normal=df_normal,
        feature_cols=feature_cols,
        min_scale=min_scale,
    )
    baselines = {"fallback": fallback}

    for mode_name, mode_col in (
        ("rest", "is_rest"),
        ("charging", "is_charging"),
        ("discharging", "is_discharging"),
    ):
        if mode_col not in df_normal.columns:
            baselines[mode_name] = fallback
            continue

        subset = df_normal[df_normal[mode_col] == 1]
        if len(subset) < min_mode_rows:
            baselines[mode_name] = fallback
            continue

        baselines[mode_name] = compute_robust_baseline(
            df_normal=subset,
            feature_cols=feature_cols,
            min_scale=min_scale,
        )

    baselines["unknown"] = fallback
    return baselines


def get_baseline_for_mode(
    mode: str,
    mode_baselines: dict[str, tuple[pd.Series, pd.Series]],
) -> tuple[pd.Series, pd.Series]:
    """
    Return the precomputed baseline for a mode or the shared fallback.
    """
    return mode_baselines.get(mode, mode_baselines["fallback"])


# =========================================================
# ROBUST STATISTICS
# =========================================================
def _robust_center(x: pd.Series) -> float:
    return float(x.median())


def _robust_mad(x: pd.Series) -> float:
    """
    Median absolute deviation.
    """
    med = x.median()
    mad = np.median(np.abs(x - med))
    return float(mad)


def _robust_scale_from_mad(
    x: pd.Series,
    min_scale: float = MIN_EXPLAINER_SCALE,
) -> float:
    """
    Convert MAD to a robust sigma estimate.
    1.4826 * MAD approximates std for Gaussian data.

    If the scale is too small, return NaN so the feature is ignored
    in explanation ranking rather than exploding numerically.
    """
    mad = _robust_mad(x)
    scale = 1.4826 * mad

    if scale < min_scale:
        return np.nan

    return float(scale)


def compute_robust_baseline(
    df_normal: pd.DataFrame,
    feature_cols: list[str],
    min_scale: float = MIN_EXPLAINER_SCALE,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute robust center and robust scale for each feature.
    """
    centers = {}
    scales = {}

    for feat in feature_cols:
        s = pd.to_numeric(df_normal[feat], errors="coerce").dropna()

        if s.empty:
            centers[feat] = np.nan
            scales[feat] = np.nan
            continue

        centers[feat] = _robust_center(s)
        scales[feat] = _robust_scale_from_mad(s, min_scale=min_scale)

    return pd.Series(centers), pd.Series(scales)


def clip_robust_z(z: pd.Series, clip_value: float = ROBUST_Z_CLIP) -> pd.Series:
    """
    Clip robust z-scores so near-zero-scale artifacts do not dominate.
    """
    return z.clip(lower=-clip_value, upper=clip_value)


def compute_relative_delta_score(
    delta: float,
    center: float,
    floor: float = RELATIVE_CENTER_FLOOR,
) -> float:
    """
    Compute a unitless relative deviation score.

    This complements robust z by rewarding physically meaningful shifts,
    especially when a feature's center is far from zero.
    """
    if pd.isna(delta) or pd.isna(center):
        return np.nan

    denom = max(abs(float(center)), floor)
    return abs(float(delta)) / denom


def compute_ranking_score(
    robust_z_clipped: float,
    relative_delta_score: float,
    z_weight: float = Z_WEIGHT,
    relative_delta_weight: float = RELATIVE_DELTA_WEIGHT,
) -> float:
    """
    Combine clipped robust z and relative delta into one explanation ranking score.
    """
    z_part = 0.0 if pd.isna(robust_z_clipped) else abs(float(robust_z_clipped))
    d_part = 0.0 if pd.isna(relative_delta_score) else float(relative_delta_score)
    return z_weight * z_part + relative_delta_weight * d_part


# =========================================================
# FEATURE FAMILY HELPERS
# =========================================================
def get_feature_families() -> dict[str, set[str]]:
    return {
        "pack_imbalance": {
            "pack_temp_spread_C",
            "pack_temp_std_C",
            "pack_max_dev_from_median_C",
            "max_intra_module_pack_delta_C",
        },
        "internal_pack_nonuniformity": {
            "max_pack_internal_spread_C",
            "mean_pack_internal_spread_C",
            "max_pack_internal_std_C",
            "mean_pack_internal_std_C",
            "max_pack_hotspot_delta_C",
            "mean_pack_hotspot_delta_C",
        },
        "persistent_thermal_imbalance": {
            "pack_temp_spread_mean_120s",
            "pack_temp_std_mean_120s",
            "pack_max_dev_from_median_mean_120s",
            "max_intra_module_pack_delta_mean_120s",
        },
        "electrical_imbalance": {
            "cell_voltage_spread_V",
            "Vspread_mean_120s",
        },
    }


def family_of_feature(feature: str) -> str:
    families = get_feature_families()
    for family_name, feats in families.items():
        if feature in feats:
            return family_name
    return "other"


def aggregate_family_scores(feature_summaries: list[dict]) -> dict[str, float]:
    """
    Aggregate explanation score by family.
    Uses ranking_score if available, otherwise falls back to abs(robust_z).
    """
    scores = {k: 0.0 for k in get_feature_families().keys()}
    scores["other"] = 0.0

    for item in feature_summaries:
        feat = item["feature"]

        if "ranking_score" in item and pd.notna(item["ranking_score"]):
            score_value = float(item["ranking_score"])
        else:
            z = item["robust_z"]
            if pd.isna(z):
                continue
            score_value = abs(float(z))

        fam = family_of_feature(feat)
        scores[fam] += score_value

    return scores


# =========================================================
# DIAGNOSTIC LOCALIZATION
# =========================================================
def extract_pack_diagnostics(row: pd.Series) -> dict[str, object]:
    """
    Optional diagnostic localization from engineered per-pack columns.
    This does NOT affect anomaly scoring.
    """
    diagnostics: dict[str, object] = {
        "worst_pack_by_internal_spread": None,
        "worst_pack_internal_spread_C": np.nan,
        "worst_pack_by_internal_std": None,
        "worst_pack_internal_std_C": np.nan,
        "worst_pack_by_hotspot_delta": None,
        "worst_pack_hotspot_delta_C": np.nan,
        "hottest_pack_by_mean": None,
        "hottest_pack_temp_mean_C": np.nan,
        "coldest_pack_by_mean": None,
        "coldest_pack_temp_mean_C": np.nan,
    }

    spread_candidates = {}
    std_candidates = {}
    hotspot_candidates = {}
    mean_candidates = {}

    for col in row.index:
        if col.endswith("_temp_spread_C") and col.startswith("mod"):
            spread_candidates[col.replace("_temp_spread_C", "")] = row[col]
        elif col.endswith("_temp_std_C") and col.startswith("mod"):
            std_candidates[col.replace("_temp_std_C", "")] = row[col]
        elif col.endswith("_hotspot_delta_C") and col.startswith("mod"):
            hotspot_candidates[col.replace("_hotspot_delta_C", "")] = row[col]
        elif col.endswith("_temp_mean_C") and col.startswith("mod"):
            mean_candidates[col.replace("_temp_mean_C", "")] = row[col]

    if spread_candidates:
        worst_pack = max(spread_candidates, key=spread_candidates.get)
        diagnostics["worst_pack_by_internal_spread"] = worst_pack
        diagnostics["worst_pack_internal_spread_C"] = spread_candidates[worst_pack]

    if std_candidates:
        worst_pack = max(std_candidates, key=std_candidates.get)
        diagnostics["worst_pack_by_internal_std"] = worst_pack
        diagnostics["worst_pack_internal_std_C"] = std_candidates[worst_pack]

    if hotspot_candidates:
        worst_pack = max(hotspot_candidates, key=hotspot_candidates.get)
        diagnostics["worst_pack_by_hotspot_delta"] = worst_pack
        diagnostics["worst_pack_hotspot_delta_C"] = hotspot_candidates[worst_pack]

    if mean_candidates:
        hottest_pack = max(mean_candidates, key=mean_candidates.get)
        coldest_pack = min(mean_candidates, key=mean_candidates.get)
        diagnostics["hottest_pack_by_mean"] = hottest_pack
        diagnostics["hottest_pack_temp_mean_C"] = mean_candidates[hottest_pack]
        diagnostics["coldest_pack_by_mean"] = coldest_pack
        diagnostics["coldest_pack_temp_mean_C"] = mean_candidates[coldest_pack]

    return diagnostics


# =========================================================
# EXPLANATION TEXT HELPERS
# =========================================================
def build_reason_text(item: dict) -> str:
    feat = item["feature"]
    value = item["value"]
    center = item["normal_center"]
    delta = item["delta"]
    rz = item["robust_z"]
    rz_clipped = item.get("robust_z_clipped", np.nan)
    rel_delta = item.get("relative_delta_score", np.nan)
    ranking_score = item.get("ranking_score", np.nan)
    direction = item["direction"]

    def fmt(x: float) -> str:
        return f"{x:.4f}" if pd.notna(x) else "nan"

    return (
        f"{feat} is {direction}: "
        f"value={fmt(value)}, "
        f"normal_center={fmt(center)}, "
        f"delta={fmt(delta)}, "
        f"robust_z={fmt(rz)}, "
        f"robust_z_clipped={fmt(rz_clipped)}, "
        f"relative_delta_score={fmt(rel_delta)}, "
        f"ranking_score={fmt(ranking_score)}"
    )


def infer_physical_interpretation(
    row: pd.Series,
    feature_summaries: list[dict],
    family_scores: dict[str, float],
    diagnostics: dict[str, object] | None = None,
) -> str:
    """
    Turn robust feature deviations into a physically meaningful explanation.
    """
    if not feature_summaries:
        return "no robust deviation explanation available"

    mode = get_row_mode(row)
    ranked = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
    primary_family, primary_score = ranked[0]
    secondary_family, secondary_score = ranked[1]

    secondary_relevant = secondary_score >= max(1.5, 0.6 * primary_score)

    feature_set = {item["feature"] for item in feature_summaries}

    def has(feat: str) -> bool:
        return feat in feature_set

    if primary_family == "pack_imbalance":
        text = f"anomaly is mainly driven by pack-to-pack thermal imbalance during {mode} operation"
    elif primary_family == "internal_pack_nonuniformity":
        text = f"anomaly is mainly driven by unusually uneven temperature distribution inside one or more packs during {mode} operation"
    elif primary_family == "persistent_thermal_imbalance":
        text = f"anomaly is mainly driven by thermal imbalance that persists over time during {mode} operation"
    elif primary_family == "electrical_imbalance":
        text = f"anomaly is mainly driven by cell-voltage imbalance during {mode} operation"
    else:
        text = f"anomaly is driven by a mixed deviation pattern during {mode} operation"

    if primary_family == "electrical_imbalance":
        if has("cell_voltage_spread_V") and has("Vspread_mean_120s"):
            text = f"anomaly is mainly driven by sustained cell-voltage imbalance during {mode} operation"
        elif has("cell_voltage_spread_V"):
            text = f"anomaly is mainly driven by instantaneous cell-voltage spread during {mode} operation"

    if primary_family == "persistent_thermal_imbalance":
        if has("pack_temp_spread_mean_120s") or has("pack_max_dev_from_median_mean_120s"):
            text = f"anomaly is mainly driven by persistent pack-level thermal separation during {mode} operation"

    if primary_family == "internal_pack_nonuniformity" and diagnostics:
        culprit = diagnostics.get("worst_pack_by_internal_spread")
        if culprit:
            text += f"; strongest internal spread is observed in {culprit}"

    if primary_family == "pack_imbalance" and diagnostics:
        hot = diagnostics.get("hottest_pack_by_mean")
        cold = diagnostics.get("coldest_pack_by_mean")
        if hot and cold:
            text += f"; hottest pack is {hot} and coldest pack is {cold}"

    if secondary_relevant:
        secondary_text = None
        if secondary_family == "pack_imbalance":
            secondary_text = "pack-to-pack imbalance is also elevated"
        elif secondary_family == "internal_pack_nonuniformity":
            secondary_text = "internal pack non-uniformity is also elevated"
        elif secondary_family == "persistent_thermal_imbalance":
            secondary_text = "the deviation is also persistent over time"
        elif secondary_family == "electrical_imbalance":
            secondary_text = "electrical imbalance is also elevated"

        if secondary_text:
            text += f"; {secondary_text}"

    return text


# =========================================================
# MAIN EXPLAINER
# =========================================================
    

def add_anomaly_reasons(
    df: pd.DataFrame,
    feature_cols: list[str],
    anomaly_col: str = "anomaly",
    top_k: int = 4,
    min_mode_rows: int = MIN_MODE_ROWS,
    add_diagnostics: bool = True,
    baseline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add explanation columns directly to the main dataframe.

    Uses:
    - mode-aware baseline
    - robust z-score (median / MAD)
    - clipped robust z
    - combined ranking score
    - family aggregation
    - top-feature detail
    - optional pack diagnostics

    IMPORTANT:
    Explanation scores are computed ONLY from the supplied feature_cols.
    """
    df = df.copy()
    baseline_source = df if baseline_df is None else baseline_df.copy()

    usable_features = [c for c in feature_cols if c in df.columns]
    if not usable_features:
        raise ValueError("No usable feature columns found for explanation.")

    df_normal = baseline_source[baseline_source[anomaly_col] == 1].copy()
    if df_normal.empty:
        raise ValueError("No normal rows found. Cannot build baseline statistics.")

    df["battery_mode"] = get_battery_mode_series(df)
    if "battery_mode" not in baseline_source.columns:
        baseline_source["battery_mode"] = get_battery_mode_series(baseline_source)
    df_normal["battery_mode"] = baseline_source.loc[df_normal.index, "battery_mode"]
    mode_baselines = precompute_mode_baselines(
        df_normal=df_normal,
        feature_cols=usable_features,
        min_mode_rows=min_mode_rows,
        min_scale=MIN_EXPLAINER_SCALE,
    )

    # Top-feature explanation columns
    for i in range(1, top_k + 1):
        df[f"top_feature_{i}"] = None
        df[f"top_anomaly_reason_{i}"] = None
        df[f"top_feature_{i}_robust_z"] = np.nan
        df[f"top_feature_{i}_robust_z_clipped"] = np.nan
        df[f"top_feature_{i}_delta"] = np.nan
        df[f"top_feature_{i}_relative_delta_score"] = np.nan
        df[f"top_feature_{i}_ranking_score"] = np.nan
        df[f"top_feature_{i}_normal_center"] = np.nan
        df[f"top_feature_{i}_family"] = None

    # Family score columns
    for fam in get_feature_families().keys():
        df[f"family_score_{fam}"] = np.nan

    df["interpretation"] = None

    # Optional diagnostics
    text_diagnostic_cols = [
        "worst_pack_by_internal_spread",
        "worst_pack_by_internal_std",
        "worst_pack_by_hotspot_delta",
        "hottest_pack_by_mean",
        "coldest_pack_by_mean",
    ]

    numeric_diagnostic_cols = [
        "worst_pack_internal_spread_C",
        "worst_pack_internal_std_C",
        "worst_pack_hotspot_delta_C",
        "hottest_pack_temp_mean_C",
        "coldest_pack_temp_mean_C",
    ]

    if add_diagnostics:
        for col in text_diagnostic_cols:
            df[col] = None

        for col in numeric_diagnostic_cols:
            df[col] = np.nan

    anomaly_indices = df.index[df[anomaly_col] == -1]
    if len(anomaly_indices) == 0:
        if add_diagnostics:
            for col in numeric_diagnostic_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    for idx in anomaly_indices:
        row = df.loc[idx]

        # -------------------------
        # Robust baseline
        # -------------------------
        normal_center, robust_scale = get_baseline_for_mode(
            mode=row["battery_mode"],
            mode_baselines=mode_baselines,
        )

        robust_z = ((row[usable_features] - normal_center) / robust_scale).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        robust_z_clipped = clip_robust_z(robust_z, clip_value=ROBUST_Z_CLIP)
        deltas = row[usable_features] - normal_center

        ranking_items = []
        for feat in usable_features:
            value = row[feat]
            center = normal_center[feat]
            delta = deltas[feat]
            rz = robust_z[feat]
            rz_clipped = robust_z_clipped[feat]

            if pd.isna(rz_clipped):
                continue

            rel_delta_score = compute_relative_delta_score(delta=delta, center=center)
            ranking_score = compute_ranking_score(
                robust_z_clipped=rz_clipped,
                relative_delta_score=rel_delta_score,
            )

            ranking_items.append(
                {
                    "feature": feat,
                    "value": value,
                    "normal_center": center,
                    "delta": delta,
                    "robust_z": rz,
                    "robust_z_clipped": rz_clipped,
                    "relative_delta_score": rel_delta_score,
                    "ranking_score": ranking_score,
                }
            )

        ranking_items.sort(
            key=lambda item: (-np.inf if pd.isna(item["ranking_score"]) else item["ranking_score"]),
            reverse=True,
        )
        ranking_items = ranking_items[:top_k]

        feature_summaries = []
        for item in ranking_items:
            rz = item["robust_z_clipped"]

            if pd.isna(rz):
                direction = "unusual"
            elif rz > 0:
                direction = "high"
            else:
                direction = "low"

            feature_summaries.append(
                {
                    "feature": item["feature"],
                    "value": item["value"],
                    "normal_center": item["normal_center"],
                    "delta": item["delta"],
                    "robust_z": item["robust_z"],
                    "robust_z_clipped": item["robust_z_clipped"],
                    "relative_delta_score": item["relative_delta_score"],
                    "ranking_score": item["ranking_score"],
                    "direction": direction,
                    "family": family_of_feature(item["feature"]),
                }
            )

        # -------------------------
        # Family aggregation
        # -------------------------
        family_scores = aggregate_family_scores(feature_summaries)

        for fam, score in family_scores.items():
            col = f"family_score_{fam}"
            if col in df.columns:
                df.at[idx, col] = score

        # -------------------------
        # Diagnostics
        # -------------------------
        diagnostics = extract_pack_diagnostics(row) if add_diagnostics else None
        if diagnostics:
            for k, v in diagnostics.items():
                if k in df.columns:
                    df.at[idx, k] = v

        # -------------------------
        # Store top-feature details
        # -------------------------
        for i, item in enumerate(feature_summaries, start=1):
            df.at[idx, f"top_feature_{i}"] = item["feature"]
            df.at[idx, f"top_anomaly_reason_{i}"] = build_reason_text(item)
            df.at[idx, f"top_feature_{i}_robust_z"] = item["robust_z"]
            df.at[idx, f"top_feature_{i}_robust_z_clipped"] = item["robust_z_clipped"]
            df.at[idx, f"top_feature_{i}_delta"] = item["delta"]
            df.at[idx, f"top_feature_{i}_relative_delta_score"] = item["relative_delta_score"]
            df.at[idx, f"top_feature_{i}_ranking_score"] = item["ranking_score"]
            df.at[idx, f"top_feature_{i}_normal_center"] = item["normal_center"]
            df.at[idx, f"top_feature_{i}_family"] = item["family"]

        # -------------------------
        # Final interpretation
        # -------------------------
        df.at[idx, "interpretation"] = infer_physical_interpretation(
            row=row,
            feature_summaries=feature_summaries,
            family_scores=family_scores,
            diagnostics=diagnostics,
        )

    # Force numeric diagnostic columns back to numeric dtype
    if add_diagnostics:
        for col in numeric_diagnostic_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
