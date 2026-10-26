import pandas as pd
from sklearn.ensemble import IsolationForest

from src.anomaly_explainer import add_anomaly_reasons
from src.config import ANOMALY_CONTAMINATION, ANOMALY_FEATURES, RANDOM_STATE


def get_available_anomaly_features(df: pd.DataFrame) -> list[str]:
    """Determine which anomaly features are actually usable in the given DataFrame.

    The function takes the target list of all model features defined centrally in
    feature engineering and checks, for each feature, whether the matching column
    really exists in the current dataset. This prevents later model steps from
    running with missing columns and producing hard-to-read errors.

    Call context:
    Used in `run_isolation_forest` to match the model inputs exactly to the
    feature set that is currently produced.
    """
    return [c for c in ANOMALY_FEATURES if c in df.columns]


def run_isolation_forest(
    df: pd.DataFrame,
    contamination: float = ANOMALY_CONTAMINATION,
) -> pd.DataFrame:
    """Run the actual anomaly detection with an Isolation Forest model.

    First, all feature columns available for detection are selected and turned
    into a numeric matrix. This matrix is standardised so that features with
    different scales do not unintentionally bias the model. Then an
    `IsolationForest` is trained and applied directly to the same data to compute
    a label (`anomaly`) and a numeric score (`anomaly_score`) for every row.

    After scoring, the DataFrame is extended not only with the raw anomaly results
    but also with the textual and more structured explanation columns from
    `add_anomaly_reasons`. The returned object is therefore the central, fully
    enriched dataset for export, analysis and visualisation.

    Call context:
    Called by `src.main.detect_anomalies` as the central model step of the
    pipeline.
    """
    feature_cols = get_available_anomaly_features(df)

    if not feature_cols:
        raise ValueError("No anomaly features found in dataframe.")

    # Isolation Forest is tree-based and does not require standardized inputs.
    # Use float32 to reduce memory bandwidth during fit/score.
    X = df[feature_cols].astype("float32", copy=False)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    df = df.copy()
    df["anomaly"] = model.fit_predict(X)          # -1 anomaly, 1 normal
    df["anomaly_score"] = model.decision_function(X)

    anomaly_mask = df["anomaly"] == -1
    if anomaly_mask.any():
        df_anomalies = add_anomaly_reasons(
            df.loc[anomaly_mask].copy(),
            feature_cols,
            baseline_df=df,
        )
        explanation_cols = [
            col for col in df_anomalies.columns
            if col not in df.columns or col == "battery_mode"
        ]

        for col in explanation_cols:
            df[col] = None

        for col in explanation_cols:
            df.loc[anomaly_mask, col] = df_anomalies[col]

    return df


def get_all_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Filter only the anomaly rows out of an already scored dataset.

    Expects a DataFrame that already contains the `anomaly` column. Exactly this
    column is used to extract all rows that the model labelled `-1`. The result
    is a copy of the filtered slice and can therefore be processed further or
    saved safely.

    Call context:
    Used in `src.main.detect_anomalies` before the anomaly-only CSV file is
    written.
    """
    if "anomaly" not in df.columns:
        raise ValueError("No anomaly column found. Run anomaly detection first.")
    return df[df["anomaly"] == -1].copy()


def get_top_anomalies(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the strongest anomaly candidates based on the model score.

    The function sorts the dataset by `anomaly_score` in ascending order. For an
    Isolation Forest, lower values mean stronger deviations from the normal
    pattern in this context. Only the first `n` rows are kept, so that developers
    or analysts can quickly inspect the most conspicuous cases.

    Call context:
    Used in `src.main.detect_anomalies` to produce the compact top-N output for
    manual review.
    """
    if "anomaly_score" not in df.columns:
        raise ValueError("No anomaly_score column found. Run anomaly detection first.")
    return df.sort_values("anomaly_score", ascending=True).head(n).copy()
