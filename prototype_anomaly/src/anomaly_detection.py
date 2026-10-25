import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import ANOMALY_CONTAMINATION, RANDOM_STATE
from src.anomaly_explainer import add_anomaly_reasons
from src.config import ANOMALY_FEATURES



def get_available_anomaly_features(df: pd.DataFrame) -> list[str]:
    """Ermittelt die tatsaechlich nutzbaren Anomalie-Features im uebergebenen DataFrame.

    Die Funktion nimmt die im Feature-Engineering zentral definierte Soll-Liste
    aller Modell-Features und prueft fuer jedes Feature, ob die entsprechende
    Spalte im aktuellen Datensatz wirklich vorhanden ist. Dadurch wird
    verhindert, dass spaetere Modellschritte mit fehlenden Spalten laufen und
    dabei schwer lesbare Fehler erzeugen.

    Aufrufkontext:
    Diese Funktion wird in `run_isolation_forest` verwendet, um die
    Modell-Eingaben exakt an den aktuell erzeugten Feature-Stand anzupassen.
    """
    return [c for c in ANOMALY_FEATURES if c in df.columns]


def run_isolation_forest(
    df: pd.DataFrame,
    contamination: float = ANOMALY_CONTAMINATION,
) -> pd.DataFrame:
    """Fuehrt die eigentliche Anomalieerkennung mit einem Isolation-Forest-Modell aus.

    Zuerst werden alle fuer die Erkennung verfuegbaren Feature-Spalten
    ausgewaehlt und in eine numerische Matrix ueberfuehrt. Diese Matrix wird
    standardisiert, damit Merkmale mit unterschiedlichen Skalen das Modell nicht
    ungewollt verzerren. Anschliessend wird ein `IsolationForest` trainiert und
    direkt auf denselben Daten ausgefuehrt, um fuer jede Zeile ein Label
    (`anomaly`) sowie einen numerischen Score (`anomaly_score`) zu berechnen.

    Nach der Modellbewertung wird der DataFrame nicht nur mit den rohen
    Anomalie-Ergebnissen erweitert, sondern auch mit textuellen und
    strukturierteren Erklaerungs-Spalten aus `add_anomaly_reasons`. Das
    Rueckgabeobjekt ist deshalb der zentrale, voll angereicherte Datensatz fuer
    Export, Analyse und Visualisierung.

    Aufrufkontext:
    Diese Funktion wird von `src.main.detect_anomalies` als zentraler
    Modellschritt der Pipeline aufgerufen.
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
    """Filtert aus einem bereits bewerteten Datensatz nur die Anomalie-Zeilen heraus.

    Erwartet wird ein DataFrame, in dem die Spalte `anomaly` bereits vorhanden
    ist. Genau diese Spalte wird genutzt, um alle Zeilen mit dem vom Modell
    vergebenen Label `-1` zu extrahieren. Das Ergebnis ist eine Kopie des
    gefilterten Ausschnitts und kann deshalb gefahrlos weiterverarbeitet oder
    gespeichert werden.

    Aufrufkontext:
    Diese Funktion wird in `src.main.detect_anomalies` verwendet, bevor die
    anomaly-only CSV-Datei geschrieben wird.
    """
    if "anomaly" not in df.columns:
        raise ValueError("No anomaly column found. Run anomaly detection first.")
    return df[df["anomaly"] == -1].copy()


def get_top_anomalies(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Liefert die staerksten Anomalie-Kandidaten anhand des Modell-Scores zurueck.

    Die Funktion sortiert den Datensatz nach `anomaly_score` aufsteigend. Bei
    einem Isolation-Forest bedeuten niedrigere Werte in diesem Kontext staerkere
    Abweichungen vom normalen Muster. Anschliessend werden nur die ersten `n`
    Zeilen uebernommen, damit Entwickler oder Analysten schnell die
    auffaelligsten Faelle inspizieren koennen.

    Aufrufkontext:
    Diese Funktion wird in `src.main.detect_anomalies` genutzt, um die kompakte
    Top-N-Ausgabe fuer die manuelle Sichtpruefung zu erzeugen.
    """
    if "anomaly_score" not in df.columns:
        raise ValueError("No anomaly_score column found. Run anomaly detection first.")
    return df.sort_values("anomaly_score", ascending=True).head(n).copy()
