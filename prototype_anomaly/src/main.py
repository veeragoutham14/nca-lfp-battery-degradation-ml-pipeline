import pandas as pd

from src.config import (
    DATA_DIR,
    OUT_DIR,
    ANOMALY_OUT_DIR,
    ARCHIVE_BASE_DIR,
    FEATURE_OUTPUT_FILENAME,
)
from src.io_utils import (
    load_all_csvs,
    save_feature_table,
    load_feature_table,
    save_anomaly_tables,
)
from src.preprocessing import prepare_base_dataframe
from src.feature_engineering import build_feature_table
from src.visualization import run_all_visual_checks, plot_anomalies
from src.anomaly_detection import (
    run_isolation_forest,
    get_all_anomalies,
    get_top_anomalies,
    get_available_anomaly_features,
)
from src.archive_utils import archive_run_outputs


def extract() -> None:
    """Fuehrt die komplette Rohdatenaufbereitung bis zur gespeicherten Feature-Datei aus.

    Der Ablauf umfasst das Einlesen aller Roh-CSV-Dateien, das Preprocessing,
    das anschliessende Feature Engineering und das Schreiben der finalen
    Feature-Tabelle in den Processed-Bereich. Diese Funktion bildet damit den
    Einstiegspunkt fuer alle Faelle, in denen sich die zugrunde liegenden
    Rohdaten geaendert haben.

    Aufrufkontext:
    Kann direkt aus `main()` aktiviert werden und dient als vorgelagerter
    Schritt vor Visualisierung oder Anomalieerkennung.
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
    """Laedt die Feature-Tabelle und startet die grafischen Plausibilitaetspruefungen.

    Diese Funktion dient nicht der eigentlichen Modellberechnung, sondern der
    qualitativen Kontrolle der erzeugten Features. Entwickler koennen damit
    zeitliche Verlaeufe, Verteilungen und Zusammenhaenge inspizieren, bevor sie
    die Features produktiv fuer die Anomalieerkennung verwenden.

    Aufrufkontext:
    Optionale Pipeline-Funktion, die aus `main()` aktiviert werden kann.
    """
    print("\n--- Loading processed feature table ---")
    df_feat = load_feature_table(OUT_DIR)

    print("\n--- Running visualization checks ---")
    run_all_visual_checks(df_feat)


def detect_anomalies() -> None:
    """Fuehrt den Kernschritt der Anomalieerkennung auf der Feature-Tabelle aus.

    Zuerst wird die bereits erzeugte Feature-Datei geladen. Anschliessend wird
    der Datensatz mit dem Isolation-Forest-Modell bewertet, in reine
    Anomalie-Zeilen und eine Top-Auswahl aufgeteilt und schliesslich in mehrere
    Ergebnisdateien geschrieben. Optional koennen danach auch Visualisierungen
    einzelner Anomalie-Features erzeugt werden.

    Aufrufkontext:
    Dies ist der zentrale Laufzeitschritt, der aktuell in `main()`
    standardmaessig aktiviert ist.
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
    """Archiviert die erzeugten Laufartefakte nach einem erfolgreichen Durchlauf.

    Die Funktion uebergibt die relevanten Verzeichnisse und Dateien an die
    Archivlogik und sorgt dafuer, dass temporaere Modellergebnisse sicher
    weggeschrieben werden, waehrend die verarbeitete Feature-Datei am gewohnten
    Ort bestehen bleibt. Das ist besonders hilfreich, wenn mehrere Runs
    miteinander verglichen oder historisch dokumentiert werden sollen.

    Aufrufkontext:
    Wird typischerweise in `main()` nach `detect_anomalies` aufgerufen.
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
    """Steuert die aktuell aktivierten Top-Level-Schritte der Pipeline.

    In dieser Funktion wird festgelegt, welche Phasen eines Laufs wirklich
    ausgefuehrt werden, zum Beispiel nur Anomalieerkennung oder zusaetzlich auch
    Extraktion, Visualisierung und Archivierung. Ausserdem kapselt sie das
    Fehlerhandling auf oberster Ebene, damit bei Problemen der Grund fuer den
    Abbruch klar ausgegeben wird.

    Aufrufkontext:
    Der `__main__`-Block am Dateiende ruft diese Funktion auf, wenn das Modul
    als Skript gestartet wird.
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
