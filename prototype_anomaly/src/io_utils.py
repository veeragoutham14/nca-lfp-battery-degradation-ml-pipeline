from pathlib import Path

import pandas as pd

from src.config import (
    ANOMALY_ALL_OUTPUT_FILENAME,
    ANOMALY_ONLY_OUTPUT_FILENAME,
    ANOMALY_TOP20_OUTPUT_FILENAME,
    CSV_GLOB,
    FEATURE_OUTPUT_FILENAME,
    TIMESTAMP_FORMAT,
    TS_COL,
)


def read_logger_csv(path: Path) -> pd.DataFrame:
    """Laedt eine einzelne rohe Logger-CSV im erwarteten Exportformat.

    Die Funktion beruecksichtigt dabei die projektspezifische CSV-Struktur mit
    Kopfzeilenversatz, Semikolon-Trennung und Komma als Dezimaltrennzeichen.
    Danach werden die Spaltennamen bereinigt und der Name der Quelldatei in
    einer eigenen Spalte gespeichert, damit spaeter nachvollziehbar bleibt, aus
    welcher Datei eine Zeile stammt.

    Aufrufkontext:
    Diese Funktion wird von `load_all_csvs` pro gefundener Datei aufgerufen.
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
    """Laedt alle Rohdaten-CSV-Dateien aus einem Verzeichnis und fuegt sie zusammen.

    Jede Datei wird einzeln ueber `read_logger_csv` eingelesen. Erfolgreich
    geladene Dateien werden gesammelt, fehlerhafte Dateien nur protokolliert,
    damit der Gesamtlauf moeglichst robust bleibt. Am Ende werden alle gueltigen
    Teil-DataFrames untereinander konkateniert und als gemeinsamer Rohdatensatz
    zurueckgegeben.

    Aufrufkontext:
    Dies ist der erste Datenlade-Schritt in `src.main.extract`.
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
    """Speichert die aufbereiteten Feature-Daten als verarbeitete CSV-Datei.

    Die Funktion schreibt den uebergebenen DataFrame mit den projektspezifischen
    CSV-Einstellungen in das Zielverzeichnis und gibt den finalen Dateipfad
    zurueck. Damit steht die Datei spaeter fuer Visualisierung,
    Anomalieerkennung oder Archivierung konsistent zur Verfuegung.

    Aufrufkontext:
    Wird in `src.main.extract` nach Abschluss von `build_feature_table`
    aufgerufen.
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
    """Laedt die bereits erzeugte Feature-Tabelle wieder aus dem Dateisystem.

    Zusaetzlich wird die Timestamp-Spalte, falls vorhanden, erneut in ein
    Datumsformat ueberfuehrt. Die Funktion bildet damit die standardisierte
    Lade-Schnittstelle fuer alle Pipeline-Schritte, die nicht mit Rohdaten,
    sondern mit bereits verarbeiteten Features arbeiten.

    Aufrufkontext:
    Wird von `src.main.visualize` und `src.main.detect_anomalies` verwendet.
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
    """Schreibt die drei zentralen Ergebnisdateien der Anomalieerkennung auf Disk.

    Gespeichert werden der komplette bewertete Datensatz, die reine
    Anomalie-Teilmenge sowie eine kompakte Top-N-Auswahl der auffaelligsten
    Faelle. Alle Dateien werden mit einheitlichem CSV-Format geschrieben, damit
    sie spaeter unkompliziert weiterverwendet, verglichen oder archiviert werden
    koennen.

    Aufrufkontext:
    Die Funktion wird in `src.main.detect_anomalies` nach Modellbewertung und
    Erklaerungsanreicherung ausgefuehrt.
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
    """Laedt eine zuvor gespeicherte Ergebnisdatei der Anomalieerkennung.

    Diese Hilfsfunktion ist fuer spaetere Analyse-, Notebook- oder
    Visualisierungszwecke gedacht. Falls die Timestamp-Spalte vorhanden ist,
    wird sie beim Einlesen direkt in ein Datumsformat konvertiert, damit der
    Datensatz ohne weitere Vorbereitung weiterverarbeitet werden kann.

    Aufrufkontext:
    Gedacht als allgemeine Hilfsfunktion fuer nachgelagerte Auswertungen auf den
    von `save_anomaly_tables` erzeugten Dateien.
    """
    if not path.exists():
        raise FileNotFoundError(f"Anomaly file not found: {path}")

    df = pd.read_csv(path, sep=";", decimal=",")

    if TS_COL in df.columns:
        df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")

    print("Loaded anomaly table:", path)
    print("Shape:", df.shape)
    return df
