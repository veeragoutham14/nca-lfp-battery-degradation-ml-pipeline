from pathlib import Path
import pandas as pd


INPUT_DIR = Path(r"data/models")
EXCEL_DIR = Path(r"data/Excel_view")


def get_smallest_csv(csv_files: list[Path]) -> list[Path]:
    """
    Waehlt aus einer Liste von CSV-Dateien die kleinste Datei aus.

    Das ist nuetzlich, wenn fuer eine schnelle Sichtpruefung oder einen ersten
    Excel-Export nicht alle Dateien verarbeitet werden sollen. Statt eine
    einzelne `Path`-Instanz zurueckzugeben, liefert die Funktion bewusst wieder
    eine Liste, damit der weitere Verarbeitungsfluss in `export_csvs_to_excel`
    unveraendert bleiben kann.

    Aufrufkontext:
    Wird in `export_csvs_to_excel` verwendet, wenn der Modus `smallest`
    gewaehlt wurde.
    """
    smallest = min(csv_files, key=lambda f: f.stat().st_size)
    print(f"Selected smallest file: {smallest.name} ({smallest.stat().st_size / 1e6:.2f} MB)")
    return [smallest]


def export_csvs_to_excel(
    input_dir: Path,
    output_dir: Path,
    mode: str = "all",  # "all" or "smallest"
) -> None:
    """
    Konvertiert CSV-Ergebnisdateien in Excel-Dateien fuer die manuelle Analyse.

    Die Funktion durchsucht ein Eingabeverzeichnis nach CSV-Dateien, entscheidet
    je nach Modus, ob alle oder nur die kleinste Datei verarbeitet werden soll,
    und schreibt anschliessend fuer jede geladene CSV eine `.xlsx`-Datei in das
    Zielverzeichnis. Fehler beim Einlesen oder Schreiben einzelner Dateien werden
    protokolliert, damit ein Problem nicht den gesamten Exportlauf abbricht.

    Aufrufkontext:
    Diese Funktion wird ueber den Script-Block am Ende der Datei ausgefuehrt und
    ist nicht Bestandteil des normalen Haupt-Pipelineschritts in `src.main`.

    mode:
        - "all" → export all CSVs
        - "smallest" → export only the smallest CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV file(s).")

    # 🔥 Mode selection
    if mode == "smallest":
        csv_files = get_smallest_csv(csv_files)
    elif mode == "all":
        print("Exporting ALL files...")
    else:
        raise ValueError("mode must be 'all' or 'smallest'")

    # 🔁 Export loop
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, sep=";", decimal=",")
            excel_file = output_dir / f"{csv_file.stem}.xlsx"
            df.to_excel(excel_file, index=False)
            print(f"Exported: {excel_file}")
        except Exception as e:
            print(f"Failed to export {csv_file.name}: {e}")

    print("\nDone.")


# 🎮 Main control
if __name__ == "__main__":
    # Change this to:
    # "all" → export everything
    # "smallest" → export only smallest file
    MODE = "all"

    export_csvs_to_excel(INPUT_DIR, EXCEL_DIR, mode=MODE)
