from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import shutil


def build_archive_filename(original_name: str, timestamp: str) -> str:
    """
    Erzeugt aus einem Dateinamen eine archivgeeignete, zeitgestempelte Variante.

    Dabei bleibt die urspruengliche Dateiendung erhalten, waehrend der
    Zeitstempel unmittelbar vor der Endung eingefuegt wird. So koennen mehrere
    Durchlaeufe desselben Pipelineschritts archiviert werden, ohne dass
    Namenskollisionen entstehen.

    Aufrufkontext:
    Interne Hilfsfunktion fuer die Kopier- und Archivierungsroutinen dieses
    Moduls.

    Example:
        anomaly_results_all.csv
    ->
        anomaly_results_all_2026-04-08_11-42-10.csv
    """
    p = Path(original_name)
    return f"{p.stem}_{timestamp}{p.suffix}"


def copy_folder_contents_with_timestamp(src: Path, dst: Path, timestamp: str) -> None:
    """
    Kopiert den Inhalt eines Quellordners rekursiv in den Archivbereich.

    Fuer jede gefundene Datei wird ein neuer Name mit Zeitstempel erzeugt, damit
    die Archivstruktur historisierte Artefakte enthaelt und keine bestehende
    Datei ueberschrieben wird. Unterordner werden beibehalten und ebenfalls
    rekursiv verarbeitet.

    Aufrufkontext:
    Diese Funktion wird von `archive_run_outputs` verwendet, um die temporaeren
    Ergebnisordner eines Pipeline-Laufs sicher in das Archiv zu ueberfuehren.
    """
    if not src.exists():
        return

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.is_dir():
            sub_dst = dst / item.name
            copy_folder_contents_with_timestamp(item, sub_dst, timestamp)
        else:
            new_name = build_archive_filename(item.name, timestamp)
            shutil.copy2(item, dst / new_name)


def clear_folder_contents(folder: Path) -> None:
    """
    Entfernt alle Inhalte eines Arbeitsordners, laesst den Ordner selbst aber bestehen.

    Diese Funktion wird genutzt, um nach erfolgreicher Archivierung die
    transienten Laufartefakte zu loeschen. Dadurch startet der naechste Lauf mit
    einer sauberen Arbeitsumgebung, ohne dass Pfade oder Verzeichnisstruktur neu
    angelegt werden muessen.

    Aufrufkontext:
    Wird am Ende von `archive_run_outputs` fuer temporaere Output-Ordner
    ausgefuehrt.
    """
    if not folder.exists():
        return

    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def count_files(folder: Path) -> int:
    """
    Zaehlt alle Dateien in einem Ordner inklusive aller Unterordner.

    Die Rueckgabe dient nicht der Fachlogik, sondern der Plausibilisierung des
    Archivierungsprozesses. Vor und nach dem Kopieren kann damit geprueft werden,
    ob die erwartete Anzahl an Dateien wirklich uebernommen wurde.

    Aufrufkontext:
    Wird in `archive_run_outputs` fuer eine einfache Verifikationspruefung
    eingesetzt.
    """
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file())


def build_file_info(path: Path) -> dict | None:
    """
    Sammelt Basis-Metadaten zu einer einzelnen Datei.

    Falls die Datei existiert, werden Name, voller Pfad, Dateigroesse und
    Zeitstempel der letzten Aenderung in einem Dictionary zurueckgegeben. Fehlt
    die Datei, liefert die Funktion `None`. Damit kann die Archiv-Metadatei
    spaeter auch nachvollziehbar dokumentieren, welche Eingabedateien zu einem
    Lauf gehoerten.

    Aufrufkontext:
    Wird in `archive_run_outputs` beim Schreiben der `metadata.json` genutzt.
    """
    if not path.exists():
        return None

    stat = path.stat()
    return {
        "name": path.name,
        "full_path": str(path),
        "size_bytes": stat.st_size,
        "modified_time_epoch": stat.st_mtime,
    }


def archive_run_outputs(
    models_dir: Path,
    processed_feature_file: Path,
    archive_base_dir: Path,
    run_label: str | None = None,
    extra_metadata: dict | None = None,
) -> Path:
    """
    Archiviert die Ergebnisartefakte eines abgeschlossenen Pipeline-Laufs.

    Die Funktion erzeugt zuerst ein neues Laufverzeichnis mit Zeitstempel,
    kopiert relevante Ergebnisordner hinein, legt bei Bedarf eine Snapshot-Kopie
    der verarbeiteten Feature-Datei ab und schreibt anschliessend eine
    `metadata.json` mit Zusatzinformationen zum Lauf. Danach wird ueber einen
    Dateizaehler grob geprueft, ob die Anzahl der archivierten Dateien zur
    Quellmenge passt. Erst wenn diese Pruefung erfolgreich war, werden die
    temporaeren Arbeitsordner geleert.

    Der Rueckgabewert ist der Pfad auf das erzeugte Archivverzeichnis des Runs.

    Aufrufkontext:
    Diese Funktion wird von `src.main.archive_outputs` nach erfolgreicher
    Anomalie-Erkennung und Ausgabeerzeugung aufgerufen.

    Archives:
      - models_dir
      - excel_dir
      - snapshot of processed_feature_file (copy only)

    Clears only:
      - models_dir
      - excel_dir

    Notes:
      - Archived filenames get a timestamp suffix to avoid Excel filename clashes.
      - The processed feature file remains in place and is NOT deleted.
      - run_label is optional and stored only in metadata if provided.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    archive_run_dir = archive_base_dir / timestamp
    archive_models_dir = archive_run_dir / "models"
    #archive_excel_dir = archive_run_dir / "Excel_view"
    archive_processed_snapshot_dir = archive_run_dir / "processed_snapshot"

    archive_run_dir.mkdir(parents=True, exist_ok=True)

    source_counts = {
        "models_files": count_files(models_dir),
        #"excel_files": count_files(excel_dir),
    }

    copy_folder_contents_with_timestamp(models_dir, archive_models_dir, timestamp)
    #copy_folder_contents_with_timestamp(excel_dir, archive_excel_dir, timestamp)

    processed_snapshot_copied = False
    processed_snapshot_archived_name = None

    if processed_feature_file.exists():
        archive_processed_snapshot_dir.mkdir(parents=True, exist_ok=True)
        processed_snapshot_archived_name = build_archive_filename(
            processed_feature_file.name, timestamp
        )
        shutil.copy2(
            processed_feature_file,
            archive_processed_snapshot_dir / processed_snapshot_archived_name,
        )
        processed_snapshot_copied = True

    archived_counts = {
        "models_files": count_files(archive_models_dir),
        #"excel_files": count_files(archive_excel_dir),
    }

    if source_counts != archived_counts:
        raise RuntimeError(
            "Archive verification failed: source and archive file counts differ.\n"
            f"Source: {source_counts}\n"
            f"Archive: {archived_counts}"
        )

    metadata = {
        "timestamp": timestamp,
        "archive_dir": str(archive_run_dir),
        "run_label": run_label,
        "archived_sources": {
            "models_dir": str(models_dir),
            #"excel_dir": str(excel_dir),
        },
        "file_counts": archived_counts,
        "processed_feature_snapshot_copied": processed_snapshot_copied,
        "processed_feature_snapshot_archived_name": processed_snapshot_archived_name,
        "processed_feature_info": build_file_info(processed_feature_file),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = archive_run_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    latest_run_file = archive_base_dir / "latest_run.txt"
    latest_run_file.write_text(str(archive_run_dir), encoding="utf-8")

    clear_folder_contents(models_dir)
    #clear_folder_contents(excel_dir)

    return archive_run_dir
