from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


def build_archive_filename(original_name: str, timestamp: str) -> str:
    """
    Create an archive-ready, timestamped variant of a file name.

    The original file extension is kept, and the timestamp is inserted directly
    before the extension. This way several runs of the same pipeline step can be
    archived without name collisions.

    Call context:
    Internal helper for the copy and archiving routines in this module.

    Example:
        anomaly_results_all.csv
    ->
        anomaly_results_all_2026-04-08_11-42-10.csv
    """
    p = Path(original_name)
    return f"{p.stem}_{timestamp}{p.suffix}"


def copy_folder_contents_with_timestamp(src: Path, dst: Path, timestamp: str) -> None:
    """
    Recursively copy the contents of a source folder into the archive area.

    Every file found gets a new name with a timestamp, so the archive structure
    holds versioned artefacts and no existing file is overwritten. Subfolders are
    kept and processed recursively as well.

    Call context:
    Used by `archive_run_outputs` to move the temporary result folders of a
    pipeline run safely into the archive.
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
    Remove all contents of a working folder but keep the folder itself.

    Used to delete the transient run artefacts after a successful archive step.
    The next run then starts with a clean working area without having to
    recreate paths or the directory structure.

    Call context:
    Runs at the end of `archive_run_outputs` for the temporary output folders.
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
    Count all files in a folder, including all subfolders.

    The return value is not part of the domain logic; it is a plausibility check
    for the archiving process. It lets the caller check before and after copying
    whether the expected number of files was really transferred.

    Call context:
    Used in `archive_run_outputs` for a simple verification check.
    """
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file())


def build_file_info(path: Path) -> dict | None:
    """
    Collect basic metadata about a single file.

    If the file exists, its name, full path, file size and last-modified
    timestamp are returned in a dictionary. If the file is missing, the function
    returns `None`. This lets the archive metadata file record which input files
    belonged to a run.

    Call context:
    Used in `archive_run_outputs` when writing `metadata.json`.
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
    Archive the result artefacts of a finished pipeline run.

    The function first creates a new timestamped run directory, copies the
    relevant result folders into it, stores a snapshot copy of the processed
    feature file if needed, and then writes a `metadata.json` with additional
    information about the run. A file counter then roughly checks whether the
    number of archived files matches the source. Only if this check passes are
    the temporary working folders cleared.

    The return value is the path to the archive directory created for the run.

    Call context:
    Called by `src.main.archive_outputs` after anomaly detection and output
    generation have finished successfully.

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
