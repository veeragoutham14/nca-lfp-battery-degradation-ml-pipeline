from pathlib import Path

import pandas as pd

INPUT_DIR = Path(r"data/models")
EXCEL_DIR = Path(r"data/Excel_view")


def get_smallest_csv(csv_files: list[Path]) -> list[Path]:
    """
    Pick the smallest file from a list of CSV files.

    This is useful when a quick visual check or a first Excel export should not
    process every file. Instead of returning a single `Path` instance, the
    function deliberately returns a list again, so the rest of the flow in
    `export_csvs_to_excel` can stay unchanged.

    Call context:
    Used in `export_csvs_to_excel` when the `smallest` mode is selected.
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
    Convert CSV result files into Excel files for manual analysis.

    The function searches an input directory for CSV files, decides based on the
    mode whether all files or only the smallest one should be processed, and then
    writes one `.xlsx` file per loaded CSV into the target directory. Errors while
    reading or writing individual files are logged, so one problem does not abort
    the whole export run.

    Call context:
    Run through the script block at the end of the file; it is not part of the
    normal main pipeline step in `src.main`.

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
