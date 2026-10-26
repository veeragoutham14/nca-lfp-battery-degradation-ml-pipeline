import gzip
import os
import shutil


def extract_all_gz(source_dir: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)

    for file in os.listdir(source_dir):
        if file.endswith(".csv.gz"):
            gz_path = os.path.join(source_dir, file)
            output_path = os.path.join(target_dir, file.replace(".gz", ""))

            print(f"Extracting: {file}")

            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    print("All .gz files extracted successfully.")


def main():
    # 🔧 Adjust these paths
    source_dir = r"data/raw_archives"
    target_dir = r"data/raw"

    extract_all_gz(source_dir, target_dir)


if __name__ == "__main__":
    main()