"""Read the header of every train and test DICOM into dicom_headers.csv (one row per file, all fields as text).

Usage: python scripts/extract_dicom_headers.py [--workers N]

Only headers are read, not pixels. The table is the input of 01_eda and of the metadata table in 02.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from ich.data.dicom import read_header
from ich.paths import load_paths

SPLITS = {"train": "stage_2_train", "test": "stage_2_test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    paths = load_paths()
    raw_dir, out_dir = paths["raw_data_dir"], paths["processed_data_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    for split, folder in SPLITS.items():
        with os.scandir(raw_dir / folder) as entries:
            files = sorted(e.path for e in entries if e.name.endswith(".dcm"))
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(tqdm(pool.map(read_header, files, chunksize=512), total=len(files), desc=split))
        tables.append(pd.DataFrame(rows).assign(split=split))

    headers = pd.concat(tables, ignore_index=True)
    headers.to_csv(out_dir / "dicom_headers.csv", index=False)
    n_errors = headers["read_error"].notna().sum() if "read_error" in headers else 0
    print(f"files: {len(headers):,} | fields: {headers.shape[1]} | read errors: {n_errors} -> {out_dir / 'dicom_headers.csv'}")


if __name__ == "__main__":
    main()
