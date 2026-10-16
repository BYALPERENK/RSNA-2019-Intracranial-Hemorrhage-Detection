"""Build the slice table and the image store (02_data_preparation, section 7).

Usage: python scripts/prepare_data.py [--workers N]

Writes into processed_data_dir:
- png16/<image>.png: every slice as PNG 16-bit, HU + 1024 (02, section 6). Files already there are kept, so an
  interrupted run continues where it stopped;
- slices.csv: one row per slice (`ich.data.prepare.build_table`) with the pixel statistics of `pixel_stats`.
  Read it with `ich.data.prepare.read_table`.

Needs dicom_headers.csv from scripts/extract_dicom_headers.py.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from ich.data.prepare import build_table, load_labels, store_path, write_slice
from ich.paths import load_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count() - 2)
    args = parser.parse_args()

    paths = load_paths()
    raw_dir, out_dir = paths["raw_data_dir"], paths["processed_data_dir"]
    headers = pd.read_csv(out_dir / "dicom_headers.csv", dtype=str)
    table = build_table(headers, load_labels(raw_dir / "stage_2_train.csv"))
    print(f"slices: {len(table):,} ({table['split'].value_counts().to_dict()})", flush=True)

    store = out_dir / "png16"
    store.mkdir(exist_ok=True)
    sources = [str(raw_dir / f"stage_2_{split}" / f"{image}.dcm") for split, image in zip(table["split"], table["image"])]
    targets = [str(store_path(store, image)) for image in table["image"]]
    with ProcessPoolExecutor(args.workers) as pool:
        stats = list(tqdm(pool.map(write_slice, sources, targets, chunksize=256), total=len(targets), mininterval=30, ascii=True))

    table = table.join(pd.DataFrame(stats, index=table.index))
    table.to_csv(out_dir / "slices.csv", index=False)
    size = sum(os.path.getsize(t) for t in targets)
    print(f"store: {len(targets):,} files, {size / 1e9:.1f} GB -> {store}")
    print(f"table: {table.shape[0]:,} rows, {table.shape[1]} columns -> {out_dir / 'slices.csv'}")


if __name__ == "__main__":
    main()
