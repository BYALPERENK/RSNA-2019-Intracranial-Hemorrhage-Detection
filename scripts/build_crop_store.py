"""Build a cropped store of the train slices (04_slice_models): every slice cut to its box and resized to 256.

Usage: python scripts/build_crop_store.py [--boxes series|slice256] [--workers N]

Writes into processed_data_dir, for the box of the series (`ich.data.crop.series_boxes`) or the box of the slice of at
least MIN_SIDE pixels (`slice_boxes`):
- crop_boxes.csv / slice256_boxes.csv: the boxes, train and test;
- png16_crop256/ / png16_slice256/<image>.png: the train slices, PNG 16-bit HU + 1024 as in png16/. Files already
  there are kept.

The store makes screening faster: training and its validation read it, because its files equal the crops made on the
fly; at the end every slice of fold 0 is checked for it. The check is per file: with the slice box, a neighbour read
from the store is cut by its own box, not by the box of the centre slice as on the fly (04, section 5). Final models are validated, and test images are read, from the full images in png16/ cropped on the fly with the
same function.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ich.data.crop import MIN_SIDE, crop, series_boxes, slice_boxes
from ich.data.prepare import STORE, read_table, store_path
from ich.paths import load_paths


def write_crop(source, target, top, left, side):
    if os.path.exists(target):
        return
    cropped = crop(STORE.decode(Path(source).read_bytes()), top, left, side)
    temporary = target + ".tmp"
    Path(temporary).write_bytes(STORE.encode(cropped))
    os.replace(temporary, target)


def same_as_on_the_fly(source, target, top, left, side):
    return np.array_equal(STORE.decode(Path(target).read_bytes()), crop(STORE.decode(Path(source).read_bytes()), top, left, side))


STORES = {"series": ("crop_boxes.csv", "png16_crop256"), "slice256": ("slice256_boxes.csv", "png16_slice256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--boxes", choices=list(STORES), default="series")
    parser.add_argument("--workers", type=int, default=os.cpu_count() - 2)
    args = parser.parse_args()

    out_dir = load_paths()["processed_data_dir"]
    table = read_table(out_dir / "slices.csv")
    box_file, store_name = STORES[args.boxes]
    if args.boxes == "series":
        boxes = series_boxes(table)
        train = table[table["split"] == "train"].join(boxes, on="series")
    else:
        boxes = slice_boxes(table, min_side=MIN_SIDE).set_index(table["image"])
        train = table[table["split"] == "train"].join(boxes, on="image")
    boxes.to_csv(out_dir / box_file)
    print(f"boxes: {len(boxes):,}, median side {boxes['side'].median():.0f} px", flush=True)

    store = out_dir / store_name
    store.mkdir(exist_ok=True)
    sources = [str(store_path(out_dir / "png16", image)) for image in train["image"]]
    targets = [str(store_path(store, image)) for image in train["image"]]
    with ProcessPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(write_crop, sources, targets, train["top"], train["left"], train["side"], chunksize=256),
                  total=len(targets), mininterval=30, ascii=True))
    size = sum(os.path.getsize(t) for t in targets)
    print(f"store: {len(targets):,} files, {size / 1e9:.1f} GB -> {store}", flush=True)

    check = np.flatnonzero((train["fold"] == 0).to_numpy())
    columns = [sources, targets, train["top"].tolist(), train["left"].tolist(), train["side"].tolist()]
    with ProcessPoolExecutor(args.workers) as pool:
        same = list(pool.map(same_as_on_the_fly, *[[column[i] for i in check] for column in columns], chunksize=256))
    assert all(same), f"{len(same) - sum(same):,} files of fold 0 differ from the crops made on the fly"
    print(f"check: all {len(same):,} slices of fold 0 equal the crops made on the fly")


if __name__ == "__main__":
    main()
