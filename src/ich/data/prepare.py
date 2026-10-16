"""The slice table and the image store, as derived step by step in 02_data_preparation (sections 1-6).

`build_table` gives one row per slice (train and test) with its position in the scan, the flags of section 3,
the labels and the fold. `write_slice` writes one slice to the store (PNG 16-bit HU, section 6) and returns its
pixel statistics. `scripts/prepare_data.py` runs both; 02, section 7, checks the result.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ich.data.dicom import HU_MIN, read_hu, scaled_spacing
from ich.data.formats import FORMATS

SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
LABELS = SUBTYPES + ["any"]
CORRUPT = "ID_6431af929"  # its pixels cannot be read (01, 1.2)
SAME_POSITION_MM = 2.0     # slices closer than this count as one position (02, 3.1)
N_FOLDS, SEED = 5, 42
STORE = FORMATS["png16"]
TABLE_DTYPES = {**{c: "Int8" for c in LABELS}, "fold": "Int8", "gap_prev": float, "gap_next": float}


# --- labels and headers (02, sections 1-2) ------------------------------------------------------------------------

def load_labels(csv_path) -> pd.DataFrame:
    """One row per training image with the six labels, without the repeated rows and the corrupt image."""
    raw = pd.read_csv(csv_path)
    parts = raw["ID"].str.rsplit("_", n=1, expand=True)
    raw["image"], raw["label"] = parts[0], parts[1]
    labels = raw.drop_duplicates("ID").pivot(index="image", columns="label", values="Label")[LABELS]
    return labels.drop(index=CORRUPT)


def slice_table(headers: pd.DataFrame) -> pd.DataFrame:
    """One row per slice from the header table: ids, writer, storage, size, spacing, start and position along the normal."""
    def numbers(col, n):
        return headers[col].str.split("\\", expand=True).iloc[:, :n].astype(float).to_numpy()

    image = headers["file"].str.removesuffix(".dcm")
    slices = pd.DataFrame({"image": image, "split": headers["split"], "patient": headers["PatientID"], "study": headers["StudyInstanceUID"],
                           "series": headers["SeriesInstanceUID"], "writer": headers["ImplementationVersionName"]})
    slices["storage"] = headers["BitsStored"] + " bit / repr " + headers["PixelRepresentation"]
    slices["rows"], slices["cols"] = headers["Rows"].astype(int), headers["Columns"].astype(int)
    slices["pixel_spacing"] = numbers("PixelSpacing", 1)[:, 0]
    slices["intercept_fix"] = ((headers["BitsStored"] == "12") & (headers["PixelRepresentation"] == "0")
                               & (headers["RescaleIntercept"].astype(float) > -100))
    position = numbers("ImagePositionPatient", 3)
    orientation = numbers("ImageOrientationPatient", 6)
    normal = np.cross(orientation[:, :3], orientation[:, 3:])
    slices["x0"], slices["y0"] = position[:, 0], position[:, 1]
    slices["u0"], slices["v0"] = (position * orientation[:, :3]).sum(axis=1), (position * orientation[:, 3:]).sum(axis=1)
    slices["pos"] = (position * normal).sum(axis=1)
    slices["tilt"] = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0, 1)))
    return slices[slices["image"] != CORRUPT].sort_values(["series", "pos", "image"]).reset_index(drop=True)


# --- positions (02, section 3) ------------------------------------------------------------------------------------

def number_positions(series, pos, limit=SAME_POSITION_MM) -> np.ndarray:
    """Position index within each series: a new position starts `limit` mm or more above the first slice of the current one."""
    out = np.empty(len(pos), dtype=int)
    for i in range(len(pos)):
        if i == 0 or series[i] != series[i - 1]:
            k, first = 0, pos[i]
        elif pos[i] - first >= limit:
            k, first = k + 1, pos[i]
        out[i] = k
    return out


def add_positions(slices: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`exact_pos`, `position`, `copies` and `keep` (one slice per position), and the labels per train image.

    Train: the kept slice is the one with the most positive labels (then the lowest, then the first ID) and gets the
    maximum of each label over its position; dropped slices keep their own labels. Test: no slice is dropped, `keep`
    marks the lowest slice of each position.
    """
    slices = slices.copy()
    slices["exact_pos"] = (~(slices.groupby("series")["pos"].diff() < 0.01)).groupby(slices["series"]).cumsum() - 1
    slices["position"] = number_positions(slices["series"].to_numpy(), slices["pos"].to_numpy())
    slices["copies"] = slices.groupby(["series", "position"])["image"].transform("size")

    is_train = slices["split"] == "train"
    train = slices[is_train].join(labels, on="image")
    n_positive = train[LABELS].sum(axis=1)
    kept = (train.assign(n_positive=n_positive)
            .sort_values(["series", "position", "n_positive", "pos", "image"], ascending=[True, True, False, True, True])
            .drop_duplicates(["series", "position"]).index)
    union_labels = train.groupby(["series", "position"])[LABELS].transform("max")
    image_labels = train[LABELS].copy()
    image_labels.loc[kept] = union_labels.loc[kept]
    image_labels = image_labels.set_axis(train["image"])

    slices["keep"] = True
    slices.loc[train.index, "keep"] = train.index.isin(kept)
    test_kept = slices[~is_train].sort_values(["series", "position", "pos", "image"]).drop_duplicates(["series", "position"]).index
    slices.loc[~is_train, "keep"] = slices.index[~is_train].isin(test_kept)
    return slices, image_labels


# --- distances and stacks (02, section 3.2) -----------------------------------------------------------------------

def stack_ids(a, b, tilt) -> np.ndarray:
    """A slice joins an earlier stack if its start (a, b) is within 1 mm and its tilt within 0.5 degrees of that stack's first slice."""
    firsts, ids = [], []
    for x, y, t in zip(a, b, tilt):
        for k, (fx, fy, ft) in enumerate(firsts):
            if np.hypot(x - fx, y - fy) < 1 and abs(t - ft) < 0.5:
                ids.append(k)
                break
        else:
            firsts.append((x, y, t))
            ids.append(len(firsts) - 1)
    return np.array(ids)


def series_stacks(g: pd.DataFrame) -> pd.Series:
    """Stacks of one series: grouped by (x0, y0) or by (u0, v0), whichever gives fewer."""
    by_xy = stack_ids(g["x0"], g["y0"], g["tilt"])
    by_plane = stack_ids(g["u0"], g["v0"], g["tilt"])
    return pd.Series(by_xy if by_xy.max() <= by_plane.max() else by_plane, index=g.index)


def add_scan_columns(slices: pd.DataFrame) -> pd.DataFrame:
    """Per slice the position in the evenly spaced scan, the gaps, the local spacing and the stack; per series the spacing change,
    the number of stacks and whether they interleave. Slices that share a position take its values."""
    scan = slices[slices["keep"]].copy()
    by_series = scan.groupby("series")
    scan["slice"] = by_series.cumcount()
    scan["n_slices"] = by_series["image"].transform("size")
    scan["gap_prev"] = by_series["pos"].diff()
    scan["gap_next"] = -by_series["pos"].diff(-1)
    scan["local_spacing"] = scan.groupby("series")["gap_prev"].transform(lambda g: g.rolling(5, center=True, min_periods=1).median())
    scan["wide_gap"] = scan["gap_prev"] > 1.5 * scan["local_spacing"]
    per_series = scan.groupby("series").agg(thin=("local_spacing", "min"), thick=("local_spacing", "max"))
    per_series["spacing_change"] = per_series["thick"] / per_series["thin"] >= 1.5

    slices = slices.copy()
    slices["stack"] = 0
    start = slices["x0"].round(1).astype(str) + "," + slices["y0"].round(1).astype(str) + "," + slices["tilt"].round(1).astype(str)
    start_values = start.groupby(slices["series"]).nunique()
    part = slices[slices["series"].isin(start_values[start_values > 1].index)]
    slices.loc[part.index, "stack"] = part.groupby("series", group_keys=False)[["x0", "y0", "u0", "v0", "tilt"]].apply(series_stacks)
    stacks = slices.groupby("series")["stack"].agg(n_stacks="nunique", switches=lambda s: int((s.diff().fillna(0) != 0).sum()))
    stacks["interleaved"] = stacks["switches"] > 1

    columns = ["slice", "n_slices", "gap_prev", "gap_next", "local_spacing", "wide_gap"]
    slices = slices.merge(scan[["series", "position"] + columns], on=["series", "position"], how="left")
    return slices.join(per_series[["spacing_change"]], on="series").join(stacks[["n_stacks", "interleaved"]], on="series")


# --- folds (02, section 4) ----------------------------------------------------------------------------------------

def iterative_folds(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Iterative stratification (Sechidis et al. 2011). y: patients x labels (0/1); every row needs at least one 1."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(y), -1)
    want = np.full(k, len(y) / k)
    want_label = np.tile(y.sum(axis=0) / k, (k, 1))
    left = np.ones(len(y), bool)
    while left.any():
        remaining = y[left].sum(axis=0).astype(float)
        remaining[remaining == 0] = np.inf
        label = remaining.argmin()
        for i in rng.permutation(np.flatnonzero(left & (y[:, label] == 1))):
            best = np.flatnonzero(want_label[:, label] == want_label[:, label].max())
            best = best[want[best] == want[best].max()]
            f = rng.choice(best)
            fold[i] = f
            want[f] -= 1
            want_label[f] -= y[i]
            left[i] = False
    return fold


def patient_folds(slices: pd.DataFrame, image_labels: pd.DataFrame) -> pd.Series:
    """Fold of every train patient: stratified by the six labels of its kept slices and by its writer."""
    train = slices[(slices["split"] == "train") & slices["keep"]].join(image_labels, on="image")
    patients = train.groupby("patient")[LABELS].max().join(pd.get_dummies(train.groupby("patient")["writer"].first()).astype(int))
    return pd.Series(iterative_folds(patients.to_numpy(), N_FOLDS, SEED), index=patients.index, name="fold")


# --- the table ----------------------------------------------------------------------------------------------------

def build_table(headers: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per slice, train and test, ordered by series and position along the normal.

    `pixel_spacing` is that of the 512 x 512 image. Train rows hold the labels (kept slices: of their position, dropped
    ones: their own) and the fold of their patient; test rows hold neither.
    """
    slices, image_labels = add_positions(slice_table(headers), labels)
    slices = add_scan_columns(slices)
    slices["pixel_spacing"] = scaled_spacing(slices["pixel_spacing"], slices["rows"], slices["cols"])
    folds = patient_folds(slices, image_labels)
    return slices.join(image_labels.astype("Int8"), on="image").join(folds.astype("Int8"), on="patient")


def read_table(path) -> pd.DataFrame:
    """The slice table as written by `scripts/prepare_data.py`, with its column types."""
    return pd.read_csv(path, dtype=TABLE_DTYPES, float_precision="round_trip")


# --- the image store ----------------------------------------------------------------------------------------------

def pixel_stats(hu: np.ndarray) -> dict:
    """Share of brain pixels (0 < HU < 80, as fastai's `pct_in_window` for the brain window) and the box of the head:
    the largest connected region above -300 HU (skin, brain, bone), as top, bottom, left, right (bottom and right exclusive)."""
    stats = {"brain_share": float(((hu > 0) & (hu < 80)).mean())}
    n, _, boxes, _ = cv2.connectedComponentsWithStats((hu > -300).astype(np.uint8), connectivity=8)
    if n > 1:
        x, y, w, h, area = boxes[1 + boxes[1:, cv2.CC_STAT_AREA].argmax()]
        stats.update(head_top=int(y), head_bottom=int(y + h), head_left=int(x), head_right=int(x + w), head_share=area / hu.size)
    else:
        stats.update(head_top=-1, head_bottom=-1, head_left=-1, head_right=-1, head_share=0.0)
    return stats


def store_path(store_dir, image) -> Path:
    return Path(store_dir) / f"{image}{STORE.suffix}"


def write_slice(dicom_path, out_path) -> dict:
    """Write one slice to the store (unless it is already there) and return its pixel statistics, computed on the stored HU.

    The file is written under a temporary name and renamed, so an interrupted run never leaves a partial file behind.
    """
    out_path = Path(out_path)
    if out_path.exists():
        hu = STORE.decode(out_path.read_bytes()).astype(np.int16) + HU_MIN
    else:
        stored = STORE.to_array(read_hu(dicom_path))
        temporary = out_path.with_name(out_path.name + ".tmp")
        temporary.write_bytes(STORE.encode(stored))
        os.replace(temporary, out_path)
        hu = stored.astype(np.int16) + HU_MIN
    return pixel_stats(hu)
