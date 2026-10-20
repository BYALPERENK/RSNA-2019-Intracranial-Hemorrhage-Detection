"""Cropping a slice to the head of its series (04_slice_models).

The box of a series is one square around the head boxes of all its slices (`pixel_stats` in prepare.py), plus a margin:
the scale stays the same along the scan, so adjacent slices line up as channels. A box can reach past the image; the
part outside is air. `crop` works on the stored values of the PNG 16-bit store, so the cropped store
(`scripts/build_crop_store.py`) and a crop made on the fly from the full image give the same pixels.
"""
import cv2
import numpy as np
import pandas as pd

from ich.data.dicom import IMAGE_SIZE

MARGIN = 8           # pixels of the 512 image on each side of the head
CROP_SIZE = 256
MIN_SIDE = 256       # smallest slice box: a crop to 256 is then never enlarged, so tiny slices at the top are not blurred


def series_boxes(slices: pd.DataFrame, margin: int = MARGIN) -> pd.DataFrame:
    """One square box per series as top, left and side, in pixels of the 512 image.

    Slices without a head box (-1) are left out; a series without any gets the whole image.
    """
    found = slices[slices["head_top"] >= 0]
    union = found.groupby("series").agg(top=("head_top", "min"), bottom=("head_bottom", "max"),
                                        left=("head_left", "min"), right=("head_right", "max"))
    side = np.maximum(union["bottom"] - union["top"], union["right"] - union["left"]) + 2 * margin
    boxes = pd.DataFrame({"top": (union["top"] + union["bottom"] - side) // 2,
                          "left": (union["left"] + union["right"] - side) // 2,
                          "side": side})
    whole = pd.DataFrame({"top": 0, "left": 0, "side": IMAGE_SIZE}, index=pd.Index(slices["series"].unique(), name="series"))
    return boxes.combine_first(whole).astype(int).loc[whole.index]


def slice_boxes(slices: pd.DataFrame, margin: int = MARGIN, min_side: int = 0) -> pd.DataFrame:
    """One square box per slice around its own head box (the variant of 04, 5), aligned with `slices`.

    The scale changes along the scan. With adjacent slices as channels, the box of the centre slice cuts all of them,
    so the channels stay aligned. Slices without a head box get the box of their series. A box smaller than
    `min_side` grows to it around the same centre.
    """
    side = np.maximum(slices["head_bottom"] - slices["head_top"], slices["head_right"] - slices["head_left"]) + 2 * margin
    boxes = pd.DataFrame({"top": (slices["head_top"] + slices["head_bottom"] - side) // 2,
                          "left": (slices["head_left"] + slices["head_right"] - side) // 2,
                          "side": side}, index=slices.index)
    series = slices[["series"]].join(series_boxes(slices, margin), on="series")[["top", "left", "side"]]
    missing = slices["head_top"] < 0
    boxes[missing] = series[missing]
    grow = np.maximum(min_side - boxes["side"], 0)
    boxes["top"], boxes["left"], boxes["side"] = boxes["top"] - grow // 2, boxes["left"] - grow // 2, boxes["side"] + grow
    return boxes.astype(int)


def crop(stored: np.ndarray, top: int, left: int, side: int, size: int = CROP_SIZE) -> np.ndarray:
    """The box of a stored slice (uint16, HU + 1024), padded with air (0) where it leaves the image, resized to size x size."""
    rows, cols = stored.shape
    out = np.zeros((side, side), dtype=stored.dtype)
    r0, r1, c0, c1 = max(top, 0), min(top + side, rows), max(left, 0), min(left + side, cols)
    out[r0 - top:r1 - top, c0 - left:c1 - left] = stored[r0:r1, c0:c1]
    if side != size:
        interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
        out = cv2.resize(out, (size, size), interpolation=interpolation)
    return out
