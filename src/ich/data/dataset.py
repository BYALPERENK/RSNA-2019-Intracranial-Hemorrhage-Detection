"""Reading slices for the image models (04_slice_models).

A sample is the stored uint16 values (HU + 1024) of one slice, or of the slice with the kept slices of the positions
below and above it (2.5D), as a tensor of shape (n slices, size, size). Windows and augmentation are done on the GPU
(`ich.training`). The Dataset class lives here, not in a notebook: on Windows the loader workers must import it.

Two sources:
- the cropped training store (png16_crop256/), already cut to the box of the series;
- the full images (png16/), cropped on the fly when boxes are given. Validation and test always use this one
  so that they are read as the real test images are.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ich.data.crop import crop
from ich.data.prepare import STORE, store_path


def neighbour_images(slices: pd.DataFrame) -> pd.DataFrame:
    """For every slice, the images of the kept slices one position below and above in its series (02, 3.1).

    A dropped slice gets the neighbours of its position. At the ends of a series the slice itself is repeated.
    """
    scan = slices[slices["keep"]].sort_values(["series", "slice"])
    by_series = scan.groupby("series")["image"]
    keyed = pd.DataFrame({"series": scan["series"], "position": scan["position"],
                          "below": by_series.shift(1), "above": by_series.shift(-1)})
    out = slices[["series", "position"]].merge(keyed, on=["series", "position"], how="left")
    out.index = slices.index
    for column in ["below", "above"]:
        out[column] = out[column].fillna(slices["image"])
    return out[["below", "above"]]


class SliceDataset(Dataset):
    """Item i: the stored values of image i (and its neighbours) as uint16 (n slices, size, size), and i.

    `images`: one row per slice, columns of image IDs in the order of the channels (e.g. below, image, above).
    `boxes`: top, left and side per row to crop from the full images; None when `store_dir` is already cropped.
    """

    def __init__(self, images: np.ndarray, store_dir, boxes: np.ndarray | None = None, size: int = 256):
        self.images = np.asarray(images, dtype=object).reshape(len(images), -1)
        self.store_dir = Path(store_dir)
        self.boxes = None if boxes is None else np.asarray(boxes, dtype=int)
        self.size = size

    def __len__(self):
        return len(self.images)

    def read(self, image, i):
        stored = STORE.decode(store_path(self.store_dir, image).read_bytes())
        if self.boxes is not None:
            stored = crop(stored, *self.boxes[i], size=self.size)
        return stored

    def __getitem__(self, i):
        x = np.stack([self.read(image, i) for image in self.images[i]])
        return torch.from_numpy(x), i
