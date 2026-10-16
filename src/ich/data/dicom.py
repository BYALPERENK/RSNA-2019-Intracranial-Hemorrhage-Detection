"""Reading the DICOM files."""
import os

import cv2
import numpy as np
import pydicom
from pydicom.multival import MultiValue

# The image IDs (`ID_...`) are not valid DICOM UIDs; without this pydicom warns on every file.
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

# HU outside this range carry nothing for the brain: below is padding outside the scanned field, above is metal (01_eda, 6.3).
HU_MIN, HU_MAX = -1024, 3071
IMAGE_SIZE = 512


def as_text(value) -> str:
    """A header value as text; multi-valued fields are joined with a backslash, as stored in DICOM."""
    if isinstance(value, MultiValue):
        return "\\".join(str(v) for v in value)
    return str(value)


def read_header(path: str) -> dict:
    """Every header field of one file as text (file meta included, pixels not read), plus the file size.

    A file that cannot be read gets a `read_error` instead of its fields, so no file is lost silently.
    """
    row = {"file": os.path.basename(path), "file_size": os.path.getsize(path)}
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as e:
        row["read_error"] = repr(e)
        return row
    for elem in [*ds.file_meta, *ds]:
        if elem.keyword:
            row[elem.keyword] = as_text(elem.value)
    return row


def needs_intercept_fix(bits_stored: int, pixel_representation: int, intercept: float) -> bool:
    """Files marked 12 bit unsigned with an intercept near 0. They store HU modulo 4096 (01_eda, 6.1)."""
    return bits_stored == 12 and pixel_representation == 0 and intercept > -100


def to_hu(stored: np.ndarray, slope: float, intercept: float, bits_stored: int, pixel_representation: int) -> np.ndarray:
    """Stored pixel values to Hounsfield units, as float32.

    The values are cast to float first: uint16 plus a negative intercept would wrap around.
    For the files of `needs_intercept_fix`: add 1000, subtract 4096 from values of 4096 and above,
    and use an intercept of -1000. Stored values from 3,096 up then become -1000 to -1 HU.
    """
    hu = np.array(stored, dtype=np.float32)
    if needs_intercept_fix(bits_stored, pixel_representation, intercept):
        hu += 1000
        hu[hu >= 4096] -= 4096
        intercept = -1000.0
    return hu * np.float32(slope) + np.float32(intercept)


def to_square(hu: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Pad a non-square image with air (HU_MIN) to a centred square, then resize it to size x size.

    The pixels are square in every file, so padding first keeps the shape of the head.
    """
    rows, cols = hu.shape
    if rows != cols:
        side = max(rows, cols)
        top, left = (side - rows) // 2, (side - cols) // 2
        square = np.full((side, side), HU_MIN, dtype=hu.dtype)
        square[top:top + rows, left:left + cols] = hu
        hu = square
    if hu.shape[0] != size:
        interpolation = cv2.INTER_AREA if hu.shape[0] > size else cv2.INTER_LINEAR
        hu = cv2.resize(hu, (size, size), interpolation=interpolation)
    return hu


def scaled_spacing(pixel_spacing, rows, cols, size: int = IMAGE_SIZE):
    """Pixel spacing (mm) of the image after `to_square`; works on numbers and on table columns."""
    return pixel_spacing * np.maximum(rows, cols) / size


def read_hu(path, size: int = IMAGE_SIZE) -> np.ndarray:
    """One slice in Hounsfield units: float32, clipped to [HU_MIN, HU_MAX], size x size.

    Raises if the pixels cannot be read; a broken file never becomes an empty image.
    """
    ds = pydicom.dcmread(path)
    hu = to_hu(ds.pixel_array, float(ds.RescaleSlope), float(ds.RescaleIntercept), int(ds.BitsStored), int(ds.PixelRepresentation))
    return to_square(np.clip(hu, HU_MIN, HU_MAX), size)
