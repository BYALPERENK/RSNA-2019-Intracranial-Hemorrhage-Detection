"""Candidate formats for storing the slices, compared in 02_data_preparation (section 6).

A format keeps one slice of `read_hu` as an array (`to_array`), turns the array into the bytes of a file (`encode`)
and the bytes back into the array (`decode`). DICOM is the source itself: its `decode` is `read_hu`.
"""
import ctypes
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ich.data.dicom import HU_MIN, read_hu
from ich.data.windows import WINDOWS, to_uint8, window

HU8_LOW = -40  # 8-bit HU: -40 to 215 HU, one level per HU


def hu_int16(hu: np.ndarray) -> np.ndarray:
    """HU as int16. Exact for every slice that is not resized: all files have slope 1 and an integer intercept."""
    return np.rint(hu).astype(np.int16)


def hu_uint8(hu: np.ndarray) -> np.ndarray:
    """HU from -40 to 215 as 0-255: the brain (0 to 80 HU) and subdural (-20 to 180 HU) windows, with room to shift them."""
    return (np.clip(np.rint(hu), HU8_LOW, HU8_LOW + 255) - HU8_LOW).astype(np.uint8)


def windows_int16(hu: np.ndarray) -> np.ndarray:
    """The three windows as HU above their lower edge (0 to the width), height x width x 3."""
    h = hu_int16(hu)
    return np.stack([np.clip(h, level - width // 2, level + width // 2) - (level - width // 2)
                     for level, width in WINDOWS.values()], axis=-1).astype(np.int16)


def windows_uint8(hu: np.ndarray) -> np.ndarray:
    """The three windows at 8 bits, height x width x 3."""
    return np.stack([to_uint8(window(hu, level, width)) for level, width in WINDOWS.values()], axis=-1)


def npy_encode(a: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, a)
    return buffer.getvalue()


def npy_decode(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b))


def png_encode(a: np.ndarray, level: int = 6) -> bytes:
    """`level` 0-9: zlib compression. 6 (02, 6.1) makes the files 9-18% smaller than the OpenCV default 1, at the same reading speed."""
    return cv2.imencode(".png", a, [cv2.IMWRITE_PNG_COMPRESSION, level])[1].tobytes()


def jpg_encode(a: np.ndarray, quality: int = 95) -> bytes:
    """Quality 95 (the OpenCV default). Three channels are kept at full resolution (4:4:4), not subsampled."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    return cv2.imencode(".jpg", a, params)[1].tobytes()


def image_decode(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)


@dataclass(frozen=True)
class Format:
    label: str
    content: str
    suffix: str
    to_array: Callable | None  # HU image of read_hu -> stored array
    encode: Callable | None    # array -> bytes of the file
    decode: Callable           # bytes of the file -> array


FORMATS = {
    "dicom": Format("DICOM", "HU", ".dcm", None, None, lambda b: read_hu(io.BytesIO(b))),
    "npy16": Format("npy int16", "HU", ".npy", hu_int16, npy_encode, npy_decode),
    "npy16x3": Format("npy int16", "3 windows", ".npy", windows_int16, npy_encode, npy_decode),
    "png16": Format("PNG 16-bit", "HU", ".png", lambda hu: (hu_int16(hu) - HU_MIN).astype(np.uint16), png_encode, image_decode),
    "png16x3": Format("PNG 16-bit", "3 windows", ".png", lambda hu: windows_int16(hu).astype(np.uint16), png_encode, image_decode),
    "png8": Format("PNG 8-bit", "8-bit HU", ".png", hu_uint8, png_encode, image_decode),
    "png8x3": Format("PNG 8-bit", "3 windows", ".png", windows_uint8, png_encode, image_decode),
    "jpg8": Format("JPEG", "8-bit HU", ".jpg", hu_uint8, jpg_encode, image_decode),
    "jpg8x3": Format("JPEG", "3 windows", ".jpg", windows_uint8, jpg_encode, image_decode),
}


def read_series_slices(path, start: int, n: int) -> np.ndarray:
    """Slices `start` to `start + n - 1` of a series saved as one npy array (slices x height x width), without reading the rest."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = read_header(f)
        size = shape[1] * shape[2] * dtype.itemsize
        f.seek(start * size, os.SEEK_CUR)
        return np.frombuffer(f.read(n * size), dtype).reshape(n, shape[1], shape[2])


class DecodeDataset(Dataset):
    """Item i decodes file i with its format: for timing data loaders. `items` holds (format name, path) pairs."""

    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, path = self.items[i]
        return torch.from_numpy(FORMATS[name].decode(Path(path).read_bytes()))


if os.name == "nt":
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    _k32.ReadFile.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    _k32.VirtualAlloc.restype = wintypes.LPVOID
    _k32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
    _INVALID_HANDLE = wintypes.HANDLE(-1).value


def read_uncached(path) -> bytes:
    """The bytes of a file read from the disk, past the Windows file cache: what a read costs when the file is not in memory.

    Windows only (FILE_FLAG_NO_BUFFERING). The read size and the buffer must be multiples of the sector size (4,096 bytes).
    """
    size = -(-os.path.getsize(path) // 4096) * 4096 or 4096
    buffer = _k32.VirtualAlloc(None, size, 0x3000, 0x04)  # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE: page aligned
    handle = _k32.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x20000000, None)  # GENERIC_READ, share read, OPEN_EXISTING, NO_BUFFERING
    try:
        if handle == _INVALID_HANDLE:
            raise ctypes.WinError(ctypes.get_last_error())
        done = wintypes.DWORD()
        if not _k32.ReadFile(handle, buffer, size, ctypes.byref(done), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return ctypes.string_at(buffer, done.value)
    finally:
        if handle != _INVALID_HANDLE:
            _k32.CloseHandle(handle)
        _k32.VirtualFree(buffer, 0, 0x8000)  # MEM_RELEASE
