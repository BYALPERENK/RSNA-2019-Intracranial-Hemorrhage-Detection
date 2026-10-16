"""Tests for the storage formats in src/ich/data/formats.py and the windows in src/ich/data/windows.py."""
import os

import numpy as np
import pytest

from ich.data.dicom import HU_MAX, HU_MIN
from ich.data.formats import FORMATS, HU8_LOW, hu_int16, hu_uint8, read_series_slices, read_uncached, windows_int16
from ich.data.windows import WINDOWS, to_uint8, window


def head(size=64, seed=0):
    """A synthetic slice: air, a ring of bone, brain with a little noise and a bleed of 70 HU."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size] - size / 2
    r = np.hypot(x, y)
    hu = np.where(r < size * 0.45, 1500.0, -1000.0)
    brain = r < size * 0.4
    hu[brain] = 30 + rng.integers(-3, 4, brain.sum())
    hu[np.hypot(x - size * 0.15, y) < size * 0.1] = 70
    return hu.astype(np.float32)


@pytest.mark.parametrize("name", [n for n, f in FORMATS.items() if f.encode is not None and f.suffix != ".jpg"])
def test_lossless_formats_give_back_the_stored_array(name):
    f = FORMATS[name]
    stored = f.to_array(head())
    out = f.decode(f.encode(stored))
    assert out.dtype == stored.dtype and np.array_equal(out, stored)


@pytest.mark.parametrize("name", ["jpg8", "jpg8x3"])
def test_jpeg_gives_back_a_close_array(name):
    f = FORMATS[name]
    stored = f.to_array(head())
    out = f.decode(f.encode(stored))
    assert out.shape == stored.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - stored).mean() < 3  # levels; the noise of the synthetic brain is hard for JPEG


def test_hu_int16_is_exact_for_integer_hu():
    hu = np.arange(HU_MIN, HU_MAX + 1, dtype=np.float32)
    assert np.array_equal(hu_int16(hu), hu)


def test_hu_uint8_keeps_one_level_per_hu_in_its_range():
    hu = np.arange(-100, 300, dtype=np.float32)
    out = hu_uint8(hu)
    inside = (hu >= HU8_LOW) & (hu <= HU8_LOW + 255)
    assert np.array_equal(out[inside].astype(int) + HU8_LOW, hu[inside])
    assert (out[hu < HU8_LOW] == 0).all() and (out[hu > HU8_LOW + 255] == 255).all()


def test_window_ends():
    assert window(np.array([-1, 0, 40, 80, 81]), 40, 80).tolist() == [0, 0, 0.5, 1, 1]


@pytest.mark.parametrize("name", ["brain", "subdural"])
def test_narrow_windows_lose_nothing_at_8_bits(name):
    level, width = WINDOWS[name]
    hu = np.arange(level - width // 2, level + width // 2 + 1)
    assert np.unique(to_uint8(window(hu, level, width))).size == hu.size


def test_windows_int16_holds_each_window_above_its_lower_edge():
    out = windows_int16(np.array([[-1024, 0, 40, 100, 3071]], dtype=np.float32))
    assert out.shape == (1, 5, 3)
    assert out[0, :, 0].tolist() == [0, 0, 40, 80, 80]        # brain: 0 to 80 HU
    assert out[0, :, 1].tolist() == [0, 20, 60, 120, 200]     # subdural: -20 to 180 HU
    assert out[0, :, 2].tolist() == [0, 800, 840, 900, 2800]  # bone: -800 to 2000 HU


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
@pytest.mark.parametrize("size", [0, 1, 4096, 5000])
def test_read_uncached_gives_the_bytes_of_the_file(tmp_path, size):
    data = os.urandom(size)
    path = tmp_path / "a.bin"
    path.write_bytes(data)
    assert read_uncached(path) == data


def test_read_series_slices_reads_the_slices_asked_for(tmp_path):
    volume = np.arange(5 * 4 * 3, dtype=np.int16).reshape(5, 4, 3)
    np.save(tmp_path / "s.npy", volume)
    assert np.array_equal(read_series_slices(tmp_path / "s.npy", 1, 3), volume[1:4])
    assert np.array_equal(read_series_slices(tmp_path / "s.npy", 4, 1), volume[4:])
