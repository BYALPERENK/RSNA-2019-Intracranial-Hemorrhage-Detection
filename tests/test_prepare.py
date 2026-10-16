"""Tests for the slice table helpers and the store writer in src/ich/data/prepare.py."""
import numpy as np
import pytest

from ich.data.dicom import HU_MIN, read_hu
from ich.data.prepare import STORE, iterative_folds, number_positions, pixel_stats, stack_ids, write_slice
from test_dicom import write_dicom


def test_a_position_starts_2_mm_above_the_first_slice_of_the_current_one():
    series = np.array(["a"] * 6 + ["b"] * 2)
    pos = np.array([0, 0.5, 1.9, 2.0, 2.4, 5.0, 0, 1.0])
    assert number_positions(series, pos).tolist() == [0, 0, 0, 1, 1, 2, 0, 0]


def test_positions_do_not_chain():
    # 1.5 mm steps: each slice is close to the one below, but 3.0 is 3 mm above the first slice of its position
    assert number_positions(np.array(["a"] * 4), np.array([0, 1.5, 3.0, 4.5])).tolist() == [0, 0, 1, 1]


def test_stacks_group_slices_by_start_and_tilt():
    a = [0, 0.5, 10, 0.2, 10.3, 0]
    tilt = [0, 0, 0, 0, 0.2, 1]
    assert stack_ids(a, [0] * 6, tilt).tolist() == [0, 0, 1, 0, 1, 2]


def test_iterative_folds_spread_a_rare_label_evenly():
    y = np.zeros((100, 2), dtype=int)
    y[:, 1] = 1   # every patient has one writer
    y[:10, 0] = 1  # a rare label
    fold = iterative_folds(y, 5, seed=0)
    assert np.bincount(fold).tolist() == [20] * 5
    assert np.bincount(fold[:10], minlength=5).tolist() == [2] * 5


def test_pixel_stats_find_the_head_and_the_brain_pixels():
    hu = np.full((64, 64), -1000, dtype=np.int16)
    hu[10:30, 20:50] = 40   # head
    hu[50:52, 0:5] = 100    # a small object apart from it, not in the brain window
    stats = pixel_stats(hu)
    assert stats["brain_share"] == pytest.approx(600 / 4096)
    assert (stats["head_top"], stats["head_bottom"], stats["head_left"], stats["head_right"]) == (10, 30, 20, 50)
    assert stats["head_share"] == pytest.approx(600 / 4096)


def test_pixel_stats_of_an_empty_slice():
    stats = pixel_stats(np.full((64, 64), -1000, dtype=np.int16))
    assert stats["brain_share"] == 0 and stats["head_share"] == 0 and stats["head_top"] == -1


def test_write_slice_stores_the_hu_and_keeps_existing_files(tmp_path):
    stored = np.full((512, 512), 24, dtype=np.int16)  # -1000 HU with the intercept of -1024
    stored[100:300, 150:350] = 1064                   # 40 HU
    dicom = write_dicom(tmp_path / "a.dcm", stored)
    out = tmp_path / "a.png"
    stats = write_slice(dicom, out)
    back = STORE.decode(out.read_bytes()).astype(np.int16) + HU_MIN
    assert np.array_equal(back, np.rint(read_hu(dicom)))
    assert not (tmp_path / "a.png.tmp").exists()
    assert (stats["head_top"], stats["head_left"]) == (100, 150)

    written = out.stat().st_mtime_ns
    assert write_slice(dicom, out) == stats and out.stat().st_mtime_ns == written
