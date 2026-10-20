"""Tests for the series crop in src/ich/data/crop.py."""
import numpy as np
import pandas as pd

from ich.data.crop import crop, series_boxes, slice_boxes


def boxes_of(rows, margin=0):
    columns = ["series", "head_top", "head_bottom", "head_left", "head_right"]
    return series_boxes(pd.DataFrame(rows, columns=columns), margin)


def test_the_box_is_a_square_around_all_slices_of_the_series():
    box = boxes_of([["a", 100, 300, 150, 250], ["a", 120, 280, 100, 260]]).loc["a"]
    assert box.side == 200 and box.top == 100 and box.left == 80


def test_slices_without_a_head_box_are_ignored_and_a_series_without_any_gets_the_whole_image():
    boxes = boxes_of([["a", -1, -1, -1, -1], ["a", 0, 100, 0, 100], ["b", -1, -1, -1, -1]])
    assert boxes.loc["a"].tolist() == [0, 0, 100]
    assert boxes.loc["b"].tolist() == [0, 0, 512]


def test_a_slice_box_surrounds_its_own_head_and_falls_back_to_the_series_box():
    columns = ["series", "head_top", "head_bottom", "head_left", "head_right"]
    slices = pd.DataFrame([["a", 100, 300, 150, 250], ["a", 120, 280, 100, 260], ["a", -1, -1, -1, -1]], columns=columns)
    boxes = slice_boxes(slices, margin=0)
    assert boxes.iloc[0].tolist() == [100, 100, 200]
    assert boxes.iloc[1].tolist() == [120, 100, 160]
    assert boxes.iloc[2].tolist() == [100, 80, 200]  # the series box


def test_a_small_slice_box_grows_to_the_minimum_around_its_centre():
    columns = ["series", "head_top", "head_bottom", "head_left", "head_right"]
    slices = pd.DataFrame([["a", 100, 300, 150, 250], ["a", 200, 240, 200, 220]], columns=columns)
    boxes = slice_boxes(slices, margin=0, min_side=100)
    assert boxes.iloc[0].tolist() == [100, 100, 200]
    assert boxes.iloc[1].tolist() == [170, 160, 100]


def test_the_part_of_the_box_outside_the_image_is_air():
    stored = np.full((4, 4), 7, dtype=np.uint16)
    out = crop(stored, top=-2, left=0, side=4, size=4)
    assert (out[:2] == 0).all() and (out[2:] == 7).all()


def test_a_box_of_the_target_size_is_cut_without_resizing():
    stored = np.arange(512 * 512, dtype=np.uint32).reshape(512, 512).astype(np.uint16)
    assert np.array_equal(crop(stored, 10, 20, 256), stored[10:266, 20:276])
