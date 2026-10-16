"""Tests for the submission writer in src/ich/submission.py."""
import numpy as np
import pandas as pd
import pytest

from ich.data.prepare import LABELS
from ich.submission import check_submission, to_submission


def sample_for(images):
    return pd.DataFrame({"ID": [f"{image}_{label}" for image in images for label in LABELS], "Label": 0.5})


def test_each_value_lands_on_its_image_and_label():
    p = np.arange(12, dtype=float).reshape(2, 6) / 100
    submission = to_submission(["ID_a", "ID_b"], p)
    assert submission.set_index("ID").loc["ID_b_intraparenchymal", "Label"] == pytest.approx(0.07)
    assert submission.set_index("ID").loc["ID_a_any", "Label"] == pytest.approx(0.05)


def test_the_check_follows_the_sample_order_not_the_row_order():
    p = np.array([[0.1] * 6, [0.9] * 6])
    sample = sample_for(["ID_b", "ID_a"])
    out = check_submission(to_submission(["ID_a", "ID_b"], p), sample)
    assert out["ID"].tolist() == sample["ID"].tolist()
    assert out["Label"].tolist() == [0.9] * 6 + [0.1] * 6


def test_the_check_fails_on_a_missing_image_or_a_bad_value():
    sample = sample_for(["ID_a", "ID_b"])
    with pytest.raises(AssertionError):
        check_submission(to_submission(["ID_a"], np.full((1, 6), 0.1)), sample)
    with pytest.raises(AssertionError):
        check_submission(to_submission(["ID_a", "ID_b"], np.full((2, 6), 1.5)), sample)
