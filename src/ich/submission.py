"""Submission files: one row per test image and label, `ID_<image>_<label>`, matched to the sample submission by ID."""
import numpy as np
import pandas as pd

from ich.data.prepare import LABELS


def to_submission(images, p) -> pd.DataFrame:
    """Long format from one row per image (`images`) and one column per label (`p`, in the order of `LABELS`)."""
    images = np.asarray(images, dtype=str)
    p = np.asarray(p, dtype=float)
    assert p.shape == (len(images), len(LABELS)), p.shape
    ids = np.char.add(np.repeat(images, len(LABELS)), np.tile(np.array([f"_{label}" for label in LABELS]), len(images)))
    return pd.DataFrame({"ID": ids, "Label": p.reshape(-1)})


def check_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """The submission in the row order of the sample, joined by ID; fails on a missing, repeated or extra ID or a bad value."""
    assert not submission["ID"].duplicated().any(), "repeated IDs"
    missing = set(sample["ID"]) - set(submission["ID"])
    extra = set(submission["ID"]) - set(sample["ID"])
    assert not missing and not extra, f"{len(missing)} IDs missing, {len(extra)} IDs not in the sample"
    out = sample[["ID"]].merge(submission, on="ID", how="left", validate="one_to_one")
    assert out["Label"].notna().all(), "missing values"
    assert out["Label"].between(0, 1).all(), "values outside [0, 1]"
    return out
