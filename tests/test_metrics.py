"""Tests for the competition metric and the patient bootstrap in src/ich/metrics.py."""
import numpy as np
import pytest
from sklearn.metrics import log_loss

from ich.data.prepare import LABELS
from ich.metrics import bootstrap, image_losses, paired_bootstrap, score, score_by_label


def random_labels(n, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 6)) < 0.2).astype(int)
    y[:, LABELS.index("any")] = y[:, :5].max(axis=1)
    return y, rng.random((n, 6))


def test_a_perfect_prediction_scores_zero():
    y, _ = random_labels(50)
    assert score(y, y) == pytest.approx(0, abs=1e-12)


def test_a_constant_half_scores_log_2():
    y, _ = random_labels(50)
    assert score(y, np.full(y.shape, 0.5)) == pytest.approx(np.log(2))


def test_each_label_matches_sklearn_and_any_weighs_two_sevenths():
    y, p = random_labels(200)
    per_label = [log_loss(y[:, j], p[:, j], labels=[0, 1]) for j in range(6)]
    weights = [2 if label == "any" else 1 for label in LABELS]
    assert score(y, p) == pytest.approx(np.dot(per_label, weights) / 7)
    parts = score_by_label(y, p)
    assert sum(part["share of the score"] for part in parts.values()) == pytest.approx(score(y, p))
    assert parts["any"]["log loss"] == pytest.approx(per_label[LABELS.index("any")])


def test_dividing_by_six_would_not_pass():
    # two published solutions divided by 6 instead of 7, which inflates every score by 7/6
    y, p = random_labels(200)
    by_six = (np.array([log_loss(y[:, j], p[:, j], labels=[0, 1]) for j in range(6)]) * [1, 1, 1, 1, 1, 2]).sum() / 6
    assert score(y, p) == pytest.approx(by_six * 6 / 7)
    assert score(y, p) != pytest.approx(by_six)


def test_predictions_are_clipped_to_1e_15():
    y = np.ones((1, 6))
    assert score(y, np.zeros((1, 6))) == pytest.approx(-np.log(1e-15))
    assert np.isfinite(score(1 - y, np.ones((1, 6))))


def test_the_score_is_the_mean_of_the_image_losses():
    y, p = random_labels(30)
    assert image_losses(y, p).shape == (30,)
    assert image_losses(y, p).mean() == pytest.approx(score(y, p))


def test_bootstrap_resamples_whole_groups():
    # two groups with constant losses 0 and 1: every resample mean is a mix of whole groups
    losses = np.array([0.0] * 10 + [1.0] * 10)
    groups = np.array(["a"] * 10 + ["b"] * 10)
    result = bootstrap(losses, groups, n_boot=500, seed=1)
    assert result["mean"] == pytest.approx(0.5)
    assert {result["low"], result["high"]} <= {0.0, 0.5, 1.0}


def test_bootstrap_weights_groups_by_their_images():
    losses = np.array([1.0, 1.0, 1.0, 0.0])
    groups = np.array(["a", "a", "a", "b"])
    assert bootstrap(losses, groups, n_boot=10)["mean"] == pytest.approx(0.75)


def test_paired_bootstrap_of_a_constant_difference_has_no_width():
    rng = np.random.default_rng(0)
    a = rng.random(100)
    groups = rng.integers(0, 20, 100)
    result = paired_bootstrap(a, a + 0.01, groups, n_boot=200)
    assert result["mean"] == pytest.approx(0.01)
    assert result["high"] - result["low"] == pytest.approx(0, abs=1e-12)
