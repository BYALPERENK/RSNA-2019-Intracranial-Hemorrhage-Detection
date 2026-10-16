"""The competition metric and the patient bootstrap used to compare models (03_metric_and_baselines, sections 1-2).

The score is the weighted multi-label log loss of the competition: per image, the log loss of the six labels with
weight 2 for `any` and 1 for each subtype, divided by 7, averaged over the images. Predictions are clipped to
[1e-15, 1 - 1e-15]. `y` and `p` are arrays of shape (n, 6) with the columns in the order of `LABELS`.
"""
import numpy as np

from ich.data.prepare import LABELS

WEIGHTS = np.array([2.0 if label == "any" else 1.0 for label in LABELS])
EPS = 1e-15


def label_losses(y, p) -> np.ndarray:
    """Log loss of every image and label, shape (n, 6)."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def image_losses(y, p) -> np.ndarray:
    """Weighted log loss of every image, shape (n,); their mean is the score."""
    return label_losses(y, p) @ WEIGHTS / WEIGHTS.sum()


def score(y, p) -> float:
    """The competition score."""
    return float(image_losses(y, p).mean())


def score_by_label(y, p) -> dict:
    """Mean log loss of each label and its weighted share of the score; the shares add up to the score."""
    mean = label_losses(y, p).mean(axis=0)
    return {label: {"log loss": float(m), "share of the score": float(m * w / WEIGHTS.sum())}
            for label, m, w in zip(LABELS, mean, WEIGHTS)}


def _group_sums(groups, *values):
    _, index = np.unique(np.asarray(groups), return_inverse=True)
    return [np.bincount(index, weights=v) for v in values], np.bincount(index)


def bootstrap(losses, groups, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of the image losses with a percentile interval from resampling whole groups (patients) with replacement.

    Slices of one scan are strongly correlated, so resampling slices would give far too narrow an interval.
    """
    (sums,), counts = _group_sums(groups, np.asarray(losses, dtype=float))
    rng, uniform = np.random.default_rng(seed), np.full(len(counts), 1 / len(counts))
    batch = max(1, 20_000_000 // len(counts))  # draws per batch, so that memory stays small even with one group per slice
    means = []
    for start in range(0, n_boot, batch):
        draws = rng.multinomial(len(counts), uniform, size=min(batch, n_boot - start))
        means.append((draws @ sums) / (draws @ counts))
    means = np.concatenate(means)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(sums.sum() / counts.sum()), "low": float(low), "high": float(high)}


def paired_bootstrap(losses_a, losses_b, groups, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Difference of two models' scores (b minus a) on the same images, with an interval from resampling groups."""
    difference = np.asarray(losses_b, dtype=float) - np.asarray(losses_a, dtype=float)
    return bootstrap(difference, groups, n_boot, seed, alpha)
