"""Turning stored slices into model inputs on the GPU, the loss, prediction and the batches of an epoch (04_slice_models).

The loader sends the stored uint16 values (HU + 1024); here they become windows in [0, 1], are augmented and
normalised. Order: window, then augment (padding with 0 is air in every window), then normalise.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

from ich.data.dicom import HU_MIN
from ich.data.windows import MODEL_WINDOWS
from ich.metrics import WEIGHTS

MEAN, STD = 0.45, 0.225  # about the ImageNet mean and std of the three colour channels, the same for every channel
LOSS_WEIGHTS = torch.tensor(WEIGHTS, dtype=torch.float32)


def to_windows(stored: torch.Tensor, windows: list[str], jitter: tuple[float, float] | None = None) -> torch.Tensor:
    """(B, slices, H, W) stored values -> (B, slices x windows, H, W) in [0, 1], slice by slice.

    `jitter` = (level in HU, relative width): every sample gets its own random shift of the level and scale of the width.
    """
    hu = stored.float() + HU_MIN
    level = torch.tensor([MODEL_WINDOWS[w][0] for w in windows], device=hu.device, dtype=torch.float32)
    width = torch.tensor([MODEL_WINDOWS[w][1] for w in windows], device=hu.device, dtype=torch.float32)
    level, width = level.expand(len(hu), -1), width.expand(len(hu), -1)
    if jitter is not None:
        level = level + (torch.rand_like(level) * 2 - 1) * jitter[0]
        width = width * (1 + (torch.rand_like(width) * 2 - 1) * jitter[1])
    low = (level - width / 2)[:, None, :, None, None]
    x = ((hu[:, :, None] - low) / width[:, None, :, None, None]).clamp(0, 1)  # (B, slices, windows, H, W)
    return x.flatten(1, 2)


def augment(x: torch.Tensor, rotate: float = 15, scale: float = 0.1, shift: float = 0.05) -> torch.Tensor:
    """Random horizontal flip, rotation (degrees), zoom and shift (share of the side), one draw per sample."""
    n = len(x)
    angle = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(rotate)
    zoom = 1 + (torch.rand(n, device=x.device) * 2 - 1) * scale
    flip = torch.where(torch.rand(n, device=x.device) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    offset = (torch.rand(n, 2, device=x.device) * 2 - 1) * shift * 2  # grid coordinates run from -1 to 1
    theta = torch.stack([torch.stack([cos * flip, -sin, offset[:, 0]], 1),
                         torch.stack([sin * flip, cos, offset[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def normalise(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN) / STD


def prepare(stored, windows, size, train: bool, jitter=None) -> torch.Tensor:
    """Stored values on the GPU -> model input. Resized to `size` if the source is larger (full images without a crop)."""
    x = to_windows(stored, windows, jitter if train else None)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    if train:
        x = augment(x)
    return normalise(x).contiguous(memory_format=torch.channels_last)


def weighted_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The competition loss: binary cross-entropy per label, `any` counted twice, divided by 7, mean over the batch."""
    losses = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
    weights = LOSS_WEIGHTS.to(logits.device)
    return (losses @ weights / weights.sum()).mean()


@torch.no_grad()
def predict_batch(model, stored, windows, size):
    """Probabilities and pooled features of one batch of stored values, as float32 numpy arrays."""
    x = prepare(stored.cuda(non_blocking=True), windows, size, train=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, pooled = model(x)
    return torch.sigmoid(logits.float()).cpu().numpy(), pooled.float().cpu().numpy()


@torch.no_grad()
def predict(model, loader, windows, size, embeddings: bool = False):
    """Probabilities (n, 6) in the order of the dataset, and the pooled features (n, d) as float16 if asked."""
    model.eval()
    n = len(loader.dataset)
    probs, features = np.zeros((n, 6), dtype=np.float32), None
    for stored, index in loader:
        p, pooled = predict_batch(model, stored, windows, size)
        probs[index.numpy()] = p
        if embeddings:
            if features is None:
                features = np.zeros((n, pooled.shape[1]), dtype=np.float16)
            features[index.numpy()] = pooled.astype(np.float16)
    return probs, features


class EpochBatches(Sampler):
    """The batches of one epoch over a ConcatDataset of (train, val): the shuffled training batches (the last
    incomplete step of `accumulate` batches dropped), then the validation batches in order.

    One loader, and so one set of workers, serves both. On Windows a second set of persistent workers for validation
    pushed the memory committed past the limit (error 1455): every worker commits about 2 GB, the training process
    about 28 GB with the GPU memory counted in (04).
    """

    def __init__(self, n_train: int, n_val: int, train_batch: int, val_batch: int, seed: int, accumulate: int = 1):
        self.n_train, self.n_val, self.train_batch, self.val_batch = n_train, n_val, train_batch, val_batch
        self.n_train_batches = n_train // (train_batch * accumulate) * accumulate  # whole optimizer steps only
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        order = torch.randperm(self.n_train, generator=self.generator).tolist()
        for b in range(self.n_train_batches):
            yield order[b * self.train_batch:(b + 1) * self.train_batch]
        for start in range(0, self.n_val, self.val_batch):
            yield list(range(self.n_train + start, self.n_train + min(start + self.val_batch, self.n_val)))

    def __len__(self):
        return self.n_train_batches + math.ceil(self.n_val / self.val_batch)
