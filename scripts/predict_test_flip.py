"""Test-time augmentation of slice models (04_slice_models, 9): every test slice mirrored left to right, the flip of
training, read once and passed through all the given models. Reading the images is the slow part, not the models.

Usage: python scripts/predict_test_flip.py --runs RUN [RUN ...] [--batch 64] [--workers 12]

The runs need the same input (slices, windows, crop, size) and their test features from scripts/extract_features.py
(RUN/test_images.csv, test_logits.npy). Writes outputs/features/RUN/test_flip_logits.npy (float32, n x 6, in the order
of RUN/test_images.csv). As a check the first batches are also predicted without the flip and compared with
RUN/test_logits.npy.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from extract_features import channel_images, with_boxes
from ich.data.dataset import SliceDataset
from ich.data.prepare import LABELS, read_table
from ich.models.slice_model import SliceModel
from ich.paths import load_paths
from ich.training import prepare

CHECK_BATCHES = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def logits_of(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):  # as extract_features.py: backbone in bfloat16, head in float32
        pooled = model.backbone(x)
    return model.head(pooled.float()).cpu().numpy()


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    paths = load_paths()
    processed, features = paths["processed_data_dir"], paths["outputs_dir"] / "features"

    configs = {run: json.loads((paths["outputs_dir"] / "runs" / run / "config.json").read_text()) for run in args.runs}
    setup = {k: configs[args.runs[0]][k] for k in ["slices", "windows", "crop", "size"]}
    assert all({k: c[k] for k in setup} == setup for c in configs.values()), "the runs need the same input"

    table = read_table(processed / "slices.csv")
    frame = with_boxes(table[table["split"] == "test"].reset_index(drop=True), setup["crop"])
    for run in args.runs:
        assert pd.read_csv(features / run / "test_images.csv")["image"].equals(frame["image"]), run

    models = []
    for run in args.runs:
        model = SliceModel(configs[run]["backbone"], in_chans=setup["slices"] * len(setup["windows"]), pretrained=False)
        model.load_state_dict(torch.load(paths["models_dir"] / f"{run}.pt", weights_only=True))
        models.append(model.cuda().eval().to(memory_format=torch.channels_last))

    boxes = None if setup["crop"] == "none" else frame[["top", "left", "side"]].to_numpy()
    dataset = SliceDataset(channel_images(frame, setup["slices"]), processed / "png16", boxes, setup["size"])
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, pin_memory=True, timeout=120)
    flipped = np.zeros((len(args.runs), len(dataset), len(LABELS)), dtype=np.float32)
    plain = {run: 1 / (1 + np.exp(-np.load(features / run / "test_logits.npy"))) for run in args.runs}
    largest, start = 0.0, time.time()
    with torch.no_grad():
        for b, (stored, index) in enumerate(loader):
            x = prepare(stored.cuda(non_blocking=True), setup["windows"], setup["size"], train=False)
            i = index.numpy()
            for m, (run, model) in enumerate(zip(args.runs, models)):
                flipped[m, i] = logits_of(model, torch.flip(x, dims=[-1]))
                if b < CHECK_BATCHES:
                    same = 1 / (1 + np.exp(-logits_of(model, x)))
                    largest = max(largest, float(np.abs(same - plain[run][i]).max()))
            if b + 1 == CHECK_BATCHES:
                print(f"check without the flip, {CHECK_BATCHES * args.batch:,} slices x {len(args.runs)} models: "
                      f"largest probability difference to test_logits.npy {largest:.4f}", flush=True)
            if (b + 1) % 200 == 0:
                print(f"  {(b + 1) * args.batch:,}/{len(dataset):,} slices, "
                      f"{(b + 1) * args.batch / (time.time() - start):.0f} slices/s", flush=True)
    for m, run in enumerate(args.runs):
        np.save(features / run / "test_flip_logits.npy", flipped[m])
    print(f"done in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
