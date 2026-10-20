"""Extract the pooled features and logits of a trained slice model, the inputs of the sequence models (05_sequence_models).

Usage: python scripts/extract_features.py --name RUN [--parts oof train test] [--flip] [--batch 64] [--workers 12]

The setup (fold, backbone, slices, windows, crop, size) comes from outputs/runs/RUN/config.json, the weights from
models/RUN.pt. Every slice is read from the full images and cropped on the fly, as the test images are. Parts:
- oof: every slice of the run's validation fold (patients the model never saw);
- train: every slice of the other folds (in-sample: the model was trained on them; runs on 100% of the patients only);
- test: every test slice.

Writes outputs/features/RUN/PART_features.npy (float16, n x d), PART_logits.npy (float32, n x 6) and PART_images.csv
(image, series, fold, keep), in the same order. With --flip every slice is mirrored left to right first (the
flip of training, used as test-time augmentation) and the files are named PART_flip_*. A part that is already written is skipped, so a stopped run continues
with the next part. For `oof` the probabilities are checked against the run's validation predictions.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ich.data.crop import MIN_SIDE, series_boxes, slice_boxes
from ich.data.dataset import SliceDataset, neighbour_images
from ich.data.prepare import LABELS, read_table
from ich.metrics import score
from ich.models.slice_model import SliceModel
from ich.paths import load_paths
from ich.training import prepare

PARTS = ["oof", "train", "test"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS)
    parser.add_argument("--flip", action="store_true", help="mirror every slice left to right (test-time augmentation)")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def channel_images(frame: pd.DataFrame, slices: int) -> np.ndarray:
    if slices == 1:
        return frame[["image"]].to_numpy()
    return pd.concat([frame["below"], frame["image"], frame["above"]], axis=1).to_numpy()


def with_boxes(table: pd.DataFrame, crop: str) -> pd.DataFrame:
    table = table.join(neighbour_images(table))
    if crop == "series":
        return table.join(series_boxes(table), on="series")
    return table.join(slice_boxes(table, min_side=MIN_SIDE if crop == "slice256" else 0))


@torch.no_grad()
def extract(model, frame, config, processed, batch, workers, flip=False):
    boxes = None if config["crop"] == "none" else frame[["top", "left", "side"]].to_numpy()
    dataset = SliceDataset(channel_images(frame, config["slices"]), processed / "png16", boxes, config["size"])
    loader = DataLoader(dataset, batch_size=batch, num_workers=workers, pin_memory=True, timeout=120)
    features = np.zeros((len(dataset), model.backbone.num_features), dtype=np.float16)
    logits = np.zeros((len(dataset), len(LABELS)), dtype=np.float32)
    start = time.time()
    for b, (stored, index) in enumerate(loader):
        x = prepare(stored.cuda(non_blocking=True), config["windows"], config["size"], train=False)
        if flip:
            x = torch.flip(x, dims=[-1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pooled = model.backbone(x)
        pooled = pooled.float()
        features[index.numpy()] = pooled.cpu().numpy().astype(np.float16)
        logits[index.numpy()] = model.head(pooled).cpu().numpy()  # head in float32; dropout is off in eval
        if (b + 1) % 500 == 0:
            print(f"  {(b + 1) * batch:,}/{len(dataset):,} slices, {(b + 1) * batch / (time.time() - start):.0f} slices/s", flush=True)
    return features, logits


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    paths = load_paths()
    processed = paths["processed_data_dir"]
    run_dir = paths["outputs_dir"] / "runs" / args.name
    out_dir = paths["outputs_dir"] / "features" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "config.json").read_text())

    table = read_table(processed / "slices.csv")
    train_rows = with_boxes(table[table["split"] == "train"].reset_index(drop=True), config["crop"])
    frames = {"oof": train_rows[train_rows["fold"] == config["fold"]].reset_index(drop=True),
              "train": train_rows[train_rows["fold"] != config["fold"]].reset_index(drop=True),
              "test": with_boxes(table[table["split"] == "test"].reset_index(drop=True), config["crop"])}
    if "train" in args.parts:
        assert config["train_share"] >= 1, "in-sample features need a run trained on all patients of its folds"

    model = SliceModel(config["backbone"], in_chans=config["slices"] * len(config["windows"]), pretrained=False)
    model.load_state_dict(torch.load(paths["models_dir"] / f"{args.name}.pt", weights_only=True))
    model = model.cuda().eval().to(memory_format=torch.channels_last)

    for part in args.parts:
        name = f"{part}_flip" if args.flip else part
        if (out_dir / f"{name}_images.csv").exists():
            print(f"{name}: already written, skipped", flush=True)
            continue
        frame = frames[part]
        print(f"{name}: {len(frame):,} slices", flush=True)
        start = time.time()
        features, logits = extract(model, frame, config, processed, args.batch, args.workers, args.flip)
        np.save(out_dir / f"{name}_features.npy", features)
        np.save(out_dir / f"{name}_logits.npy", logits)
        frame[["image", "series", "fold", "keep"]].to_csv(out_dir / f"{name}_images.csv", index=False)  # written last: marks the part done
        print(f"{name}: done in {(time.time() - start) / 60:.1f} min, features {features.shape}", flush=True)

        if part == "oof" and not args.flip:  # same slices, same order as the run's validation
            val = pd.read_csv(run_dir / "val.csv")
            assert val["image"].equals(frame["image"])
            reference = np.load(run_dir / f"val_epoch{config['epochs']}.npy")
            probs = 1 / (1 + np.exp(-logits))
            kept = frame["keep"].to_numpy()
            y = frame.loc[kept, LABELS].astype(float).to_numpy()
            print(f"oof check: kept score {score(y, probs[kept]):.4f} (run: {score(y, reference[kept]):.4f}), "
                  f"largest probability difference {np.abs(probs - reference).max():.4f}", flush=True)


if __name__ == "__main__":
    main()
