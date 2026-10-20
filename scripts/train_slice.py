"""Train one slice model on one fold and predict its validation fold (04_slice_models).

Usage: python scripts/train_slice.py --name NAME [--fold 0] [--backbone seresnext50_32x4d] [--slices 1|3]
       [--windows brain subdural bone] [--crop series|slice|slice256|none] [--size 256] [--epochs 3] [--train-share 0.75]
       [--batch 192] [--accumulate 1] [--lr 3.5e-4] [--dropout 0.2] [--jitter] [--seed 0] [--embeddings] [--val-full] [--resume]

- Training: the kept slices of the other folds with the labels of their position (03), optionally a fixed share of
  their patients (drawn with its own seed, by writer and by whether the patient has a bleed, so that every run with
  the same share sees the same patients). With `--crop series` or `slice256` at `--size 256` they come from a cropped store.
- Validation: every slice of the fold, scored after every epoch on the kept slices (decisions) and on all slices with
  their own labels; the AUROC of each label on the kept slices is logged beside them (for information, not for
  decisions). From the cropped store when the setup matches it (its files equal the crops made on the fly, checked
  once for every slice of fold 0); otherwise, and always with --val-full (final models), from the full images cropped
  on the fly, as the test images are read (user decision, 2026-10-16).

Writes outputs/runs/NAME/: config.json, log.csv (one row per epoch), val.csv (the validation slices in prediction
order), val_epoch<k>.npy (probabilities), writers.csv, inputs.png (the first training batch, for a visual check) and,
with --embeddings, val_features.npy of the last epoch. The weights go to models/NAME.pt.

After every epoch the state of the run (weights, optimizer, schedule, random generators, log) goes to
models/NAME.ckpt, removed when the run ends; with --resume a run stopped midway continues after its last whole epoch.
"""
import argparse
import json
import math
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader

from ich.data.crop import MIN_SIDE, series_boxes, slice_boxes
from ich.data.dataset import SliceDataset, neighbour_images
from ich.data.prepare import LABELS, load_labels, read_table
from ich.metrics import score
from ich.models.slice_model import SliceModel
from ich.paths import load_paths
from ich.training import EpochBatches, predict_batch, prepare, weighted_loss

SUBSET_SEED = 0
STORES = {"series": "png16_crop256", "slice256": "png16_slice256"}  # scripts/build_crop_store.py


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--backbone", default="seresnext50_32x4d")
    parser.add_argument("--slices", type=int, choices=[1, 3], default=1)
    parser.add_argument("--windows", nargs="+", default=["brain", "subdural", "bone"])
    parser.add_argument("--crop", choices=["series", "slice", "slice256", "none"], default="series",
                        help=f"box of the series, box of the (centre) slice, the same of at least {MIN_SIDE} px, or the full image")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-share", type=float, default=0.75)
    parser.add_argument("--batch", type=int, default=192)  # about 60% of the 32 GB at 256 px (04)
    parser.add_argument("--accumulate", type=int, default=1,
                        help="split every batch into this many parts with their gradients summed (larger inputs)")
    parser.add_argument("--lr", type=float, default=3.5e-4)  # 2e-4 at batch 64, scaled by the square root of the batch
    parser.add_argument("--dropout", type=float, default=0.2, help="before the last layer")
    parser.add_argument("--jitter", action="store_true", help="window jitter: level +-10 HU, width +-10%%")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--val-full", action="store_true",
                        help="read validation from the full images, cropped on the fly, as the test images are")
    parser.add_argument("--workers", type=int, default=8)  # each commits ~2 GB on Windows; 8 still feed 3 slices at ~1,300/s (04)
    parser.add_argument("--resume", action="store_true", help="continue after the last epoch in models/NAME.ckpt")
    parser.add_argument("--limit", type=int, default=0, help="smoke test: use only this many training and validation slices")
    return parser.parse_args()


def patient_subset(train: pd.DataFrame, share: float) -> np.ndarray:
    """A fixed share of the patients, drawn within each writer and bleed / no bleed group."""
    if share >= 1:
        return np.ones(len(train), dtype=bool)
    patients = train.groupby("patient").agg(writer=("writer", "first"), bleed=("any", "max"))
    rng = np.random.default_rng(SUBSET_SEED)
    chosen = [rng.choice(g.index.to_numpy(), size=round(len(g) * share), replace=False)
              for _, g in patients.groupby(["writer", "bleed"])]
    return train["patient"].isin(np.concatenate(chosen)).to_numpy()


def channel_images(frame: pd.DataFrame, slices: int) -> np.ndarray:
    if slices == 1:
        return frame[["image"]].to_numpy()
    return pd.concat([frame["below"], frame["image"], frame["above"]], axis=1).to_numpy()


def save_inputs(x: torch.Tensor, path, n: int = 8):
    """The first n samples of a batch, every channel side by side, as one grey image (after augmentation)."""
    from ich.training import MEAN, STD
    grid = (x[:n].float() * STD + MEAN).clamp(0, 1).cpu().numpy()
    rows = [np.concatenate(list(sample), axis=1) for sample in grid]
    cv2.imwrite(str(path), (np.concatenate(rows, axis=0) * 255).astype(np.uint8))


def main():
    args = parse_args()
    torch.manual_seed(args.seed), np.random.seed(args.seed), random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    paths = load_paths()
    processed = paths["processed_data_dir"]
    run_dir = paths["outputs_dir"] / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (paths["models_dir"]).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    table = read_table(processed / "slices.csv")
    table = table[table["split"] == "train"].reset_index(drop=True)
    table = table.join(neighbour_images(table))
    if args.crop == "series":
        table = table.join(series_boxes(table), on="series")
    else:
        table = table.join(slice_boxes(table, min_side=MIN_SIDE if args.crop == "slice256" else 0))
    own = load_labels(paths["raw_data_dir"] / "stage_2_train.csv")

    train = table[(table["fold"] != args.fold) & table["keep"]]
    train = train[patient_subset(train, args.train_share)].reset_index(drop=True)
    val = table[table["fold"] == args.fold].reset_index(drop=True)
    if args.limit:
        train = train.sample(min(args.limit, len(train)), random_state=0).reset_index(drop=True)
        val = val.sample(min(args.limit, len(val)), random_state=0).reset_index(drop=True)
    y_train = torch.tensor(train[LABELS].astype(float).to_numpy(), dtype=torch.float32)
    y_kept = val[LABELS].astype(float).to_numpy()
    y_own = own.loc[val["image"], LABELS].astype(float).to_numpy()
    kept = val["keep"].to_numpy()
    val[["image", "patient", "writer", "keep"]].to_csv(run_dir / "val.csv", index=False)
    print(f"train: {len(train):,} slices, {train['patient'].nunique():,} patients | "
          f"val: {len(val):,} slices ({kept.sum():,} kept), {val['patient'].nunique():,} patients", flush=True)

    in_store = args.crop in STORES and args.size == 256

    def dataset(frame, from_store):
        if from_store:
            return SliceDataset(channel_images(frame, args.slices), processed / STORES[args.crop])
        boxes = None if args.crop == "none" else frame[["top", "left", "side"]].to_numpy()
        return SliceDataset(channel_images(frame, args.slices), processed / "png16", boxes, args.size)

    train_set, val_set = dataset(train, in_store), dataset(val, in_store and not args.val_full)
    # validation in batches of the training size: a new, larger shape made cuDNN allocate more GPU memory, which
    # Windows counts against the commit limit (04)
    assert args.batch % args.accumulate == 0, "--batch must be a multiple of --accumulate"
    part = args.batch // args.accumulate
    batches = EpochBatches(len(train_set), len(val_set), part, part, args.seed, args.accumulate)
    loader = DataLoader(ConcatDataset([train_set, val_set]), batch_sampler=batches, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True, timeout=120)

    model = SliceModel(args.backbone, in_chans=args.slices * len(args.windows), dropout=args.dropout).cuda().to(memory_format=torch.channels_last)
    rate = y_train.mean(0).clamp(1e-4, 1 - 1e-4)
    with torch.no_grad():
        model.head.bias.copy_(torch.log(rate / (1 - rate)))  # start from the label rates
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * batches.n_train_batches // args.accumulate  # optimizer steps
    warmup = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    jitter = (10.0, 0.1) if args.jitter else None

    log, step, first = [], 0, 1
    checkpoint = paths["models_dir"] / f"{args.name}.ckpt"
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(state["model"]), optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"]), batches.generator.set_state(state["batches"])
        torch.set_rng_state(state["cpu"]), torch.cuda.set_rng_state(state["cuda"])
        np.random.set_state(state["numpy"]), random.setstate(state["random"])
        log, step, first = state["log"], state["step"], state["epoch"] + 1
        print(f"resumed after epoch {state['epoch']}", flush=True)
    for epoch in range(first, args.epochs + 1):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        last = epoch == args.epochs
        probs, features = np.zeros((len(val_set), 6), dtype=np.float32), None
        for b, (stored, index) in enumerate(loader):
            if b >= batches.n_train_batches:  # validation batches, after the training batches of the epoch
                if b == batches.n_train_batches:
                    train_time, start = time.time() - start, time.time()
                    model.eval()
                p, pooled = predict_batch(model, stored, args.windows, args.size)
                probs[index.numpy()] = p
                if args.embeddings and last:
                    if features is None:
                        features = np.zeros((len(val_set), pooled.shape[1]), dtype=np.float16)
                    features[index.numpy()] = pooled.astype(np.float16)
                continue
            x = prepare(stored.cuda(non_blocking=True), args.windows, args.size, train=True, jitter=jitter)
            y = y_train[index].cuda(non_blocking=True)
            if step == 0 and b == 0:
                save_inputs(x, run_dir / "inputs.png")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            loss = weighted_loss(logits, y)
            (loss / args.accumulate).backward()
            running, seen = running + loss.item() * len(x), seen + len(x)
            if (b + 1) % args.accumulate:
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            schedule.step()
            step += 1
            if step % 500 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {running / seen:.4f} "
                      f"{seen / (time.time() - start):.0f} slices/s", flush=True)

        np.save(run_dir / f"val_epoch{epoch}.npy", probs)
        row = {"epoch": epoch, "train loss": running / seen, "kept": score(y_kept[kept], probs[kept]),
               "all": score(y_own, probs), "train min": train_time / 60, "val min": (time.time() - start) / 60,
               "peak GB": torch.cuda.max_memory_allocated() / 1e9, "reserved GB": torch.cuda.max_memory_reserved() / 1e9}
        aucs = {f"auc {label}": roc_auc_score(y_kept[kept, j], probs[kept, j]) for j, label in enumerate(LABELS)}
        row.update({"auc mean": float(np.mean(list(aucs.values()))), **aucs})
        log.append(row)
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                 "batches": batches.generator.get_state(), "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                 "numpy": np.random.get_state(), "random": random.getstate(), "log": log, "step": step, "epoch": epoch}
        torch.save(state, str(checkpoint) + ".tmp")
        os.replace(str(checkpoint) + ".tmp", checkpoint)  # a stop while writing leaves the previous checkpoint whole
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)

    if first > args.epochs:  # stopped after the checkpoint of the last epoch, before the end of the run
        probs, features = np.load(run_dir / f"val_epoch{args.epochs}.npy"), None
    if features is not None:
        np.save(run_dir / "val_features.npy", features)
    writers = pd.DataFrame({"writer": val["writer"][kept], "true any": y_kept[kept, -1], "mean predicted any": probs[kept, -1]})
    writers = writers.groupby("writer").agg(slices=("true any", "size"), **{"true any": ("true any", "mean"),
                                                                             "mean predicted any": ("mean predicted any", "mean")})
    writers["score"] = [score(y_kept[kept][(val["writer"][kept] == w).to_numpy()], probs[kept][(val["writer"][kept] == w).to_numpy()])
                        for w in writers.index]
    writers.to_csv(run_dir / "writers.csv")
    print(writers.round(4).to_string(), flush=True)
    torch.save(model.state_dict(), paths["models_dir"] / f"{args.name}.pt")
    checkpoint.unlink()


if __name__ == "__main__":
    main()
