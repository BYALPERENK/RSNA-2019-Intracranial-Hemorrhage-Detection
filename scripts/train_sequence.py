"""Train one sequence model on the features of one slice model and predict its validation fold (05_sequence_models).

Usage: python scripts/train_sequence.py --name NAME --cnn RUN [RUN...] [--source insample|oofcv|pooled] [--folds 0 1 2 3 4]
       [--inputs features logits]
       [--arch gru|lstm|tcn|transformer] [--proj 256] [--hidden 256] [--dropout 0.3] [--gaps] [--plain] [--epochs 10]
       [--batch 64] [--lr 1e-3] [--warmup 0.5] [--two-branch] [--reverse 0.5] [--subsample 0.5] [--reverse-tta]
       [--seed 0] [--stop 0.1] [--resume]

- Features from scripts/extract_features.py (outputs/features/RUN/). `--source insample` (the setup of 05): training
  on the CNN's own training folds ("train" part), validation on its fold ("oof" part); nothing in the pipeline saw
  the validation patients. `--source oofcv`: only the CNN's out-of-fold features of its fold, which the CNN never
  saw; the fold is split by patient into 5 inner folds, one sequence model per inner fold trains on the other four
  and predicts its own, so every series of the fold is predicted by a model that did not see it. Test: the mean of
  the models.
  `--source pooled`: `--cnn` names the five fold models; the out-of-fold features of all of them
  form one table, and for every fold k in `--folds` one sequence model trains on the other folds (each slice with
  the features of the CNN that did not see it) and predicts fold k from the features of CNN_k. Test: the model of
  fold k on the test features of CNN_k, mean over the folds.
- Test flip: a model on the logits only (`--inputs logits`) also predicts from the logits of the flipped test images
  (test_flip_logits.npy of scripts/predict_test_flip.py) when they exist, into test_flip_probs.npy.
- Several slice models with `--source insample` / `oofcv` (models of the same fold, an early ensemble): their
  features and logits side by side for every slice (the logits input), the mean of the logits the base of the
  residual head.
- `--two-branch`: branch 1 on the features alone, branch 2 on the logits and branch 1's logits; the loss of branch 1
  is added to the loss (deep supervision), the output is branch 2's.
- Augmentation of the training series: `--reverse P` reverses the slice order of a series with probability P,
  `--subsample P` keeps every other slice (odd or even at random) of a series of more than 10 slices with
  probability P (the dropped slices are left out of the loss of that step). `--reverse-tta`: also predict every
  series in reversed order, into val_rev_probs.npy / test_rev_probs.npy (test_flip_rev_probs.npy); the reverse TTA
  is the mean with the forward prediction, left to the scripts that read the run.
- One sequence per series: its kept slices in order (one per position). Every other slice of a position gets the
  prediction of the kept one, for the all-slices score and the test.
- Loss: the competition loss per slice, padding left out, averaged over the real slices of the batch.
- Fixed number of epochs, no choice on the validation fold; the score of every epoch is logged for information.
  With `--stop 0.1` a tenth of the training patients is held out, and each model predicts with its epoch best on
  them (the small oofcv training sets overfit after a few epochs).

Writes outputs/seq_runs/NAME/: config.json, log.csv, val.csv (the validation slices, `predicted` = predicted by a
model) with val_probs.npy (last epoch), and test_probs.npy (test_flip_probs.npy) if the CNN's test features exist; the weights (a list, one per model) in
models/seq_NAME.pt. State after every epoch in models/seq_NAME.ckpt (removed at the end); --resume continues after
the last whole epoch.
"""
import argparse
import copy
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch

from ich.data.prepare import LABELS, load_labels, read_table
from ich.metrics import score
from ich.models.sequence_model import SequenceModel
from ich.paths import load_paths
from ich.training import LOSS_WEIGHTS

MAX_LEN = 60  # longest series, train and test (kept slices)
INNER, INNER_SEED = 5, 0  # --source oofcv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--cnn", required=True, nargs="+",
                        help="run name of the slice model whose features are used; with pooled, one per fold")
    parser.add_argument("--source", choices=["insample", "oofcv", "pooled"], default="insample")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="pooled: the folds to predict")
    parser.add_argument("--inputs", nargs="+", choices=["features", "logits"], default=["features", "logits"])
    parser.add_argument("--proj", type=int, default=256, help="projection of the features; 0 = the full vector")
    parser.add_argument("--arch", choices=["gru", "lstm", "tcn", "transformer"], default="gru",
                        help="the sequence encoder (ich.models.sequence_model)")
    parser.add_argument("--hidden", type=int, default=256, help="per direction for gru / lstm, the width otherwise")
    parser.add_argument("--dropout", type=float, default=0.3, help="on the (projected) features")
    parser.add_argument("--gaps", action="store_true", help="gaps to the neighbouring positions (mm) as side inputs")
    parser.add_argument("--plain", action="store_true", help="predict the logits directly, not a correction of the CNN's")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64, help="series per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=float, default=0.5, help="linear warm-up, in epochs")
    parser.add_argument("--two-branch", action="store_true", help="features branch + logits branch, deep supervision")
    parser.add_argument("--reverse", type=float, default=0.0, help="probability of reversing a training series")
    parser.add_argument("--subsample", type=float, default=0.0,
                        help="probability of keeping every other slice of a training series (> 10 slices)")
    parser.add_argument("--reverse-tta", action="store_true", help="also predict every series in reversed order")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop", type=float, default=0.0,
                        help="hold out this share of the training patients and predict with the epoch best on them")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


class Part:
    """The features of one part (train / oof / test) of one or more slice models as padded series on the GPU.
    layout "rows": the models cover different slices (the fold models of pooled), one table below the other;
    "columns": the models cover the same slices (an ensemble), features and logits side by side (`logits_in`), the
    mean of the logits the base of the residual head."""

    def __init__(self, features_dirs, part, table, gaps: bool, logits: str = None, layout: str = "rows"):
        frames = [pd.read_csv(d / f"{part}_images.csv") for d in features_dirs]
        if layout == "columns":
            assert all(f["image"].equals(frames[0]["image"]) for f in frames), "an ensemble needs the same slices"
            images = frames[0]
        else:
            images = pd.concat(frames, ignore_index=True)
        assert not images["image"].duplicated().any()
        info = table.loc[images["image"], ["series", "position", "slice", "gap_prev", "gap_next", "local_spacing"]]
        info = info.reset_index().assign(row=np.arange(len(images)), keep=images["keep"].to_numpy(),
                                         fold=images["fold"].to_numpy() if "fold" in images else -1)
        kept = info[info["keep"]].sort_values(["series", "slice"])
        self.images, self.info = images, info
        self.series = kept["series"].unique()
        self.lengths = kept.groupby("series", sort=False).size().loc[self.series].to_numpy()
        assert self.lengths.max() <= MAX_LEN
        # rows[i, t]: row of the t-th kept slice of series i in the feature arrays, -1 = padding
        rows = np.full((len(self.series), MAX_LEN), -1, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        t = np.arange(len(kept)) - np.repeat(starts, self.lengths)
        rows[np.repeat(np.arange(len(self.series)), self.lengths), t] = kept["row"].to_numpy()
        self.rows = torch.from_numpy(rows).cuda()
        # every slice -> the row of the kept slice of its position
        key = kept[["series", "position", "row"]].rename(columns={"row": "kept_row"})
        self.kept_row = info[["series", "position"]].merge(key, on=["series", "position"], how="left")["kept_row"].to_numpy()
        assert not np.isnan(self.kept_row.astype(float)).any(), "every slice needs a kept slice at its position"

        logits = logits or f"{part}_logits"
        features = [np.load(d / f"{part}_features.npy") for d in features_dirs]  # float16
        all_logits = [np.load(d / f"{logits}.npy") for d in features_dirs]
        if layout == "columns":
            self.features = torch.from_numpy(np.concatenate(features, axis=1)).cuda()
            self.logits = torch.from_numpy(np.mean(all_logits, axis=0).astype(np.float32)).cuda()
            self.logits_in = torch.from_numpy(np.concatenate(all_logits, axis=1).astype(np.float32)).cuda()
        else:
            self.features = torch.from_numpy(np.concatenate(features)).cuda()
            self.logits = torch.from_numpy(np.concatenate(all_logits)).cuda()
            self.logits_in = self.logits
        if gaps:
            g = info[["gap_prev", "gap_next"]].fillna(0).clip(0, 10).to_numpy() / 5  # 0 at the ends of the scan
            self.side = torch.tensor(g, dtype=torch.float32).cuda()
        else:
            self.side = torch.zeros((len(images), 0)).cuda()

    def batch(self, index, reverse=None, subsample=None):
        """The padded series `index`. reverse: (B,) bool, the series to reverse; subsample: (B,) int, -1 = all slices,
        0 / 1 = every other slice from the first / second one; subsample first, then reverse."""
        rows = self.rows[index]
        lengths = torch.from_numpy(self.lengths[index.cpu().numpy()])
        if subsample is not None:
            rows, lengths = every_other(rows, lengths, subsample)
        if reverse is not None:
            rows = reversed_rows(rows, lengths, reverse)
        rows = rows[:, :int(lengths.max())]
        safe = rows.clamp(min=0)
        side = self.side[safe]
        if reverse is not None and side.shape[-1]:  # the gap to the previous slice becomes the gap to the next one
            side = torch.where(reverse.to(side.device)[:, None, None], side.flip(-1), side)
        return (self.features[safe].float(), self.logits[safe], side, lengths, rows, self.logits_in[safe])


def reversed_rows(rows, lengths, which):
    """rows (B, T) with the first lengths[i] entries of row i reversed where `which` (B,) is true."""
    t = torch.arange(rows.shape[1], device=rows.device)[None, :]
    n = lengths.to(rows.device)[:, None]
    source = torch.where(which.to(rows.device)[:, None] & (t < n), n - 1 - t, t)
    return rows.gather(1, source)


def every_other(rows, lengths, which):
    """rows (B, T), lengths (B,): where `which` (B,) is 0 or 1, keep the slices which[i], which[i] + 2, ..."""
    out, new = torch.full_like(rows, -1), lengths.clone()
    for i in range(len(lengths)):
        n, start = int(lengths[i]), int(which[i])
        kept = rows[i, start:n:2] if start >= 0 else rows[i, :n]
        out[i, :len(kept)], new[i] = kept, len(kept)
    return out, new


def masked_loss(logits, y, rows):
    mask = (rows >= 0).float()
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    weights = LOSS_WEIGHTS.to(logits.device)
    per_slice = losses @ weights / weights.sum()
    return (per_slice * mask).sum() / mask.sum()


def series_patients(part, table) -> np.ndarray:
    return table.reset_index().groupby("series")["patient"].first().loc[part.series].to_numpy()


@torch.no_grad()
def predict(model, part, batch, series=None, reverse=False):
    """Probabilities for every slice of the given series of the part (all series by default; kept slices from the
    model, the others copied from their position). Slices of other series stay 0. `reverse`: every series in
    reversed order."""
    model.eval()
    series = np.arange(len(part.series)) if series is None else np.asarray(series)
    out = torch.zeros((len(part.images), len(LABELS)), device="cuda")
    for start in range(0, len(series), batch):
        index = torch.from_numpy(series[start:start + batch]).cuda()
        which = torch.ones(len(index), dtype=torch.bool) if reverse else None
        features, logits, side, lengths, rows, logits_in = part.batch(index, reverse=which)
        p = torch.sigmoid(model(features, logits, side, lengths, logits_in))
        valid = rows >= 0
        out[rows[valid]] = p[valid]
    return out.cpu().numpy()[part.kept_row]


def patient_groups(patients: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Group (0..n-1) of every entry, by patient: all series of a patient in one group; fixed by `seed`."""
    unique = np.random.default_rng(seed).permutation(pd.unique(patients))
    return pd.Series(np.arange(len(unique)) % n, index=unique).loc[patients].to_numpy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed), np.random.seed(args.seed), random.seed(args.seed)
    paths = load_paths()
    features_dirs = [paths["outputs_dir"] / "features" / run for run in args.cnn]
    layout = "rows" if args.source == "pooled" else "columns"  # several slice models: fold models or an ensemble
    run_dir = paths["outputs_dir"] / "seq_runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    table = read_table(paths["processed_data_dir"] / "slices.csv").set_index("image")
    own = load_labels(paths["raw_data_dir"] / "stage_2_train.csv")
    val = Part(features_dirs, "oof", table, args.gaps, layout=layout)
    kept = val.images["keep"].to_numpy()
    y_kept = table.loc[val.images["image"], LABELS].astype(float).to_numpy()
    y_own = own.loc[val.images["image"], LABELS].astype(float).to_numpy()
    cnn = 1 / (1 + np.exp(-val.logits.cpu().numpy()))
    has_test = all((d / "test_images.csv").exists() for d in features_dirs)
    flip = "features" not in args.inputs and all((d / "test_flip_logits.npy").exists() for d in features_dirs)

    if has_test:  # the mean over the models needs the same test slices in the same order for every slice model
        test_images = pd.read_csv(features_dirs[0] / "test_images.csv")
        assert all(pd.read_csv(d / "test_images.csv")["image"].equals(test_images["image"]) for d in features_dirs)
        test_images.to_csv(run_dir / "test.csv", index=False)

    def tests(dirs):
        """The test parts a model predicts: name of the output -> part."""
        if not has_test:
            return {}
        parts = {"test_probs": Part(dirs, "test", table, args.gaps, layout=layout)}
        if flip:
            parts["test_flip_probs"] = Part(dirs, "test", table, args.gaps, logits="test_flip_logits", layout=layout)
        return parts

    # jobs: (id, training part, the series it trains on, the series of the validation part it predicts, test part)
    if args.source == "insample":
        train = Part(features_dirs, "train", table, args.gaps, layout=layout)
        test = tests(features_dirs)
        jobs = [(0, train, np.arange(len(train.series)), np.arange(len(val.series)), test)]
    elif args.source == "oofcv":
        train = val
        test = tests(features_dirs)
        inner = patient_groups(series_patients(val, table), INNER, INNER_SEED)
        jobs = [(j, val, np.flatnonzero(inner != j), np.flatnonzero(inner == j), test) for j in range(INNER)]
    else:
        train = val
        series_fold = val.info.groupby("series")["fold"].first().loc[val.series].to_numpy()
        run_of_fold = {}
        for d in features_dirs:
            folds = pd.read_csv(d / "oof_images.csv", usecols=["fold"])["fold"].unique()
            assert len(folds) == 1 and folds[0] not in run_of_fold, f"{d.name}: one fold per slice model"
            run_of_fold[int(folds[0])] = d
        assert sorted(run_of_fold) == list(range(5)), "pooled needs the five fold models"
        jobs = [(k, val, np.flatnonzero(series_fold != k), np.flatnonzero(series_fold == k),
                 tests([run_of_fold[k]])) for k in args.folds]
    y_train = torch.tensor(table.loc[train.images["image"], LABELS].astype(float).to_numpy(), dtype=torch.float32).cuda()
    series_of_row = pd.Series(np.arange(len(val.series)), index=val.series).loc[val.info["series"]].to_numpy()
    done = np.isin(series_of_row, np.concatenate([job[3] for job in jobs]))  # the slices the models predict
    print(f"{args.source}: {len(jobs)} model(s), {len(jobs[0][2]):,} training series each | val: {done.sum():,} "
          f"slices | CNN alone: kept {score(y_kept[done & kept], cnn[done & kept]):.4f}, "
          f"all {score(y_own[done], cnn[done]):.4f}", flush=True)

    log, first_job, first_epoch = [], 0, 1
    probs, test_sum, weights = np.zeros((len(val.images), len(LABELS)), dtype=np.float32), {}, []
    probs_rev = np.zeros_like(probs)
    checkpoint = paths["models_dir"] / f"seq_{args.name}.ckpt"
    state = torch.load(checkpoint, weights_only=False) if args.resume and checkpoint.exists() else None
    if state is not None:
        log, probs, test_sum, weights = state["log"], state["probs"], state["test_sum"], state["weights"]
        probs_rev = state.get("probs_rev", probs_rev)
        first_job, first_epoch = (state["job"], state["epoch"] + 1) if state["epoch"] < args.epochs else (state["job"] + 1, 1)
        print(f"resumed after model {state['job']} epoch {state['epoch']}", flush=True)
    for j in range(first_job, len(jobs)):
        job, part, fit_series, val_series, test = jobs[j]
        model = SequenceModel(part.features.shape[1], "features" in args.inputs, "logits" in args.inputs, args.proj,
                              part.side.shape[1], args.hidden, dropout=args.dropout, residual=not args.plain,
                              n_logits_in=part.logits_in.shape[1], arch=args.arch,
                              two_branch=args.two_branch).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        per_epoch = math.ceil(len(fit_series) / args.batch)
        steps, warmup = args.epochs * per_epoch, max(1, int(per_epoch * args.warmup))  # 0.5: per_epoch // 2, as before
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
        generator = torch.Generator().manual_seed(args.seed + job)
        start_epoch = first_epoch if j == first_job else 1
        if start_epoch > 1:
            model.load_state_dict(state["model"]), optimizer.load_state_dict(state["optimizer"])
            schedule.load_state_dict(state["schedule"]), generator.set_state(state["generator"])
            torch.set_rng_state(state["cpu"]), torch.cuda.set_rng_state(state["cuda"])
        in_part = np.isin(series_of_row, val_series)
        best = state["best"] if start_epoch > 1 else None  # (holdout score, epoch, weights) with --stop
        if args.stop:  # hold out a share of the training patients to choose the epoch, never the validation series
            hold = patient_groups(series_patients(part, table)[fit_series], round(1 / args.stop), INNER_SEED + 1 + job) == 0
            fit_series, hold_series = fit_series[~hold], fit_series[hold]
            part_rows = pd.Series(np.arange(len(part.series)), index=part.series).loc[part.info["series"]].to_numpy()
            in_hold = np.isin(part_rows, hold_series) & part.images["keep"].to_numpy()
            y_hold = y_train.cpu().numpy()[in_hold]
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running, seen = 0.0, 0
            order = torch.from_numpy(fit_series)[torch.randperm(len(fit_series), generator=generator)]
            for index in order.split(args.batch):
                reverse = subsample = None
                if args.reverse or args.subsample:  # draws from the run's generator only when augmenting
                    u = torch.rand((len(index), 3), generator=generator)
                    reverse = u[:, 0] < args.reverse
                    long = torch.from_numpy(part.lengths[index.numpy()] > 10)
                    subsample = torch.where((u[:, 1] < args.subsample) & long, (u[:, 2] < 0.5).long(), -1)
                features, logits, side, lengths, rows, logits_in = part.batch(index.cuda(), reverse, subsample)
                y = y_train[rows.clamp(min=0)]
                loss = masked_loss(model(features, logits, side, lengths, logits_in), y, rows)
                if args.two_branch:  # deep supervision: branch 1 alone predicts the labels too
                    loss = loss + masked_loss(model.aux, y, rows)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step(), schedule.step()
                n = int((rows >= 0).sum())
                running, seen = running + loss.item() * n, seen + n
            row = {"model": job, "epoch": epoch, "train loss": running / seen}
            if args.stop:
                row["hold"] = score(y_hold, predict(model, part, 256, hold_series)[in_hold])
                if best is None or row["hold"] < best[0]:
                    best = (row["hold"], epoch, copy.deepcopy(model.state_dict()))
            p = predict(model, val, 256, val_series)
            m = in_part & kept
            row.update({"kept": score(y_kept[m], p[m]), "all": score(y_own[in_part], p[in_part])})
            if epoch == args.epochs:  # every model predicts with its last epoch, or with --stop its best on the holdout
                if args.stop:
                    model.load_state_dict(best[2])
                    p = predict(model, val, 256, val_series)
                    row["best epoch"] = best[1]
                probs[in_part] = p[in_part]
                if args.reverse_tta:
                    probs_rev[in_part] = predict(model, val, 256, val_series, reverse=True)[in_part]
                weights.append(model.state_dict())
                for key, part_test in test.items():  # test_sum: output name -> sum over the models so far
                    t = predict(model, part_test, 256)
                    test_sum[key] = test_sum.get(key, 0) + t
                    if args.reverse_tta:
                        rev = key.replace("_probs", "_rev_probs")
                        test_sum[rev] = test_sum.get(rev, 0) + predict(model, part_test, 256, reverse=True)
            log.append(row)
            pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                     "generator": generator.get_state(), "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                     "log": log, "job": j, "epoch": epoch, "probs": probs, "probs_rev": probs_rev, "test_sum": test_sum,
                     "weights": weights,
                     "best": best}
            torch.save(state, str(checkpoint) + ".tmp")
            os.replace(str(checkpoint) + ".tmp", checkpoint)
            print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)

    print(f"all models: kept {score(y_kept[done & kept], probs[done & kept]):.4f} | "
          f"all {score(y_own[done], probs[done]):.4f}", flush=True)
    val.images.assign(predicted=done).to_csv(run_dir / "val.csv", index=False)
    np.save(run_dir / "val_probs.npy", probs)
    if args.reverse_tta:
        np.save(run_dir / "val_rev_probs.npy", probs_rev)
    for key, total in test_sum.items():
        np.save(run_dir / f"{key}.npy", total / len(jobs))
    torch.save(weights, paths["models_dir"] / f"seq_{args.name}.pt")
    checkpoint.unlink()


if __name__ == "__main__":
    # On Windows the process ends with 0xC0000409 after main() has returned and every output is written (also with
    # os._exit or TerminateProcess). Scripts that chain runs check for models/seq_NAME.pt, not the exit code.
    main()
