"""Choose one temperature for a trained sequence model on its own training slices (05_sequence_models).

Usage: python scripts/fit_temperature.py --name PREFIX [--seeds 3]

The held-out fold rule allows nothing to be chosen on the validation fold, so the temperature comes from the slices
the model trained on: the runs PREFIX_s0, _s1, ... (`--source insample`, outputs/seq_runs/ and models/seq_*.pt) predict
the training part of their slice model again, their probabilities are averaged, and T minimises the competition
loss of sigmoid(logit(p) / T) on the kept slices (bounded search in [0.3, 3]). No training.

Writes outputs/seq_runs/PREFIX_temperature.json: T and the loss on the training slices before and after.
"""
import argparse
import json

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from ich.data.prepare import LABELS, read_table
from ich.metrics import score
from ich.models.sequence_model import SequenceModel
from ich.paths import load_paths
from train_sequence import Part, predict

EPS = 1e-6


def sharpen(p, t):
    p = np.clip(p, EPS, 1 - EPS)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)) / t))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="the runs PREFIX_s0, PREFIX_s1, ...")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    paths = load_paths()
    table = read_table(paths["processed_data_dir"] / "slices.csv").set_index("image")

    configs = [json.loads((paths["outputs_dir"] / "seq_runs" / f"{args.name}_s{s}" / "config.json").read_text())
               for s in range(args.seeds)]
    c = configs[0]
    assert all(k["cnn"] == c["cnn"] for k in configs) and c["source"] == "insample", "seeds of one in-sample setup"
    part = Part([paths["outputs_dir"] / "features" / run for run in c["cnn"]], "train", table, c["gaps"],
                layout="columns")
    ps = []
    for s, k in enumerate(configs):
        model = SequenceModel(part.features.shape[1], "features" in k["inputs"], "logits" in k["inputs"], k["proj"],
                              part.side.shape[1], k["hidden"], dropout=k["dropout"], residual=not k["plain"],
                              n_logits_in=part.logits_in.shape[1], arch=k.get("arch", "gru"),
                              two_branch=k.get("two_branch", False)).cuda()
        model.load_state_dict(torch.load(paths["models_dir"] / f"seq_{args.name}_s{s}.pt", weights_only=False)[0])
        ps.append(predict(model, part, 256))
    keep = part.images["keep"].to_numpy()
    y = table.loc[part.images["image"][keep], LABELS].astype(float).to_numpy()
    p = np.mean(ps, 0)[keep]
    t = minimize_scalar(lambda t: score(y, sharpen(p, t)), bounds=(0.3, 3.0), method="bounded").x
    result = {"T": float(t), "training loss": score(y, p), "training loss at T": score(y, sharpen(p, t)),
              "kept training slices": int(keep.sum())}
    (paths["outputs_dir"] / "seq_runs" / f"{args.name}_temperature.json").write_text(json.dumps(result, indent=2))
    print(f"{args.name}: T {t:.3f} | training loss {result['training loss']:.4f} -> {result['training loss at T']:.4f}")


if __name__ == "__main__":
    main()
