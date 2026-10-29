"""Write the final submission from the final runs alone (05_sequence_models, section 7): for each of the 10 slice
models of 04 (2 backbones x 5 folds), the mean of its 3 BiGRUs on logits with the temperature from its training
folds; the submission is the mean of the 10.

Usage: python scripts/make_submission.py [--name 05_seq10_bigru_T]

Needs, for every fold k and backbone b (cx, mv), the runs of the README: outputs/seq_runs/is_b_logits_fk_s0, _s1, _s2
(test.csv, test_probs.npy) and outputs/seq_runs/is_b_logits_fk_temperature.json. Writes
outputs/submissions/NAME.csv.gz, the same file notebook 05 writes.
"""
import argparse
import json

import numpy as np
import pandas as pd

from fit_temperature import sharpen
from ich.paths import load_paths
from ich.submission import check_submission, to_submission

BACKBONES, FOLDS, SEEDS = ["cx", "mv"], range(5), range(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="05_seq10_bigru_T")
    args = parser.parse_args()
    paths = load_paths()
    seq = paths["outputs_dir"] / "seq_runs"

    images, models = None, []
    for k in FOLDS:
        for b in BACKBONES:
            runs = [seq / f"is_{b}_logits_f{k}_s{s}" for s in SEEDS]
            for run in runs:
                order = pd.read_csv(run / "test.csv")["image"]
                images = order if images is None else images
                assert order.equals(images), f"{run}: other test order"
            t = json.loads((seq / f"is_{b}_logits_f{k}_temperature.json").read_text())["T"]
            models.append(sharpen(np.mean([np.load(run / "test_probs.npy") for run in runs], 0), t))
    final = np.mean(models, 0)

    sample = pd.read_csv(paths["raw_data_dir"] / "stage_2_sample_submission.csv")
    submission = check_submission(to_submission(images, final), sample)
    out = paths["outputs_dir"] / "submissions"
    out.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out / f"{args.name}.csv.gz", index=False, float_format="%.6f")
    print(f"{args.name}.csv.gz: {len(submission):,} rows from {len(models)} models | mean predicted any {final[:, -1].mean():.2%}")


if __name__ == "__main__":
    main()
