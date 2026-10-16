"""Benchmark of the storage formats compared in 02_data_preparation (section 6).

Writes the same slices (400 per writer) in every format of `ich.data.formats` to a temporary folder in
processed_data_dir, then measures what 02 shows:
- png_levels.csv, sizes.csv: the PNG compression level, the size per slice and of the whole store, the writing time;
- identical.csv, jpeg_errors.csv: what each format gives back;
- speed.csv, loader.csv: the reading time in one process and through a DataLoader with 4, 8 and 12 workers;
- per_series.csv: one file per series against one file per slice;
- gpu.csv, end_to_end.csv: the training speed of six backbones, and whether the PNG 16-bit HU loader keeps up;
- notes.json: the single numbers printed along the way.
The tables go to outputs/storage_benchmark/. It needs slices.csv from prepare_data.py, a CUDA GPU and about 5 GB free,
and takes about an hour on the machine of 02. --quick shortens the GPU timings for a test run.

    python scripts/benchmark_storage.py
"""
import argparse
import gc
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Sampler

from ich.data.dicom import HU_MAX, HU_MIN, IMAGE_SIZE, read_hu
from ich.data.formats import (FORMATS, DecodeDataset, hu_int16, npy_decode, npy_encode, png_encode, read_series_slices,
                              read_uncached)
from ich.data.prepare import SEED, read_table
from ich.data.windows import WINDOWS, window
from ich.paths import load_paths

BACKBONES = ["efficientnet_b0", "efficientnet_b3", "densenet121", "resnext50_32x4d", "seresnext50_32x4d", "resnext101_32x8d"]
BATCH = {256: 128, 384: 64, 512: 32}  # larger batches at 512 do not fit in 32 GB for some backbones
END_TO_END = {256: ["efficientnet_b0", "resnext50_32x4d"], 384: ["efficientnet_b0", "resnext50_32x4d"],
              512: ["efficientnet_b0", "resnext50_32x4d", "seresnext50_32x4d"]}


class Indices(Sampler):
    """Yields whatever `indices` holds when an epoch starts, so that one loader (one set of workers) serves every format."""

    def __init__(self):
        self.indices = []

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def write_file(path, blob):
    with open(path, "wb") as out:
        out.write(blob)
        out.flush()
        os.fsync(out.fileno())  # on the disk, so that the cold reads find it there


def ms_per_item(items, fn):
    t = time.perf_counter()
    for item in items:
        fn(item)
    return (time.perf_counter() - t) / len(items) * 1000


def png_levels(hus):
    rows = []
    for name in ["png16", "png16x3", "png8", "png8x3"]:
        f = FORMATS[name]
        arrays = [f.to_array(hu) for hu in hus[::6]]
        for level in [1, 6, 9]:
            t = time.perf_counter()
            blobs = [png_encode(a, level) for a in arrays]
            encode = (time.perf_counter() - t) / len(arrays) * 1000
            t = time.perf_counter()
            for b in blobs:
                f.decode(b)
            rows.append({"format": f"{f.label}, {f.content}", "level": level, "KB per slice": np.mean([len(b) for b in blobs]) / 1024,
                         "encode ms": encode, "decode ms": (time.perf_counter() - t) / len(arrays) * 1000})
    return pd.DataFrame(rows).round(1)


def write_formats(bench, hus, read_ms, folder, stored, paths):
    files, write_ms = {"dicom": list(bench["path"])}, {}
    for name, f in FORMATS.items():
        if f.encode is None:
            continue
        t = time.perf_counter()
        blobs = [f.encode(f.to_array(hu)) for hu in hus]
        write_ms[name] = read_ms + (time.perf_counter() - t) / len(bench) * 1000
        (folder / name).mkdir(parents=True)
        files[name] = [folder / name / f"{image}{f.suffix}" for image in bench["image"]]
        for path, blob in zip(files[name], blobs):
            write_file(path, blob)

    # sizes per writer, weighted by the writers' share of the store
    kb = pd.DataFrame({name: [p.stat().st_size / 1024 for p in ps] for name, ps in files.items()}).groupby(bench["writer"]).mean().T
    share = stored["writer"].value_counts(normalize=True)[kb.columns]
    sizes = pd.DataFrame({"format": [FORMATS[n].label for n in files], "content": [FORMATS[n].content for n in files]}, index=list(files))
    sizes = sizes.join(kb.round(0).astype(int).add_prefix("KB "))
    sizes["KB per slice"] = (kb @ share).round(1)
    sizes["store GB"] = (sizes["KB per slice"] * 1024 * len(stored) / 1e9).round(1)
    sizes["vs DICOM"] = (sizes["KB per slice"] / sizes.loc["dicom", "KB per slice"]).round(3)
    sizes["write ms"] = pd.Series(write_ms).round(1)
    notes = {f"free GB on {drive}": int(round(shutil.disk_usage(p).free / 1e9)) for drive, p in
             [("E:", paths["processed_data_dir"]), ("F:", paths["raw_data_dir"])]}
    return files, sizes, notes


def what_is_kept(bench, hus, files):
    rows = []
    for name, f in FORMATS.items():
        if f.encode is None:
            continue
        same = sum(np.array_equal(f.decode(p.read_bytes()), f.to_array(hu)) for p, hu in zip(files[name], hus))
        rows.append({"format": f.label, "content": f.content, "files identical to the stored array": f"{same:,} / {len(hus):,}"})
    diff = np.array([np.abs(FORMATS["npy16"].decode(p.read_bytes()) - hu).max() for p, hu in zip(files["npy16"], hus)])
    resized = ((bench["rows"] != IMAGE_SIZE) | (bench["cols"] != IMAGE_SIZE)).to_numpy()
    notes = {"int16 vs read_hu, largest difference HU": float(diff.max()), "int16 vs read_hu, slices with a difference": int((diff > 0).sum()),
             "int16 vs read_hu, all of them resized": bool(resized[diff > 0].all()), "resized slices in the sample": int(resized.sum())}
    return pd.DataFrame(rows), notes


def jpeg_errors(hus, files):
    """Error of the JPEG files where the stored value lies inside the window (not 0 or 255), also for the OpenCV default 4:2:0."""
    hu_per_level = {"8-bit HU": 1.0, **{w: width / 255 for w, (_, width) in WINDOWS.items()}}
    rows, kb_420 = [], []
    for label, name, params in [("JPEG", "jpg8", None), ("JPEG 4:4:4", "jpg8x3", None),
                                ("JPEG 4:2:0 (OpenCV default)", "jpg8x3", [cv2.IMWRITE_JPEG_QUALITY, 95])]:
        f = FORMATS[name]
        channels = ["8-bit HU"] if name == "jpg8" else list(WINDOWS)
        hist = np.zeros((len(channels), 256), dtype=np.int64)  # pixels by error in levels
        for hu, path in zip(hus, files[name]):
            a = f.to_array(hu)
            if params is None:
                blob = path.read_bytes()
            else:
                blob = cv2.imencode(".jpg", a, params)[1].tobytes()
                kb_420.append(len(blob) / 1024)
            out = f.decode(blob).reshape(a.shape[0], a.shape[1], -1)
            a = a.reshape(out.shape)
            for k in range(len(channels)):
                inside = (a[..., k] > 0) & (a[..., k] < 255)
                hist[k] += np.bincount(np.abs(out[..., k][inside].astype(np.int16) - a[..., k][inside]), minlength=256)
        for k, channel in enumerate(channels):
            h, levels = hist[k], np.arange(256)
            mean = (h * levels).sum() / h.sum()
            rows.append({"format": label, "channel": channel, "% pixels changed": (1 - h[0] / h.sum()) * 100, "mean error (levels)": mean,
                         "99.9th percentile": levels[np.searchsorted(np.cumsum(h), 0.999 * h.sum())], "max": levels[h > 0].max(),
                         "mean error (HU)": mean * hu_per_level[channel]})
    notes = {"JPEG 3 windows KB 4:4:4": int(round(np.mean([p.stat().st_size for p in files["jpg8x3"]]) / 1024)),
             "JPEG 3 windows KB 4:2:0": int(round(np.mean(kb_420)))}
    return pd.DataFrame(rows).round(2), notes


def reading_speed(files, sizes):
    """Warm (file cache), cold (past the cache) and disk (16 reads at once, no decoding); median of three passes."""
    warm, cold, disk = ({name: [] for name in files} for _ in range(3))
    for name, ps in files.items():
        ms_per_item(ps, lambda p: FORMATS[name].decode(p.read_bytes()))  # into the file cache
    for _ in range(3):
        for name, ps in files.items():
            decode = FORMATS[name].decode
            warm[name].append(ms_per_item(ps, lambda p: decode(p.read_bytes())))
            cold[name].append(ms_per_item(ps, lambda p: decode(read_uncached(p))))
            with ThreadPoolExecutor(16) as pool:
                t = time.perf_counter()
                list(pool.map(read_uncached, ps))
                disk[name].append(len(ps) / (time.perf_counter() - t))
    speed = sizes[["format", "content", "KB per slice"]].copy()
    speed["warm ms"] = pd.Series({n: np.median(v) for n, v in warm.items()}).round(2)
    speed["cold ms"] = pd.Series({n: np.median(v) for n, v in cold.items()}).round(2)
    speed["slices/s warm"] = (1000 / speed["warm ms"]).round(0).astype(int)
    speed["slices/s cold"] = (1000 / speed["cold ms"]).round(0).astype(int)
    speed["disk slices/s (16 at once)"] = pd.Series({n: np.median(v) for n, v in disk.items()}).round(0).astype(int)
    speed["disk MB/s"] = (speed["disk slices/s (16 at once)"] * speed["KB per slice"] * 1024 / 1e6).round(0).astype(int)
    return speed


def loader_speed(files, n_bench, speed, sizes, n_train):
    """Slices per second through a DataLoader (files in the cache, each 4 times, shuffled), median of two runs."""
    names = list(files)
    items = [(name, str(p)) for name in names for p in files[name]]
    rng = np.random.default_rng(SEED)
    result = pd.DataFrame({"slices/s, 1 process": speed["slices/s warm"]})
    for workers in [4, 8, 12]:
        sampler = Indices()
        loader = DataLoader(DecodeDataset(items), batch_size=32, sampler=sampler, num_workers=workers, persistent_workers=True, timeout=120)
        sampler.indices = list(range(min(32 * workers, n_bench)))  # files of the first format only, so that every batch has one shape
        for _ in loader:  # starts the workers
            pass
        for k, name in enumerate(names):
            epoch = np.tile(np.arange(k * n_bench, (k + 1) * n_bench), 4)
            runs = []
            for _ in range(2):
                sampler.indices = rng.permutation(epoch).tolist()
                t = time.perf_counter()
                for _ in loader:
                    pass
                runs.append(time.perf_counter() - t)
            result.loc[name, f"slices/s, {workers} workers"] = len(epoch) / np.median(runs)
        del loader
    loaded = sizes[["format", "content", "store GB"]].join(result.round(0).astype(int))
    loaded["epoch min, in memory"] = (n_train / result["slices/s, 12 workers"] / 60).round(1)
    loaded["epoch min, disk limit"] = (n_train / speed["disk slices/s (16 at once)"] / 60).round(1)
    return loaded


def cpu_windows(files):
    lut = np.stack([window(np.arange(HU_MIN, HU_MAX + 1), level, width) for level, width in WINDOWS.values()], axis=-1)
    hu16 = [FORMATS["npy16"].decode(p.read_bytes()) for p in files["npy16"][::6]]
    win8 = [FORMATS["png8x3"].decode(p.read_bytes()) for p in files["png8x3"][::6]]
    return {"CPU windows ms per slice, from int16 HU": round(float(ms_per_item(hu16, lambda h: lut[h - HU_MIN])), 2),
            "CPU windows ms per slice, from three 8-bit windows": round(float(ms_per_item(win8, lambda a: a.astype(np.float32) / 255)), 2)}


def per_series_files(table, stored, folder):
    """Whole scans and three neighbouring slices: one npy file per series against one per slice."""
    chosen = stored.drop_duplicates("series").groupby("writer").sample(10, random_state=SEED)["series"]
    scans = table[table["series"].isin(chosen) & table["keep"]].sort_values(["series", "slice"])
    (folder / "series").mkdir()
    (folder / "slices").mkdir()
    series_file, slice_files = {}, {}
    for series, g in scans.groupby("series"):
        volume = np.stack([hu_int16(read_hu(p)) for p in g["path"]])
        series_file[series] = folder / "series" / f"{series}.npy"
        write_file(series_file[series], npy_encode(volume))
        slice_files[series] = [folder / "slices" / f"{image}.npy" for image in g["image"]]
        for path, s in zip(slice_files[series], volume):
            write_file(path, npy_encode(s))

    centre = scans[(scans["slice"] > 0) & (scans["slice"] < scans["n_slices"] - 1)].sample(300, random_state=SEED)
    triplets = list(zip(centre["series"], centre["slice"].astype(int)))
    reads = {
        ("whole scan, ms per slice", "one file per series"):
            (lambda s: npy_decode(series_file[s].read_bytes()), lambda s: npy_decode(read_uncached(series_file[s]))),
        ("whole scan, ms per slice", "one file per slice"):
            (lambda s: [npy_decode(p.read_bytes()) for p in slice_files[s]], lambda s: [npy_decode(read_uncached(p)) for p in slice_files[s]]),
        ("3 neighbouring slices, ms per read", "one file per series, memory map"):
            (lambda t: np.array(np.load(series_file[t[0]], mmap_mode="r")[t[1] - 1:t[1] + 2]), None),
        ("3 neighbouring slices, ms per read", "one file per series, read of the 3 slices"):
            (lambda t: read_series_slices(series_file[t[0]], t[1] - 1, 3), None),
        ("3 neighbouring slices, ms per read", "one file per slice"):
            (lambda t: np.stack([npy_decode(slice_files[t[0]][i].read_bytes()) for i in (t[1] - 1, t[1], t[1] + 1)]), None),
    }
    timings = {key: [] for key in reads}
    for _ in range(3):
        for key, (warm_read, cold_read) in reads.items():
            if key[0].startswith("whole"):
                per_slice = len(series_file) / len(scans)
                timings[key].append((ms_per_item(list(series_file), warm_read) * per_slice, ms_per_item(list(series_file), cold_read) * per_slice))
            else:
                timings[key].append((ms_per_item(triplets, warm_read), np.nan))
    result = pd.DataFrame({key: np.median(v, axis=0) for key, v in timings.items()}, index=["warm", "cold"]).T.round(2)
    result.index.names = ["read", "store"]
    notes = {"per-series test: series": len(series_file), "per-series test: slices": len(scans)}
    gc.collect()  # closes the memory maps, so that Windows lets the files go
    return result.reset_index(), notes


def new_model(name):
    model = timm.create_model(name, pretrained=False, num_classes=6).cuda().to(memory_format=torch.channels_last)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-4)


def train_step(model, opt, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def per_second(run, batch, seconds):
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    n, t = 0, time.perf_counter()
    while time.perf_counter() - t < seconds:
        run()
        n += 1
    torch.cuda.synchronize()
    return n * batch / (time.perf_counter() - t)


def gpu_speed(seconds):
    """Training and inference speed on synthetic batches already on the GPU: bf16 autocast, channels-last, AdamW."""
    rows = []
    for name in BACKBONES:
        for size, batch in BATCH.items():
            model, opt = new_model(name)
            x = torch.randn(batch, 3, size, size, device="cuda").to(memory_format=torch.channels_last)
            y = torch.rand(batch, 6, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            train = per_second(lambda: train_step(model, opt, x, y), batch, seconds)
            peak = torch.cuda.max_memory_reserved() / 1e9
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                infer = per_second(lambda: model(x), batch, seconds)
            rows.append({"backbone": name, "size": size, "train": train, "inference": infer, "peak GB": peak})
            del model, opt, x, y
            torch.cuda.empty_cache()
    result = pd.DataFrame(rows)
    notes = {"GPU memory GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "largest peak GB": round(float(result["peak GB"].max()), 1)}
    return result[["backbone", "size", "train", "inference"]].round(0), notes


def end_to_end(files, seconds, batches, skip):
    """PNG 16-bit HU through 12 workers, windows and resizing on the GPU, against the same step on one batch kept on the GPU."""
    lut = torch.from_numpy(np.stack([window(np.arange(HU_MIN, HU_MAX + 1), level, width) for level, width in WINDOWS.values()])).cuda()
    ds = DecodeDataset([("png16", str(p)) for p in files["png16"]])

    def windows_step(model, opt, stored_hu, y, size):
        x = lut[:, stored_hu.long()].permute(1, 0, 2, 3)  # HU + 1024 -> 3 windows
        if size != IMAGE_SIZE:
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True)
        train_step(model, opt, x.contiguous(memory_format=torch.channels_last), y)

    def loader_per_second(loader, step):
        # the first `skip` batches are not timed: the workers still fill their queues
        for i, stored_hu in enumerate(loader):
            if i == skip:
                torch.cuda.synchronize()
                t, n = time.perf_counter(), 0
            step(stored_hu)
            if i >= skip:
                n += len(stored_hu)
        torch.cuda.synchronize()
        return n / (time.perf_counter() - t)

    rows = {}
    for size, names in END_TO_END.items():
        batch = BATCH[size]
        resident = torch.stack([ds[i % len(ds)] for i in range(batch)]).cuda()  # one batch kept on the GPU
        y = torch.rand(batch, 6, device="cuda")
        for pinned in (False, True):
            loader = DataLoader(ds, batch_size=batch, sampler=RandomSampler(ds, replacement=True, num_samples=batch * batches),
                                num_workers=12, pin_memory=pinned, persistent_workers=True, timeout=120)
            for name in names:
                model, opt = new_model(name)
                row = rows.setdefault((name, size), {"backbone": name, "size": size})
                if not pinned:
                    row["data on the GPU"] = per_second(lambda: windows_step(model, opt, resident, y, size), batch, seconds)
                row["loader, pinned + non_blocking" if pinned else "loader, plain .cuda()"] = loader_per_second(
                    loader, lambda stored_hu: windows_step(model, opt, stored_hu.cuda(non_blocking=pinned), y, size))
                del model, opt
                torch.cuda.empty_cache()
            del loader
    result = pd.DataFrame(list(rows.values()))
    for column in ["loader, plain .cuda()", "loader, pinned + non_blocking"]:
        result[f"% ({column.split(', ')[1]})"] = result[column] / result["data on the GPU"] * 100
    return result.round(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-writer", type=int, default=400, help="benchmark slices per writer")
    parser.add_argument("--out", type=Path, help="output folder (default: outputs_dir/storage_benchmark)")
    parser.add_argument("--quick", action="store_true", help="short GPU timings, for a test run")
    args = parser.parse_args()

    paths = load_paths()
    out = args.out or paths["outputs_dir"] / "storage_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    folder = paths["processed_data_dir"] / "format_benchmark"
    shutil.rmtree(folder, ignore_errors=True)
    torch.backends.cudnn.benchmark = True
    seconds, batches, skip = (0.5, 40, 10) if args.quick else (4, 160, 20)

    table = read_table(paths["processed_data_dir"] / "slices.csv")
    table["path"] = [paths["raw_data_dir"] / f"stage_2_{s}" / f"{i}.dcm" for s, i in zip(table["split"], table["image"])]
    stored = table[(table["split"] == "test") | table["keep"]]  # what a store holds: the kept train slices and all test slices
    bench = stored.groupby("writer").sample(args.per_writer, random_state=SEED).reset_index(drop=True)
    notes = {"run": time.strftime("%Y-%m-%d"), "store slices": len(stored), "benchmark slices": len(bench)}

    def save(name, df):
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv written ({time.strftime('%H:%M:%S')})", flush=True)

    t = time.perf_counter()
    hus = [read_hu(p) for p in bench["path"]]
    read_ms = (time.perf_counter() - t) / len(bench) * 1000
    save("png_levels", png_levels(hus))
    files, sizes, more = write_formats(bench, hus, read_ms, folder, stored, paths)
    notes.update(more)
    save("sizes", sizes.rename_axis("name").reset_index())
    identical, more = what_is_kept(bench, hus, files)
    notes.update(more)
    save("identical", identical)
    errors, more = jpeg_errors(hus, files)
    notes.update(more)
    save("jpeg_errors", errors)
    del hus

    speed = reading_speed(files, sizes)
    save("speed", speed.rename_axis("name").reset_index())
    n_train = int((table["keep"] & (table["split"] == "train")).sum())
    save("loader", loader_speed(files, len(bench), speed, sizes, n_train).rename_axis("name").reset_index())
    notes.update(cpu_windows(files))
    series, more = per_series_files(table, stored, folder)
    notes.update(more)
    save("per_series", series)

    gpu, more = gpu_speed(seconds)
    notes.update(more)
    save("gpu", gpu)
    save("end_to_end", end_to_end(files, seconds, batches, skip))

    json.dump(notes, open(out / "notes.json", "w", encoding="utf-8"), indent=1)
    gc.collect()
    shutil.rmtree(folder)
    print("done; test files deleted:", not folder.exists())


if __name__ == "__main__":
    main()
