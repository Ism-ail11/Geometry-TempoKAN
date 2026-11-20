"""Evaluate common test-time gap, noise, and joint mounting/outage sweeps."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from Datasets.prepare import load_prepared
from Datasets.protocol import subject_splits
from Proposed_Method.predict import load_model
from .registry import DATA_NAMES, MODELS
from .stress import probabilities, rotated_test


SWEEPS = {
    "sample_gap": (0, 8, 16, 32),
    "gaussian_noise": (0, 5, 10, 20, 35),
    "rotation_with_outage": (0, 15, 30, 60),
}


def burst_gap(x, size, offsets):
    """Replace a common contiguous gap across channels by endpoint interpolation."""
    if size == 0:
        return x
    max_offset = x.shape[1] - size - 2
    first = (offsets * (max_offset + 1)).long().clamp(0, max_offset) + 1
    last = first + size
    rows = torch.arange(len(x))
    t = torch.arange(x.shape[1])[None]
    a = x[rows, first - 1]
    b = x[rows, last]
    fraction = ((t - first[:, None] + 1) / (size + 1)).clamp(0, 1)
    bridge = torch.lerp(a[:, None], b[:, None], fraction[..., None])
    inside = (t >= first[:, None]) & (t < last[:, None])
    return torch.where(inside[..., None], bridge, x)


def outage_mask(subject, channels, dataset):
    width = 6 if dataset == "pamap2" else 3
    people, inverse = np.unique(subject, return_inverse=True)
    group_id = [int(np.random.default_rng(42_817 + int(person)).integers(channels // width)) for person in people]
    missing = torch.as_tensor(group_id)[inverse]
    return (torch.arange(channels)[None] // width == missing[:, None])[:, None]


def perturbation(x, subject, dataset, family, level, mean, std, offsets, white_noise, missing):
    if family == "sample_gap":
        return burst_gap(x, level, offsets)
    if family == "gaussian_noise":
        return x + (level / 100) * std * white_noise
    rotated = rotated_test(x, subject, dataset, level) if level else x
    return torch.where(missing, mean, rotated)


def evaluate_dataset(results, cache, destination, dataset, seed=42):
    data = load_prepared(dataset, cache)
    all_indices, blocks = [], {}
    for split in subject_splits(dataset, data):
        idx = split.test
        x = torch.from_numpy(data.x[idx])
        person = data.subject[idx]
        paths = {model: results / "Runs" / dataset / "clean" / model / f"fold{split.fold:02d}_seed{seed}"
                 for model in MODELS}
        bundles = {model: load_model(path / "checkpoint.pt", "cpu") for model, path in paths.items()}
        norm = bundles[MODELS[0]][0]["normalization"]
        if any(not torch.equal(norm[key], bundle[0]["normalization"][key])
               for bundle in bundles.values() for key in ("mean", "std")):
            raise ValueError(f"Models use different training normalization in {dataset} fold {split.fold}")
        clean = {}
        for name, directory in paths.items():
            with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["index"], idx):
                    raise ValueError(f"Different test indices in {directory}")
                clean[name] = saved["pred"][saved["conditions"].tolist().index("clean")]
        rng = torch.Generator().manual_seed(900_103 + split.fold)
        offsets = torch.rand(len(x), generator=rng)
        noise = torch.randn(x.shape, generator=rng, dtype=x.dtype)
        missing = outage_mask(person, x.shape[-1], dataset)
        for family, levels in SWEEPS.items():
            for level in levels:
                if level == 0 and family != "rotation_with_outage":
                    predictions = clean
                else:
                    changed = perturbation(x, person, dataset, family, level, norm["mean"], norm["std"],
                                           offsets, noise, missing)
                    predictions = {name: probabilities(bundle, changed).argmax(-1).astype(np.int8)
                                   for name, bundle in bundles.items()}
                for name, indices in predictions.items():
                    blocks.setdefault(f"{family}_{level}_{name}", []).append(indices)
        all_indices.append(idx)
        print(f"{dataset} sweep fold {split.fold} complete", flush=True)
    indices = np.concatenate(all_indices)
    destination.mkdir(parents=True, exist_ok=True)
    out = destination / f"{dataset}_sweeps.npz"
    temporary = out.with_suffix(".part.npz")
    np.savez_compressed(temporary, index=indices, actual=data.y[indices], subject=data.subject[indices],
                        pace=data.pace[indices], **{key: np.concatenate(value) for key, value in blocks.items()})
    temporary.replace(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("Results"))
    p.add_argument("--data-dir", type=Path, default=Path("Datasets/Cache"))
    p.add_argument("--datasets", nargs="+", choices=DATA_NAMES, default=list(DATA_NAMES))
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()
    if args.jobs < 1:
        p.error("jobs must be positive")
    torch.set_num_threads(1)
    evaluate = partial(evaluate_dataset, args.results, args.data_dir, args.results / "Robustness" / "Predictions")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for task in [pool.submit(evaluate, name) for name in dict.fromkeys(args.datasets)]:
            task.result()


if __name__ == "__main__":
    main()
