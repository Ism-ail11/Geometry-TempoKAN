"""Evaluate stored checkpoints under common per-placement rotations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from Datasets.prepare import load_prepared
from Datasets.protocol import restrict_pace, subject_splits
from Proposed_Method.model import rotate_measurements
from Proposed_Method.predict import load_model
from .registry import MODELS, TASKS


ANGLES = (0, 15, 30, 60)
AVERAGE = "geometry_average"


def rotated_test(x, subjects, dataset, angle):
    """Give each participant a fixed random axis at every sensor placement."""
    group = 3 if dataset == "selfback" else 6
    people, inverse = np.unique(subjects, return_inverse=True)
    axes = torch.stack([torch.randn((x.shape[-1] // group, 3),
                                   generator=torch.Generator().manual_seed(810_443 + int(p)))
                        for p in people])
    radians = torch.full((len(x), x.shape[-1] // group), float(np.deg2rad(angle)), dtype=x.dtype)
    return rotate_measurements(x, axes[inverse], radians, group)


@torch.inference_mode()
def probabilities(bundle, raw):
    checkpoint, model = bundle
    stats = checkpoint["normalization"]
    values = (raw - stats["mean"]) / stats["std"]
    batch = checkpoint["config"]["batch"]
    return torch.cat([model(x).softmax(-1) for x in values.split(batch)]).numpy()


def evaluate_task(root, data_dir, destination, dataset, regime):
    data = load_prepared(dataset, data_dir)
    index, prediction_blocks = [], {}
    angles = ANGLES if regime == "clean" else (0,)
    for split in subject_splits(dataset, data):
        split = restrict_pace(split, data, regime)
        index.append(split.test)
        windows = torch.from_numpy(data.x[split.test])
        participants = data.subject[split.test]
        checkpoints = root / "Runs" / dataset / regime
        bundles = {model: load_model(checkpoints / model / f"fold{split.fold:02d}_seed42" / "checkpoint.pt", "cpu")
                   for model in MODELS}
        for angle in angles:
            x = windows if angle == 0 else rotated_test(windows, participants, dataset, angle)
            posterior = {model: probabilities(bundle, x) for model, bundle in bundles.items()}
            posterior[AVERAGE] = (posterior["geometry_kan"] + posterior["geometry_mlp"]) / 2
            for model, values in posterior.items():
                labels = values.argmax(-1).astype(np.int8)
                if angle == 0 and model in MODELS:
                    saved_path = checkpoints / model / f"fold{split.fold:02d}_seed42" / "predictions.npz"
                    with np.load(saved_path, allow_pickle=False) as archived:
                        if (not np.array_equal(archived["index"], split.test)
                                or not np.array_equal(archived["pred"][0], labels)):
                            raise ValueError(f"Checkpoint replay disagrees with saved labels in {saved_path}.")
                key = f"{angle}_{model}"
                prediction_blocks.setdefault(key, []).append(labels)
        print(f"{dataset}/{regime} fold {split.fold} checked", flush=True)
    index = np.concatenate(index)
    predictions = {key: np.concatenate(value) for key, value in prediction_blocks.items()}
    destination.mkdir(parents=True, exist_ok=True)
    out = destination / f"{dataset}_{regime}.npz"
    tmp = out.with_suffix(".part.npz")
    np.savez_compressed(tmp, index=index, subject=data.subject[index], actual=data.y[index],
                        pace=data.pace[index], **predictions)
    tmp.replace(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("Results"))
    parser.add_argument("--data-dir", type=Path, default=Path("Datasets/Cache"))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    torch.set_num_threads(1)
    evaluate = partial(evaluate_task, args.results, args.data_dir, args.results / "Robustness" / "Predictions")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for task in [pool.submit(evaluate, *setting) for setting in TASKS]:
            task.result()


if __name__ == "__main__":
    main()
