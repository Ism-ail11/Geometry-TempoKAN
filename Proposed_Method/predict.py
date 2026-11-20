"""Classify an N × 128 × C NumPy array using a supplied trained checkpoint."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from Experiments.registry import make_model


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    labels = checkpoint["labels"]
    model = make_model(config["model"], checkpoint["input_channels"], len(labels), config["width"], config["stride"])
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint, model.eval()


@torch.inference_mode()
def predict_checkpoint(checkpoint_path, windows, device="cpu", batch=128):
    checkpoint, model = load_model(checkpoint_path, device)
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim == 2:
        windows = windows[None, :, :]
    if windows.ndim != 3 or len(windows) == 0 or windows.shape[1:] != (128, checkpoint["input_channels"]):
        raise ValueError("The array must have 128 samples and the checkpoint's channel count")
    if not np.isfinite(windows).all():
        raise ValueError("Input contains a nonfinite value")
    standard = checkpoint["normalization"]
    x = torch.as_tensor(windows, device=device)
    x = (x - standard["mean"]) / standard["std"]
    return checkpoint["labels"], torch.cat([model(block).softmax(-1).cpu() for block in x.split(batch)]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--average-with", type=Path, help="Average probabilities with a second checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Finite unstandardized .npy windows")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    windows = np.load(args.input, allow_pickle=False)
    labels, probabilities = predict_checkpoint(args.checkpoint, windows, args.device)
    if args.average_with:
        other_labels, other = predict_checkpoint(args.average_with, windows, args.device)
        if other_labels != labels:
            parser.error("both checkpoints must have the same class labels and order")
        probabilities = 0.5 * (probabilities + other)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window", "prediction", "confidence", *labels])
        for i, row in enumerate(probabilities):
            top = int(row.argmax())
            writer.writerow([i, labels[top], f"{row[top]:.6f}", *(f"{p:.6f}" for p in row)])
    print(f"{len(probabilities):,} windows scored to {args.output}")


if __name__ == "__main__":
    main()
