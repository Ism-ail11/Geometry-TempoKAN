"""Export exact participant assignments and window counts from the prepared releases."""

import argparse
import csv
from pathlib import Path

import numpy as np

from Experiments.registry import TASKS
from .prepare import load_prepared
from .protocol import restrict_pace, subject_splits


def produce(data_dir, destination):
    fields = ("dataset", "regime", "fold", "role", "participant", "windows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset, regime in TASKS:
            data = load_prepared(dataset, data_dir)
            for split in subject_splits(dataset, data):
                split = restrict_pace(split, data, regime)
                for role in ("train", "val", "test"):
                    people, counts = np.unique(data.subject[getattr(split, role)], return_counts=True)
                    for person, count in zip(people, counts):
                        writer.writerow(dict(dataset=dataset, regime=regime, fold=split.fold, role=role,
                                             participant=int(person), windows=int(count)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("Datasets/Cache"))
    p.add_argument("--output", type=Path, default=Path("Datasets/Splits.csv"))
    args = p.parse_args()
    produce(args.data_dir, args.output)


if __name__ == "__main__":
    main()
