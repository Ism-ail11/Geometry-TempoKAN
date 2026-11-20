"""Reconcile the release plan, dataset roles, saved metrics, and figure assets."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from zipfile import ZipFile

from Datasets.prepare import CLASSES
from .analyze import collect, stress_predictions
from .expanded import CONFIDENCE_LEVELS, matched_seed_ids, sweep_predictions
from .registry import DATA_NAMES, MODELS, PRIMARY, TASKS
from .stress_sweeps import SWEEPS
from .train import Run, write_json


def archived_checkpoints(project):
    entries = json.loads((project / "Experiments/Plans/Archived_Checkpoints.json").read_text(encoding="utf-8"))
    selected = set(entries)
    if len(selected) != len(entries):
        raise ValueError("Duplicate fit in the archived checkpoint selection.")
    return selected


def run_identity(config):
    return Run(**config).identifier


def verify_source(project):
    results = project / "Results"
    groups, records, _, n_checks = collect(results)
    plan = json.loads((project / "Experiments/Plans/All.json").read_text(encoding="utf-8"))
    planned = {run_identity(c): c for c in plan}
    actual = {run_identity(record["config"]): record["config"] for record in records}
    if planned != actual or len(planned) != len(plan):
        raise ValueError("The executed fit configurations differ from the release plan.")
    checkpoints = archived_checkpoints(project)
    if checkpoints - planned.keys():
        raise ValueError("Checkpoint selection includes a fit outside the release plan.")
    with (project / "Datasets/Splits.csv").open(newline="", encoding="utf-8") as stream:
        roles, counts = defaultdict(set), defaultdict(int)
        for row in csv.DictReader(stream):
            key = (row["dataset"], row["regime"], int(row["fold"]), row["role"])
            roles[key].add(int(row["participant"]))
            counts[key] += int(row["windows"])
    stored = 0
    inventory = []
    for name in sorted(actual):
        folder = results / "Runs" / name
        item = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        config = actual[name]
        for role in ("train", "val", "test"):
            key = (config["dataset"], config["regime"], config["fold"], role)
            if set(item["subjects"][role]) != roles[key] or item["counts"][role] != counts[key]:
                raise ValueError(f"Participant allocation differs from the prepared split in {name}")
        keep = name in checkpoints
        if keep and not (folder / "checkpoint.pt").is_file():
            raise ValueError(f"Missing required checkpoint in {name}")
        stored += int(keep)
        inventory.append({"fit": name, "checkpoint_included": keep, "held_out_predictions_included": True})
    seed_counts = {}
    for dataset, regime in TASKS:
        seeds = matched_seed_ids(groups, dataset, regime)
        if not {42, 77}.issubset(seeds):
            raise ValueError(f"Missing a matched training seed for {dataset}/{regime}.")
        seed_counts[dataset, regime] = len(seeds)
        if not stress_predictions(results, dataset, regime, groups[dataset, regime, PRIMARY, 42]):
            raise ValueError(f"Missing reproducible stress predictions for {dataset}/{regime}.")
    sweep_count = sum(len(sweep_predictions(results, name, groups[name, "clean", PRIMARY, 42]))
                      for name in DATA_NAMES)
    if sweep_count != len(DATA_NAMES) * len(MODELS) * sum(map(len, SWEEPS.values())):
        raise ValueError("Incomplete stress sweep.")
    calibration_count = len(MODELS) * sum(seeds * (4 if regime == "pace" else 1)
                                         for (_, regime), seeds in seed_counts.items())
    agreement_count = len(MODELS) * sum((seeds * (seeds - 1) // 2) * (4 if regime == "pace" else 1)
                                       for (_, regime), seeds in seed_counts.items())
    tables = {"Benchmark/Matched_Seeds.csv": len(TASKS) * len(MODELS),
              "Robustness/Stress_Sweeps.csv": sweep_count,
              "Robustness/Rotation_Outage_Interaction.csv":
                  len(DATA_NAMES) * len(MODELS) * (len(SWEEPS["rotation_with_outage"]) - 1),
              "Diagnostics/Stress_Per_Class.csv": len(MODELS) * len(SWEEPS) * sum(len(CLASSES[ds]) for ds in DATA_NAMES),
              "Diagnostics/Calibration.csv": calibration_count,
              "Diagnostics/Risk_Coverage.csv": calibration_count * len(CONFIDENCE_LEVELS),
              "Diagnostics/Seed_Agreement.csv": agreement_count,
              "Diagnostics/Participant_Variability.csv": len(TASKS) * len(MODELS)}
    for path, expected in tables.items():
        with (results / path).open(newline="", encoding="utf-8") as stream:
            if sum(1 for _ in csv.DictReader(stream)) != expected:
                raise ValueError(f"Unexpected row count in {path}, expected {expected}.")
    figures = results / "Figures"
    unique = sorted(path.stem for path in figures.glob("*.svg"))
    if len(unique) != 4 or any(not (figures / f"{name}.{ext}").is_file()
                               for name in unique for ext in ("svg", "png", "pdf")):
        raise ValueError("Expected four composite figures in PDF, SVG, and PNG.")
    if not (results / "Index.html").is_file():
        raise ValueError("Missing results index.")
    with (results / "Execution/Checkpoint_Inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=inventory[0])
        writer.writeheader()
        writer.writerows(inventory)
    audit = {"verified_fits": len(records), "verified_metric_rows": n_checks,
             "archived_checkpoints": stored, "composite_figures": len(unique), "verified_protocols": len(TASKS),
             "verified_sweep_rows": sweep_count, "verified_calibration_rows": calibration_count,
             "verified_agreement_rows": agreement_count,
             "matched_training_seeds": {f"{dataset}/{regime}": count
                                        for (dataset, regime), count in seed_counts.items()}}
    write_json(results / "Execution/Release_Verification.json", audit)
    print(json.dumps(audit, indent=2))


def verify_zip(path):
    import hashlib

    with ZipFile(path) as z:
        entries = z.namelist()
        prefix = "Geometry_TempoKAN/"
        manifest = prefix + "SHA256SUMS.txt"
        lines = z.read(manifest).decode().splitlines()
        if len(lines) + 1 != len(entries) or len(set(entries)) != len(entries):
            raise ValueError("ZIP entries differ from its checksum manifest.")
        for line in lines:
            digest, name = line.split("  ", 1)
            if hashlib.sha256(z.read(prefix + name)).hexdigest() != digest:
                raise ValueError(f"Wrong checksum for {name}")
    print(f"Verified {len(lines)} SHA-256 checksums")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    if args.zip:
        verify_zip(args.zip)
    else:
        verify_source(args.project)


if __name__ == "__main__":
    main()
