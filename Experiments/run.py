"""Train a saved experiment plan with resumable participant-disjoint fits."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path

from Datasets.prepare import prepare_dataset
from .registry import BASELINE_WIDTHS, FOLDS, GEOMETRY, MODELS, TASKS
from .train import Run, execute, write_json


def make_run(dataset, regime, model, fold, seed=42, epochs=14, cap=0):
    return Run(dataset, regime, model, fold, seed, epochs, batch=256, width=BASELINE_WIDTHS.get(model, 24),
               augmentation="gait_shift" if regime == "pace" else "motion_sensor", patience=6,
               validation_warp_weight=0.20 if regime == "pace" else 0.10,
               profile="geometry" if model in GEOMETRY else "comparison", cap=cap)


def validate_plan(runs):
    if not runs:
        raise ValueError("The plan must include at least one fit.")
    if len({run.identifier for run in runs}) != len(runs):
        raise ValueError("A fit appears twice in the plan.")
    for run in runs:
        if (run.dataset, run.regime) not in TASKS or run.model not in MODELS:
            raise ValueError(f"Unknown task or model in {run.identifier}")
        if not 0 <= run.fold < FOLDS[run.dataset] or min(run.epochs, run.width, run.stride, run.batch) < 1:
            raise ValueError(f"Invalid fold or training budget in {run.identifier}")
        if run.augmentation != ("gait_shift" if run.regime == "pace" else "motion_sensor"):
            raise ValueError(f"Unexpected augmentation in {run.identifier}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--data-dir", default=Path("Datasets/Cache"), type=Path)
    parser.add_argument("--output", default=Path("Results/Reproduction"), type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--jobs", default=1, type=int)
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--log-name", default="training.log")
    args = parser.parse_args()
    if min(args.threads, args.jobs) < 1 or Path(args.log_name).name != args.log_name:
        parser.error("Use positive job and thread counts and a plain log filename.")
    if args.device.startswith("cuda") and args.jobs != 1:
        parser.error("Use one worker per CUDA device.")
    try:
        runs = [Run(**item) for item in json.loads(args.plan.read_text(encoding="utf-8"))]
        validate_plan(runs)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if not args.no_prepare:
        for dataset in sorted({run.dataset for run in runs}):
            prepare_dataset(dataset, args.data_dir)

    folder = args.output / "Execution"
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(folder / f"plan_{timestamp}.json", [asdict(run) for run in runs])

    def log(message):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {message}"
        with (folder / args.log_name).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    log(f"Running {len(runs)} fits with {args.jobs} worker(s) on {args.device}")
    complete = 0
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        tasks = {pool.submit(execute, run, args.data_dir, args.output, args.device, args.threads): run for run in runs}
        for job in as_completed(tasks):
            try:
                log(job.result())
                complete += 1
            except Exception as error:
                log(f"FAILED {tasks[job].identifier} {type(error).__name__}: {error}")
    log(f"Finished {complete}/{len(runs)} fits")
    if complete != len(runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
