# Geometry TempoKAN

This release contains the current wearable human activity recognition model, four controlled ablations, and four independently trained comparators. Open [Results/Index.html](Results/Index.html) after extracting the ZIP.

| Folder | Contents |
|---|---|
| [Proposed_Method](Proposed_Method/) | Geometry TempoKAN, spline and harmonic layers, inference, and the optional CPU kernel |
| [Datasets](Datasets/) | Dataset download and preparation scripts, release checksums, class and channel definitions, and participant split assignments |
| [Results](Results/) | Benchmark and ablation tables, four composite figures, held-out predictions, checkpoints, and execution records |
| [tests](tests/) | Data splits, mathematical layers, comparator training, stress perturbations, and result validation |
| [Baselines](Baselines/) | Temporal CNN, KAN-HAR, RL-KAN feature extractor, KAN–MLP–Mixer, and source-study details |
| [Experiments](Experiments/) | Exact training plans and scripts for training, comparison, evaluation, plotting, and packaging |


## Run locally

Python 3.10 or later is recommended. Run commands from the archive root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run_all.sh test
bash run_all.sh report
python -m Experiments.verify
```

The report command uses the bundled held-out predictions, including the simulated stress conditions. It rebuilds the analysis tables and figures offline. Use the reproduction command to prepare the datasets, retrain the models, and regenerate the simulations.

```bash
GEOMETRY_JOBS=4 bash run_all.sh reproduce
```

Use `GEOMETRY_DATA` to point to an alternate dataset cache and `GEOMETRY_REPRODUCTION` to choose the reproduction directory. Exact run configurations are in `Experiments/Plans/All.json`. The CPU archive uses the included spline kernel and a C++17 compiler. Set `GEOMETRY_SPLINE_BACKEND=pytorch` to use the pure PyTorch implementation. Numerical replay can vary with hardware, backend, or PyTorch version.

The archive includes every Geometry TempoKAN checkpoint, selected ablation checkpoints, and one seed-42 example checkpoint per baseline and protocol. The [checkpoint inventory](Results/Execution/Checkpoint_Inventory.csv) lists the included weights. Every fit includes held-out predictions, training history, metrics, and an exact training configuration.

## Predict from a checkpoint

```bash
python -m Proposed_Method.predict \
  --checkpoint Results/Runs/uci_har/clean/geometry_kan/fold00_seed42/checkpoint.pt \
  --input windows.npy --output predictions.csv
```

Input is a NumPy array of unstandardized 50 Hz windows with shape `N × 128 × channels`. The checkpoint supplies its training normalization and class order. See [dataset preparation](Datasets/README.md) for the channel layout.
