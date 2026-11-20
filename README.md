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

## Read the results

The [matched benchmark](Results/Benchmark/Matched_Seeds.csv) compares all nine models under seeds 42 and 77 on participant-disjoint splits. For each model, the analysis averages each participant’s confusion matrices across training seeds, then pools those matrices to calculate macro-F1. The 95% intervals use 1,500 participant-bootstrap draws. The table also records each score’s range across seeds. The unseen-pace endpoint averages the separate slow and fast SELFBACK evaluations, each including the same six nonwalking activities.

[Paired baseline effects](Results/Benchmark/Matched_Seed_Effects.csv) and [paired ablation effects](Results/Ablations/Matched_Seed_Effects.csv) use identical held-out windows. [Per-seed results](Results/Benchmark/Seed_Replicates.csv) include the additional seed-123 UCI HAR runs. The [seed-42 table](Results/Benchmark/Scores.csv) also evaluates the equal-probability average of the Geometry spline and MLP checkpoints.

The [baseline notes](Baselines/README.md) describe the local implementations, source studies, and published evaluation protocols. The four composite figures cover validation curves, robustness curves, confusion matrices, and one comparative bar chart. Each figure is available as PDF, SVG, and PNG.

### Stress tests and diagnostics

| Experiment | Evaluation | Results |
|---|---|---|
| Missing samples | One gap of 0, 8, 16, or 32 samples in a 128-sample window, replaced by linear interpolation across its endpoints | [Severity estimates](Results/Robustness/Stress_Sweeps.csv) |
| Gaussian sensor noise | Noise from 0 to 0.35 times each channel’s training-fold standard deviation | [Severity estimates](Results/Robustness/Stress_Sweeps.csv) |
| Rotation with sensor loss | One fixed sensor group is dropped per participant and replaced with training channel means. Remaining sensors rotate by 0°, 15°, 30°, or 60° | [Paired comparisons](Results/Robustness/Stress_Paired_Effects.csv) |
| Recorded walking pace | SELFBACK slow, medium, and fast walking, including recall and errors into jogging | [Pace estimates](Results/Robustness/Matched_Pace.csv) |
| Confidence | Top-label ECE, Brier score, and accepted error across six confidence thresholds | [Calibration](Results/Diagnostics/Calibration.csv), [risk and coverage](Results/Diagnostics/Risk_Coverage.csv) |
| Activity-level errors | Per-activity precision, recall, and F1 pooled across seeds and at peak stress | [Matched class scores](Results/Diagnostics/Matched_Per_Class.csv), [stress class scores](Results/Diagnostics/Stress_Per_Class.csv), [confusions](Results/Diagnostics/Matched_Confusions.csv) |
| Stability | Held-out participant score distributions and prediction agreement across training seeds | [Participant distributions](Results/Diagnostics/Participant_Variability.csv), [seed agreement](Results/Diagnostics/Seed_Agreement.csv) |

The severity sweeps use seed 42 for all nine models, sharing the perturbations on the same held-out windows in each fold. The sensor group is one IMU on PAMAP2 and one three-axis signal on UCI HAR and SELFBACK. At 0° in the joint sweep, the sensor outage still applies. The [rotation and outage contrasts](Results/Robustness/Rotation_Outage_Interaction.csv) compare the two individual perturbations, their combination, and their shared clean reference. The noise and gap sweeps include a clean zero-severity control. The confidence analysis uses the stored eight-bit top-label probabilities and ten equal-width bins. For unseen pace, the confidence union counts the shared nonwalking windows once.

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
