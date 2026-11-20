# Baseline implementations and published KAN-HAR studies

The executed comparators use the same 128-sample input windows, channels, participant splits, optimizer family, training budget, validation rule, and input augmentations as Geometry TempoKAN. The compared architectures have their own widths and objectives. Parameter counts are recorded in [Resources.csv](../Results/Benchmark/Resources.csv). The matching MLP ablation tests the spline readouts at similar capacity.

| Executed comparator | Implemented architecture | Source-study adaptation |
|---|---|---|
| Temporal CNN | Patch projection, four residual dilated convolution blocks, and mean, deviation, and max pooling | Conventional temporal baseline with width 48 |
| KAN-HAR | Two stacked cubic-spline projections with a SiLU base acting on the flattened window | Extends the two-layer KAN to all channels in each local dataset. Uses width 20 and end-to-end training on the current folds. |
| RL-KAN feature extractor | Shared five-output channel-wise filters, a ten-output across-channel layer, residual concatenation, and a 256–128-unit MLP head | Uses 16-sample nonoverlapping patches and fixed-grid spline projections under the current preprocessing. |
| KAN–MLP–Mixer | Cubic-spline patch embedding, two MLP mixing blocks, and an arctangent-edge classifier | Uses 16-sample patches, width 32, and a compact arctangent readout modeled on the published classifier design. |

## Published context

| Study | Original evaluation | Reported result or reference |
|---|---|---|
| Alikhani, KAN-HAR, 2025 | MotionSense with one three-axis accelerometer. Nineteen training participants and five test participants. | 90.38% accuracy and 90.52% F1 as labeled in Table 2. [Study and table](https://arxiv.org/html/2508.11186v1#S5) · [Source implementation](https://github.com/BahMoh/KAN-HAR) |
| Liu et al., KAN feature extractors, 2024 | Four inertial datasets. The PAMAP2 configuration uses 13 labels, 100 Hz, 200-sample windows, and four held-out participants per split. | The study compares residual, parallel, one-layer, and two-layer KAN feature extractors in its result figures. [Architecture and Table 2](https://arxiv.org/html/2406.11914v1) |
| Liu et al., KAN–MLP–Mixer, 2026 | Eight inertial datasets. The PAMAP2 input uses a single six-channel placement and 3-second windows. | Reports a 5.33% relative average macro-F1 improvement over the original MLP testbed across its eight datasets. [Study and Table 6](https://arxiv.org/html/2605.19031v1#S5) |
| TinyKAN-HAR, 2026 | Seen-class supervised evaluation and separate zero-shot and generalized zero-shot experiments. | Table 3 reports 98.0% UCI HAR and 97.1% PAMAP2 macro-F1 on its seen-class evaluation. [Study and Table 3](https://www.mdpi.com/2504-4990/8/3/55#sec5dot1) |

The published numbers retain the authors’ label spaces, input choices, splits, and metric definitions. The measured comparison figures use the executed reimplementations and the exact participant splits in this release.
