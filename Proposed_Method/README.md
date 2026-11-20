# Geometry TempoKAN

The elapsed-time backbone reads ordered sensor patches and window summaries. The motion specialist projects each triad onto an estimated gravity direction and onto its perpendicular plane. It encodes the signed vertical value, horizontal magnitude, two lagged alignments, and signed horizontal turning. The harmonic layer projects the five channels at 17 fundamentals and their second and third harmonics. It learns a single set of frequency weights shared across windows.

Proper rotations applied consistently to colocated sensors preserve the motion specialist’s input descriptors under this coordinate model. The elapsed branch and posture specialist retain absolute orientation cues. The posture specialist also uses axis-wise amplitude, amplitude of first differences, stabilized covariance, and drift. Each readout changes class probabilities within its own activity group while preserving the two group probabilities from the elapsed backbone.

The training objective combines weighted cross-entropy with elapsed, motion, and posture supervision and a motion consistency penalty on retimed signals. Model selection uses validation participants. Score tables identify the common participant cohorts used in paired comparisons.

| Control | Change |
|---|---|
| Original-axis features | Computes horizontal lag alignments and turning in the fixed sensor plane while retaining the scalar gravity projections |
| MLP readouts | Replaces the two spline projections with MLPs whose parameter counts match within 0.1% |
| Without posture head | Removes the posture specialist and its conditional training loss |
| Without mounting augmentation | Disables training-time mounting rotation while keeping the architecture and other augmentations |

The no-posture comparison also changes capacity. See [ablation results](../Results/Ablations/Paired_Effects.csv) and [rotation-specific ablations](../Results/Ablations/Rotation_Effects.csv).
