# Datasets and participant splits

The three official releases are available from the UCI Machine Learning Repository. Preparation records release hashes and window counts in [Metadata](Metadata/) and retains enough metadata to reconstruct subject-disjoint train, validation, and test sets.

| Dataset | Installed window shape | Channels | Evaluation |
|---|---:|---|---|
| UCI HAR | 10,299 × 128 × 6 | Total acceleration XYZ, then body gyroscope XYZ | Nine official test participants. Four of the other participants form validation. |
| PAMAP2 | 14,904 × 128 × 18 | Wrist, chest, then ankle. Each location has acceleration XYZ followed by gyroscope XYZ. | Twelve protocol activities on eight participants. Leave one participant out for test and a different one for validation. |
| SELFBACK | 35,623 × 128 × 6 | The synchronized `wt` release order for the wrist and thigh accelerometer triads | Five participant-disjoint folds covering 33 participants. The pace protocol trains and validates on medium walking and the six nonwalking activities. |

Use [Splits.csv](Splits.csv) for the exact participant IDs and window counts by fold and role. Data normalization is fitted on the training windows for each fold. PAMAP2 and SELFBACK are resampled to 50 Hz and split into 128-sample windows with 64-sample stride. Segmentation respects label, recording, and gap boundaries. The UCI HAR signals use the release’s existing windows.

```bash
python -m Datasets.prepare --data-dir Datasets/Cache
```

Place releases at `Datasets/Cache/raw/uci_har.zip`, `pamap2.zip`, and `selfback.zip` if downloading manually. Prepared data are written under `Datasets/Cache/prepared`.

- [UCI HAR release](https://doi.org/10.24432/C54S4K)
- [PAMAP2 release](https://doi.org/10.24432/C5NW2H)
- [SELFBACK release](https://doi.org/10.24432/C5SK65)
