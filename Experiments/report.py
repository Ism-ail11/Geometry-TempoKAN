"""Build the local HTML results index from the released CSV tables."""

import argparse
from html import escape
import json
from pathlib import Path

import pandas as pd

from .registry import ABLATIONS, BASELINES, DATA_NAMES, NAMES, PRIMARY, TASKS


STYLE = """
:root{color-scheme:light;--ink:#172b45;--muted:#52677e;--teal:#087e77;--grid:#d8e4ef;--soft:#e8f6f4}
*{box-sizing:border-box}body{margin:0;background:#f2f6fa;color:var(--ink);font:16px/1.6 ui-sans-serif,system-ui}
main{max-width:1150px;margin:0 auto;background:white}header{background:#172b45;color:white;padding:53px 6% 40px}
h1{font-size:clamp(34px,5vw,58px);letter-spacing:-.045em;line-height:1.1;margin:0}
header p{color:#d9e8f0;margin:14px 0 0;max-width:740px}nav{padding:14px 6%;display:flex;gap:12px 25px;flex-wrap:wrap;background:#edf5f7}
a{color:#126868;font-weight:650}section{padding:0 6% 28px}h2{font-size:27px;margin:30px 0 8px;letter-spacing:-.03em}
p{margin:9px 0 15px}.note{color:var(--muted);font-size:13px}
.scroll{overflow-x:auto;border:1px solid var(--grid);border-radius:9px;margin:14px 0 7px}
table{border-collapse:collapse;width:100%;min-width:660px;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;border-bottom:1px solid #edf2f7;padding:10px 13px}
th{background:#edf3f8}th:first-child,td:first-child{text-align:left}tbody tr:nth-child(even){background:#f9fbfd}
tbody tr.emphasis{background:var(--soft);font-weight:700}
figure{padding:10px;margin:14px 0 22px;border:1px solid var(--grid);border-radius:9px}
figure img{width:100%;display:block}figcaption{padding:8px 10px;color:var(--muted);font-size:13px}
ul{padding-left:22px}.links a{margin-right:12px}.section-end{padding-bottom:36px}
"""


def selection(rows, dataset, regime, model):
    found = rows.loc[(rows.dataset == dataset) & (rows.regime == regime) & (rows.model == model)]
    return None if found.empty else found.iloc[0]


def table(headers, rows, highlight=None):
    heading = "".join(f"<th scope='col'>{escape(str(name))}</th>" for name in headers)
    body = []
    for i, row in enumerate(rows):
        cls = " class='emphasis'" if highlight == i else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{heading}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def benchmark_table(scores):
    headers = ("Model", "UCI HAR clean", "PAMAP2 clean", "SELFBACK clean", "SELFBACK unseen pace")
    ranks = (PRIMARY, *BASELINES, *ABLATIONS)
    rows = []
    for rank in ranks:
        cells = [selection(scores, ds, regime, rank) for ds, regime in TASKS]
        if all(item is None for item in cells):
            continue
        values = [f"{item.macro_f1:.2%} [{100*item.low:.2f}, {100*item.high:.2f}]"
                  if item is not None else "—" for item in cells]
        rows.append([NAMES[rank], *values])
    return table(headers, rows, highlight=0)


def paired_table(effects, models):
    headers = ("Comparison", "UCI HAR clean", "PAMAP2 clean", "SELFBACK clean", "SELFBACK unseen pace")
    rows = []
    for model in models:
        items = [selection(effects, ds, regime, model) for ds, regime in TASKS]
        rows.append([NAMES[model], *(f"{r.delta_pp:+.2f} [{r.low_pp:+.2f}, {r.high_pp:+.2f}]" if r is not None else "—"
                                   for r in items)])
    return table(headers, rows)


def stress_table(stress):
    headers = ("Dataset and condition", "Clean macro-F1", "Stressed macro-F1", "Change · pp (95% interval)")
    labels = {"sample_gap": "25% window interpolated", "gaussian_noise": "Gaussian noise at 0.35× channel SD",
              "rotation_with_outage": "60° rotation with one sensor group absent"}
    rows = []
    for dataset, name in DATA_NAMES.items():
        primary = stress.loc[(stress.dataset == dataset) & (stress.model == PRIMARY)]
        for family, label in labels.items():
            chosen = primary.loc[primary.family == family].sort_values("level").iloc[-1]
            rows.append((f"{name} · {label}", f"{chosen.clean_macro_f1:.2%}", f"{chosen.macro_f1:.2%}",
                         f"{chosen.change_from_clean_pp:+.2f} [{chosen.change_low_pp:+.2f}, {chosen.change_high_pp:+.2f}]"))
    return table(headers, rows)


def calibration_table(calibration):
    headers = ("Test context", "Seed", "Top-label accuracy", "Mean confidence", "ECE", "Top-label Brier")
    focus = calibration.loc[(calibration.model == PRIMARY) &
                            calibration.condition.isin(("clean", "unseen_pace_union"))]
    rows = []
    for row in focus.itertuples():
        context = DATA_NAMES[row.dataset] + (" · unseen pace" if row.regime == "pace" else " · clean")
        rows.append((context, row.seeds, f"{row.accuracy:.2%}", f"{row.mean_confidence:.2%}",
                     f"{row.top_label_ece:.2%}", f"{row.top_label_brier:.4f}"))
    return table(headers, rows)


def figures(root):
    labels = {
        "01_Learning_Curves": "Validation learning curves across the four split protocols.",
        "02_Robustness_Curves": "Mounting rotation, sensor loss, missing samples, Gaussian noise, and recorded pace.",
        "03_Confusion_Matrices": "Row-normalized confusion matrices for Geometry TempoKAN.",
        "04_Benchmark_Bar_Chart": "Held-out macro-F1 with participant intervals. Each panel uses the same comparisons.",
    }
    blocks = []
    for file, label in labels.items():
        if not (root / "Figures" / f"{file}.svg").is_file():
            raise ValueError(f"Missing figure {file}.svg")
        blocks.append(
            f"<figure><img src='Figures/{file}.svg' alt='{escape(label)}'><figcaption>{escape(label)} "
            f"<a href='Figures/{file}.pdf'>PDF</a> · <a href='Figures/{file}.png'>PNG</a> · "
            f"<a href='Figures/{file}.svg'>SVG</a></figcaption></figure>"
        )
    return "\n".join(blocks)


def write_report(root):
    scores = pd.read_csv(root / "Benchmark" / "Matched_Seeds.csv")
    baseline = pd.read_csv(root / "Benchmark" / "Matched_Seed_Effects.csv")
    ablation = pd.read_csv(root / "Ablations" / "Matched_Seed_Effects.csv")
    stress = pd.read_csv(root / "Robustness" / "Stress_Sweeps.csv")
    calibration = pd.read_csv(root / "Diagnostics" / "Calibration.csv")
    audit = json.loads((root / "Execution" / "Audit.json").read_text(encoding="utf-8"))
    expanded = json.loads((root / "Execution" / "Expanded_Audit.json").read_text(encoding="utf-8"))
    output = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Geometry TempoKAN results</title><style>{STYLE}</style></head>
<body><main>
<header><h1>Geometry TempoKAN</h1><p>Participant-disjoint activity recognition on UCI HAR, PAMAP2, and SELFBACK.</p></header>
<nav aria-label="Results sections"><a href="#benchmark">Benchmark</a><a href="#ablations">Ablations</a>
<a href="#stress">Stress tests</a><a href="#calibration">Calibration</a>
<a href="#figures">Figures</a><a href="#files">Data and tables</a></nav>
<section id="benchmark">
<h2>Matched benchmark</h2>
<p>{audit['model_fits']} completed fits, including {audit['baseline_fits']} independent baseline fits.
All nine models use the same participant splits with training seeds 42 and 77.</p>
{benchmark_table(scores)}
<p class="note">Cells give macro-F1 with 95% participant-bootstrap intervals in percent. Each participant’s
confusion matrices are averaged across the two seeds before scoring. Unseen pace averages slow and fast macro-F1
with the same nonwalking windows in each condition.</p>
<h3>Paired baseline differences</h3>
{paired_table(baseline, BASELINES)}
<p class="note">Cells give Geometry TempoKAN minus the comparator in macro-F1 percentage points, with paired
95% participant intervals. Each draw resamples participants jointly across the models.</p>
<p><a href="Benchmark/Matched_Seeds.csv">Matched-seed benchmark and seed ranges</a> ·
<a href="Benchmark/Matched_Seed_Effects.csv">Paired baseline CSV</a> ·
<a href="Benchmark/Participant_Scores.csv">Participant-level scores</a> ·
<a href="Benchmark/Resources.csv">Model size and compute</a> · <a href="../Baselines/README.md">Source studies and adaptations</a></p>
</section>
<section id="ablations">
<h2>Controlled ablations</h2>
<p>Each control changes one part of the Geometry model or its training recipe. The posture-head removal
also reduces capacity.</p>
{paired_table(ablation, ABLATIONS)}
<p class="note">Cells give Geometry TempoKAN minus the named control in macro-F1 percentage points,
with paired 95% participant intervals. All ablations share the same two training seeds and held-out windows.</p>
<p><a href="Ablations/Matched_Seed_Effects.csv">Two-seed ablation effects</a> ·
<a href="Ablations/Rotation_Effects.csv">Rotation-specific ablation effects</a> ·
<a href="../Proposed_Method/README.md">Components tested</a></p>
</section>
<section id="stress">
<h2>Stress tests</h2>
<p>{expanded['extra_stress_rows']} dataset–model–severity estimates measure missing samples, Gaussian noise, and
joint sensor loss and mounting rotation on clean-protocol held-out participants. The table shows Geometry TempoKAN
at the highest tested severity in each sweep, with changes from its own clean score.</p>
{stress_table(stress)}
<p class="note">All extended simulation sweeps use seed 42. Sensor loss removes a complete IMU on PAMAP2 and one
three-axis sensor group on UCI HAR and SELFBACK. Zero degrees in the joint sweep still includes the sensor outage.
The rotation–outage contrast measures how much the joint macro-F1 change departs from adding the two single-stress changes.</p>
<p><a href="Robustness/Stress_Sweeps.csv">All models and severity levels</a> ·
<a href="Robustness/Stress_Paired_Effects.csv">Paired stress-test effects</a> ·
<a href="Robustness/Rotation_Outage_Interaction.csv">Rotation and outage contrasts</a> ·
<a href="Diagnostics/Stress_Per_Class.csv">Per-activity changes at the highest severity</a> ·
<a href="Robustness/Matched_Pace.csv">Measured walking recall and pace errors</a> ·
<a href="Robustness/Matched_Corruptions.csv">Two-seed corruption estimates</a></p>
</section>
<section id="calibration">
<h2>Confidence and selective prediction</h2>
<p>These rows compare Geometry TempoKAN’s confidence with observed correctness. The threshold table records
coverage and error among accepted predictions for all nine models.</p>
{calibration_table(calibration)}
<p class="note">Expected calibration error (ECE) uses ten equal-width bins. The top-label Brier score measures
squared error between the predicted label’s confidence and its correctness. Unseen pace pools slow and fast
walking with each nonwalking window counted once.</p>
<p><a href="Diagnostics/Calibration.csv">All {expanded['calibration_rows']} calibration contexts</a> ·
<a href="Diagnostics/Reliability_Bins.csv">Reliability bins</a> ·
<a href="Diagnostics/Risk_Coverage.csv">All {expanded['risk_coverage_rows']} confidence-threshold results</a></p>
</section>
<section id="figures"><h2>Composite figures</h2>{figures(root)}</section>
<section id="files" class="section-end">
<h2>Data and tables</h2>
<ul>
<li><a href="../Datasets/Splits.csv">Participant split assignments</a></li>
<li><a href="Benchmark/Scores.csv">Seed-42 benchmark including the checkpoint average</a> and
<a href="Benchmark/Seed_Replicates.csv">per-seed results</a></li>
<li><a href="Robustness/Rotation.csv">Rotation performance</a></li>
<li><a href="Diagnostics/Matched_Confusions.csv">Two-seed confusion matrices</a> and
<a href="Diagnostics/Matched_Per_Class.csv">per-class precision, recall, and F1</a></li>
<li><a href="Diagnostics/Participant_Variability.csv">Participant-level score distributions</a> and
<a href="Diagnostics/Seed_Agreement.csv">Prediction agreement between seeds</a></li>
<li><a href="Training/Epoch_History.csv">Per-fold learning curves</a> and
<a href="Execution/Manifest.json">executed configuration manifest</a></li>
</ul>
</section></main></body></html>
"""
    (root / "Index.html").write_text(output, encoding="utf-8")
    print(f"Wrote {root / 'Index.html'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("Results"))
    write_report(p.parse_args().results)


if __name__ == "__main__":
    main()
