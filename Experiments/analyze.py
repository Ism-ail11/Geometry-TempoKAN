"""Validate held-out predictions and generate benchmark, ablation, and stress tables."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Datasets.prepare import CLASSES
from Datasets.protocol import pace_masks
from .registry import ABLATIONS, BASELINES, DATA_NAMES, FOLDS, MODELS, NAMES, PRIMARY, TASKS
from .stress import ANGLES, AVERAGE
from .train import confusion, metrics_from_cm, write_json


BOOTSTRAP_SEED = 20261121
PACE = ("pace_slow", "pace_medium", "pace_fast")
ENDPOINTS = ("clean", "unseen_pace")


@dataclass
class Predictions:
    index: np.ndarray
    subject: np.ndarray
    actual: np.ndarray
    pace: np.ndarray
    by_condition: dict


def combined(blocks):
    parts = [value for _, value in sorted(blocks)]
    return Predictions(*(np.concatenate([getattr(p, key) for p in parts])
                         for key in ("index", "subject", "actual", "pace")),
                       {name: np.concatenate([p.by_condition[name] for p in parts])
                        for name in parts[0].by_condition})


def check_aligned(a, b, label):
    if not all(np.array_equal(getattr(a, key), getattr(b, key))
               for key in ("index", "subject", "actual", "pace")):
        raise ValueError(f"Different held-out windows for {label}.")


def read_predictions(path, record):
    cfg = record["config"]
    with np.load(path, allow_pickle=False) as saved:
        keys = ("index", "subject", "actual", "pace")
        arrays = [saved[k] for k in keys]
        labels = saved["pred"]
        if labels.shape != saved["confidence_u8"].shape or any(a.shape != (labels.shape[1],) for a in arrays):
            raise ValueError(f"Prediction dimensions differ in {path}")
        value = Predictions(*arrays, dict(zip(saved["conditions"].tolist(), labels, strict=True)))
    if len(value.index) != record["counts"]["test"] or np.unique(value.index).size != len(value.index):
        raise ValueError(f"Unexpected test size or repeated windows in {path}")
    people = {name: set(record["subjects"][name]) for name in ("train", "val", "test")}
    if (set(value.subject) != people["test"] or people["train"] & people["val"]
            or people["train"] & people["test"] or people["val"] & people["test"]):
        raise ValueError(f"Non-disjoint participant allocation in {path}")
    n = len(CLASSES[cfg["dataset"]])
    if (value.actual.min() < 0 or labels.min() < 0 or value.actual.max() >= n or labels.max() >= n):
        raise ValueError(f"Class index out of range in {path}")
    rows = {name: (np.ones(len(value.index), dtype=bool), pred) for name, pred in value.by_condition.items()}
    if cfg["regime"] == "pace":
        rows.update({name: (mask, value.by_condition["clean"]) for name, mask in pace_masks(value.pace).items()})
    claims = {entry["condition"]: entry for entry in record["metrics"]}
    if rows.keys() != claims.keys():
        raise ValueError(f"Prediction conditions differ in {path}")
    for name, (mask, pred) in rows.items():
        observed = metrics_from_cm(confusion(value.actual[mask], pred[mask], n))
        if any(abs(result - claims[name][metric]) > 1e-9 for metric, result in observed.items()):
            raise ValueError(f"Saved {name} metrics disagree with predictions in {path}.")
    return value, len(rows)


def collect(root):
    grouped, records, hist = defaultdict(list), [], []
    checks = 0
    for file in sorted((root / "Runs").glob("*/*/*/*/run.json")):
        record = json.loads(file.read_text(encoding="utf-8"))
        c = record["config"]
        ds, regime, model, seed, fold = (c[name] for name in ("dataset", "regime", "model", "seed", "fold"))
        if (ds, regime) not in TASKS or model not in MODELS or not 0 <= fold < FOLDS[ds]:
            raise ValueError(f"Unexpected run configuration in {file}")
        identifier = (ds, regime, model, f"fold{fold:02d}_seed{seed}")
        if file.relative_to(root / "Runs").parts[:-1] != identifier:
            raise ValueError(f"Wrong result path for {identifier}")
        data, count = read_predictions(file.with_name("predictions.npz"), record)
        checks += count
        grouped[ds, regime, model, seed].append((fold, data))
        keys = {"dataset": ds, "regime": regime, "model": model, "seed": seed, "fold": fold}
        for item in record["history"]:
            hist.append({**keys, **item})
        records.append({**keys, "best_epoch": record["best_epoch"], "parameters": record["params"],
                        "training_seconds": record["training_seconds"], "latency_ms": record["latency_ms_batch1"],
                        "config": c, "spline_backend": record.get("spline_backend", "pytorch")})
    for ds, regime in TASKS:
        for model in MODELS:
            if (ds, regime, model, 42) not in grouped:
                raise ValueError(f"Missing seed-42 comparison {ds}/{regime}/{model}.")
    complete = {}
    for key, blocks in grouped.items():
        folds = [fold for fold, _ in blocks]
        if sorted(folds) != list(range(FOLDS[key[0]])):
            raise ValueError(f"Incomplete or duplicate folds for {key}.")
        item = combined(blocks)
        if len(np.unique(item.index)) != len(item.index):
            raise ValueError(f"Overlapping folds for {key}.")
        complete[key] = item
    for ds, regime, model, seed in complete:
        check_aligned(complete[ds, regime, PRIMARY, 42], complete[ds, regime, model, seed],
                      f"{ds}/{regime}/{model}/{seed}")
    return complete, records, hist, checks


def macro_f1_array(matrix):
    actual = matrix.sum(axis=-1)
    predicted = matrix.sum(axis=-2)
    correct = np.diagonal(matrix, axis1=-2, axis2=-1)
    return (np.divide(2 * correct, actual + predicted, where=(actual + predicted) > 0,
                      out=np.zeros_like(correct, dtype=float)) * (actual > 0)).sum(axis=-1) / (actual > 0).sum(axis=-1).clip(1)


def participants_cm(data, prediction, visible, classes):
    rows = []
    for person in np.unique(data.subject):
        mask = (data.subject == person) & visible
        rows.append(confusion(data.actual[mask], prediction[mask], classes))
    return np.stack(rows)


def score_matrices(matrix, draws):
    n = len(matrix)
    weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(n, np.full(n, 1 / n), size=draws)
    scores = macro_f1_array(np.einsum("bs,sij->bij", weights, matrix))
    return metrics_from_cm(matrix.sum(0)), scores


def interval(samples):
    return tuple(float(a) for a in np.quantile(samples, (0.025, 0.975)))


def endpoint(data, regime, classes, draws):
    clean = data.by_condition["clean"]
    if regime == "pace":
        masks = pace_masks(data.pace)
        pairs = [score_matrices(participants_cm(data, clean, masks[c], classes), draws)
                 for c in ("pace_slow", "pace_fast")]
        return {key: float(np.mean([x[0][key] for x in pairs])) for key in ("macro_f1", "accuracy", "balanced_accuracy")}, (
            pairs[0][1] + pairs[1][1]) / 2
    visible = np.ones(len(data.actual), dtype=bool)
    return score_matrices(participants_cm(data, clean, visible, classes), draws)


def stress_predictions(root, ds, regime, control):
    path = root / "Robustness" / "Predictions" / f"{ds}_{regime}.npz"
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as source:
        candidate = Predictions(*(source[key] for key in ("index", "subject", "actual", "pace")), {})
        check_aligned(control, candidate, path)
        angles = ANGLES if regime == "clean" else (0,)
        result = {(angle, name): source[f"{angle}_{name}"] for angle in angles for name in (*MODELS, AVERAGE)}
    for array in result.values():
        if array.shape != control.actual.shape or array.min() < 0 or array.max() >= len(CLASSES[ds]):
            raise ValueError(f"Invalid stress label array in {path}")
    return result


def write_table(root, name, rows):
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False, float_format="%.8g")


def make_results(root, draws=1500):
    groups, records, history, checks = collect(root)
    scores, against, ablations, corruption, rotation, pace, confusion_rows, class_rows = [], [], [], [], [], [], [], []
    rotation_effects = []
    for ds, regime in TASKS:
        classes = CLASSES[ds]
        current = {m: groups[ds, regime, m, 42] for m in MODELS}
        reference = current[PRIMARY]
        stress = stress_predictions(root, ds, regime, reference)
        if stress:
            for m in MODELS:
                if not np.array_equal(stress[0, m], current[m].by_condition["clean"]):
                    raise ValueError(f"Stress replay at 0 degrees disagrees on {ds}/{regime}/{m}.")
            current[AVERAGE] = Predictions(reference.index, reference.subject, reference.actual, reference.pace,
                                           {"clean": stress[0, AVERAGE]})
        common = {"dataset": ds, "regime": regime, "seed": 42, "test_participants": len(np.unique(reference.subject)),
                  "folds": FOLDS[ds]}
        base_score, base_samples = endpoint(reference, regime, len(classes), draws)
        all_windows = np.ones(len(reference.actual), dtype=bool)
        show = pace_masks(reference.pace)["pace_fast"] if regime == "pace" else all_windows
        for model, prediction in current.items():
            info = {**common, "model": model, "method": NAMES[model]}
            value, samples = endpoint(prediction, regime, len(classes), draws)
            lo, hi = interval(samples)
            outcome = "unseen_pace" if regime == "pace" else "clean"
            scores.append({**info, "condition": outcome, "macro_f1": value["macro_f1"], "low": lo, "high": hi,
                           "accuracy": value["accuracy"], "balanced_accuracy": value["balanced_accuracy"]})
            if model in (*BASELINES, *ABLATIONS):
                d_lo, d_hi = interval(100 * (base_samples - samples))
                effect = {**info, "condition": outcome, "geometry_macro_f1": base_score["macro_f1"],
                          "control_macro_f1": value["macro_f1"], "delta_pp": 100*(base_score["macro_f1"] - value["macro_f1"]),
                          "low_pp": d_lo, "high_pp": d_hi}
                (ablations if model in ABLATIONS else against).append(effect)
            if regime == "pace":
                for name, selected in pace_masks(prediction.pace).items():
                    estimate, boot = score_matrices(participants_cm(prediction, prediction.by_condition["clean"],
                                                                     selected, len(classes)), draws)
                    lo, hi = interval(boot)
                    code = dict(zip(PACE, (0, 1, 2)))[name]
                    walking = prediction.by_condition["clean"][prediction.pace == code]
                    pace.append({**info, "condition": name, **estimate, "low": lo, "high": hi,
                                 "walking_recall": float((walking == 0).mean()),
                                 "walking_as_jogging": float((walking == 3).mean()), "walking_windows": len(walking)})
            elif model in MODELS:
                for condition, pred in prediction.by_condition.items():
                    estimate, boot = score_matrices(participants_cm(prediction, pred, all_windows, len(classes)), draws)
                    lo, hi = interval(boot)
                    corruption.append({**info, "condition": condition, **estimate, "low": lo, "high": hi})
            matrix = confusion(prediction.actual[show], prediction.by_condition["clean"][show], len(classes))
            for i, truth in enumerate(classes):
                support = int(matrix[i].sum())
                denominator = int(matrix[:, i].sum()) + support
                class_rows.append({**info, "condition": "pace_fast" if regime=="pace" else "clean",
                                   "activity": truth, "support": support, "f1": 2 * matrix[i,i] / denominator if denominator else 0})
                for j, estimated in enumerate(classes):
                    confusion_rows.append({**info, "condition": "pace_fast" if regime=="pace" else "clean",
                                           "actual": truth, "predicted": estimated, "count": int(matrix[i,j]),
                                           "row_fraction": matrix[i,j] / support if support else 0})
        if regime == "clean":
            moving = np.array([c.lower() not in {"sitting", "standing", "lying"} for c in classes])[reference.actual]
            by_rotation = {}
            for (angle, name), labels in stress.items():
                estimate, sample = score_matrices(participants_cm(reference, labels, all_windows, len(classes)), draws)
                move, move_sample = score_matrices(participants_cm(reference, labels, moving, len(classes)), draws)
                lo, hi = interval(sample)
                m_lo, m_hi = interval(move_sample)
                rotation.append({**common, "model": name, "method": NAMES[name], "angle": angle, **estimate,
                                 "low": lo, "high": hi, "moving_macro_f1": move["macro_f1"],
                                 "moving_low": m_lo, "moving_high": m_hi})
                by_rotation[angle, name] = ((estimate["macro_f1"], sample), (move["macro_f1"], move_sample))
            if stress:
                for angle in ANGLES:
                    for model in ABLATIONS:
                        for group, (a, a_samples), (b, b_samples) in zip(
                            ("all", "moving"), by_rotation[angle, PRIMARY], by_rotation[angle, model]
                        ):
                            lo, hi = interval(100*(a_samples-b_samples))
                            rotation_effects.append({**common, "angle": angle, "model": model, "method": NAMES[model],
                                                     "ground_truth_group": group, "delta_pp": 100*(a-b),
                                                     "low_pp": lo, "high_pp": hi})
    replicate = []
    for ds, regime, model, seed in sorted(groups):
        value, _ = endpoint(groups[ds, regime, model, seed], regime, len(CLASSES[ds]), 1)
        replicate.append({"dataset": ds, "regime": regime, "model": model, "method": NAMES[model],
                          "seed": seed, "folds": FOLDS[ds], "macro_f1": value["macro_f1"]})
    seed_summary = []
    for ds, model in sorted({(ds, model) for ds, regime, model, _ in groups if regime == "clean"}):
        seeds = sorted(seed for d, r, m, seed in groups if (d, r, m) == (ds, "clean", model))
        if len(seeds) < 2:
            continue
        matrices = []
        for seed in seeds:
            item = groups[ds, "clean", model, seed]
            selected = np.ones(len(item.actual), dtype=bool)
            matrices.append(participants_cm(item, item.by_condition["clean"], selected, len(CLASSES[ds])))
        estimate, boot = score_matrices(np.mean(matrices, axis=0), draws)
        lo, hi = interval(boot)
        seed_summary.append({"dataset": ds, "model": model, "method": NAMES[model], "seed_count": len(seeds),
                             "seeds": " ".join(map(str, seeds)), "folds": FOLDS[ds], "macro_f1": estimate["macro_f1"],
                             "low": lo, "high": hi})
    resources = pd.DataFrame([{key: value for key, value in row.items() if key != "config"} for row in records])
    grouped_resources = resources.groupby(["dataset", "regime", "model", "seed"], as_index=False).agg(
        parameters=("parameters", "first"), median_training_seconds=("training_seconds", "median"),
        median_batch1_ms=("latency_ms", "median"), median_best_epoch=("best_epoch", "median"))
    for path, rows in (
        ("Benchmark/Scores.csv", scores), ("Benchmark/Paired_Baselines.csv", against),
        ("Ablations/Paired_Effects.csv", ablations), ("Benchmark/Seed_Replicates.csv", replicate),
        ("Ablations/Rotation_Effects.csv", rotation_effects), ("Benchmark/Seed_Summary.csv", seed_summary),
        ("Robustness/Corruptions.csv", corruption), ("Robustness/Rotation.csv", rotation),
        ("Robustness/Recorded_Pace.csv", pace), ("Diagnostics/Per_Class.csv", class_rows),
        ("Diagnostics/Confusion_Counts.csv", confusion_rows), ("Training/Epoch_History.csv", history),
        ("Benchmark/Resources.csv", grouped_resources),
    ):
        write_table(root, path, rows)
    write_json(root / "Execution" / "Manifest.json", [{"config": row["config"], "spline_backend": row["spline_backend"]}
                                                       for row in records])
    audit = {"model_fits": len(records), "geometry_and_ablations": sum(r["model"] not in BASELINES for r in records),
             "baseline_fits": sum(r["model"] in BASELINES for r in records),
             "matched_tasks": len(TASKS), "validated_metric_rows": checks, "bootstrap_draws": draws,
             "rotation_rows": len(rotation)}
    write_json(root / "Execution" / "Audit.json", audit)
    print(json.dumps(audit, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("Results"))
    p.add_argument("--draws", type=int, default=1500)
    args = p.parse_args()
    if args.draws < 1:
        p.error("draws must be positive")
    make_results(args.results, args.draws)


if __name__ == "__main__":
    main()
