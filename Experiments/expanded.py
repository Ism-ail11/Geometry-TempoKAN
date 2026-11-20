"""Summarize matched-seed performance, extra stress sweeps, and confidence diagnostics."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from Datasets.prepare import CLASSES
from Datasets.protocol import pace_masks
from .analyze import (Predictions, check_aligned, collect, endpoint, interval, macro_f1_array,
                      participants_cm, score_matrices, stress_predictions, write_table)
from .registry import ABLATIONS, BASELINES, DATA_NAMES, FOLDS, MODELS, NAMES, PRIMARY, TASKS
from .stress_sweeps import SWEEPS
from .train import write_json


CONFIDENCE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def matched_seed_ids(groups, dataset, regime):
    by_model = [{seed for ds, task, candidate, seed in groups if (ds, task, candidate) == (dataset, regime, model)}
                for model in MODELS]
    return sorted(set.intersection(*by_model))


def pooled_matrices(examples, condition, classes):
    """Average each held-out participant's confusion matrices across the fits."""
    matrices = []
    for item in examples:
        masks = pace_masks(item.pace)
        visible = masks[condition] if condition in masks else np.ones(len(item.actual), dtype=bool)
        name = "clean" if condition in masks else condition
        matrices.append(participants_cm(item, item.by_condition[name], visible, classes))
    return np.mean(matrices, axis=0)


def result_for(examples, conditions, classes, draws):
    mats = [pooled_matrices(examples, name, classes) for name in conditions]
    results = [score_matrices(matrix, draws) for matrix in mats]
    keys = ("macro_f1", "accuracy", "balanced_accuracy")
    values = {key: float(np.mean([row[key] for row, _ in results])) for key in keys}
    samples = np.mean([sample for _, sample in results], axis=0)
    per_person = np.mean([macro_f1_array(matrix) for matrix in mats], axis=0)
    return values, samples, per_person


def info_for(dataset, regime, model, seeds, example):
    return {"dataset": dataset, "regime": regime, "model": model, "method": NAMES[model],
            "seed_count": len(seeds), "seeds": " ".join(map(str, seeds)), "folds": FOLDS[dataset],
            "test_participants": int(np.unique(example.subject).size)}


def activity_scores(matrix, labels):
    actual, predicted = matrix.sum(1), matrix.sum(0)
    rows = []
    for idx, label in enumerate(labels):
        support, selected, correct = actual[idx], predicted[idx], matrix[idx, idx]
        rows.append({"actual": label, "windows_per_seed": int(round(support)),
                     "recall": correct / support if support else None,
                     "precision": correct / selected if selected else None,
                     "f1": 2 * correct / (support + selected) if support + selected else None})
    return rows


def matched_results(groups, root, draws):
    scores, baselines, ablations, participants, pace, corruption = [], [], [], [], [], []
    confusions, class_scores, variability = [], [], []
    for dataset, regime in TASKS:
        seeds = matched_seed_ids(groups, dataset, regime)
        if len(seeds) < 2:
            raise ValueError(f"At least two common training seeds are required for {dataset}/{regime}.")
        classes = len(CLASSES[dataset])
        endpoint_name = "unseen_pace" if regime == "pace" else "clean"
        conditions = ("pace_slow", "pace_fast") if regime == "pace" else ("clean",)
        pool = {model: [groups[dataset, regime, model, seed] for seed in seeds] for model in MODELS}
        primary, primary_samples, _ = result_for(pool[PRIMARY], conditions, classes, draws)
        for model, examples in pool.items():
            item = examples[0]
            metadata = info_for(dataset, regime, model, seeds, item)
            estimate, samples, person_scores = result_for(examples, conditions, classes, draws)
            low, high = interval(samples)
            seed_values = [endpoint(example, regime, classes, 1)[0]["macro_f1"] for example in examples]
            scores.append({**metadata, "condition": endpoint_name, **estimate,
                           "low": low, "high": high, "seed_min": min(seed_values), "seed_max": max(seed_values),
                           "mean_participant_macro_f1": float(person_scores.mean())})
            if model != PRIMARY:
                delta_low, delta_high = interval(100 * (primary_samples - samples))
                effect = {**metadata, "condition": endpoint_name, "geometry_macro_f1": primary["macro_f1"],
                          "control_macro_f1": estimate["macro_f1"],
                          "delta_pp": 100 * (primary["macro_f1"] - estimate["macro_f1"]),
                          "low_pp": delta_low, "high_pp": delta_high}
                (baselines if model in BASELINES else ablations).append(effect)
            people = np.unique(item.subject)
            participants.extend({**metadata, "condition": endpoint_name, "participant": int(person),
                                 "macro_f1": float(value)} for person, value in zip(people, person_scores))
            lower, q1, median, q3 = np.quantile(person_scores, (0.10, 0.25, 0.50, 0.75))
            worst = int(np.argmin(person_scores))
            variability.append({**metadata, "condition": endpoint_name, "minimum": float(person_scores[worst]),
                                "minimum_participant": int(people[worst]), "p10": lower, "q1": q1, "median": median,
                                "q3": q3, "maximum": float(person_scores.max())})
            inspected = tuple(pace_masks(item.pace)) if regime == "pace" else ("clean",)
            for name in inspected:
                matrix = pooled_matrices(examples, name, classes).sum(0)
                context = {**metadata, "condition": name}
                rows = activity_scores(matrix, CLASSES[dataset])
                class_scores.extend({**context, **row} for row in rows)
                for i, row in enumerate(rows):
                    support = row["windows_per_seed"]
                    confusions.extend({**context, "predicted": result, "mean_count": matrix[i, j],
                                       "actual": row["actual"],
                                       "row_fraction": matrix[i, j] / support if support else 0}
                                      for j, result in enumerate(CLASSES[dataset]))
            if regime == "pace":
                for name, code in (("pace_slow", 0), ("pace_medium", 1), ("pace_fast", 2)):
                    val, band, _ = result_for(examples, (name,), classes, draws)
                    mask = item.pace == code
                    predictions = np.concatenate([p.by_condition["clean"][mask] for p in examples])
                    low, high = interval(band)
                    pace.append({**metadata, "condition": name, **val, "low": low, "high": high,
                                 "walking_recall": float((predictions == 0).mean()),
                                 "walking_as_jogging": float((predictions == 3).mean()),
                                 "walking_windows_per_seed": int(mask.sum())})
            else:
                for condition in item.by_condition:
                    val, band, _ = result_for(examples, (condition,), classes, draws)
                    low, high = interval(band)
                    corruption.append({**metadata, "condition": condition, **val, "low": low, "high": high})
    for filename, rows in (
        ("Benchmark/Matched_Seeds.csv", scores), ("Benchmark/Matched_Seed_Effects.csv", baselines),
        ("Benchmark/Participant_Scores.csv", participants), ("Ablations/Matched_Seed_Effects.csv", ablations),
        ("Robustness/Matched_Pace.csv", pace), ("Robustness/Matched_Corruptions.csv", corruption),
        ("Diagnostics/Matched_Confusions.csv", confusions), ("Diagnostics/Matched_Per_Class.csv", class_scores),
        ("Diagnostics/Participant_Variability.csv", variability),
    ):
        write_table(root, filename, rows)
    return scores


def sweep_predictions(root, dataset, reference):
    path = root / "Robustness" / "Predictions" / f"{dataset}_sweeps.npz"
    with np.load(path, allow_pickle=False) as source:
        stored = Predictions(*(source[key] for key in ("index", "subject", "actual", "pace")), {})
        check_aligned(reference, stored, path)
        results = {(family, level, model): source[f"{family}_{level}_{model}"]
                   for family, levels in SWEEPS.items() for level in levels for model in MODELS}
    expected_count = len(CLASSES[dataset])
    for key, values in results.items():
        if values.shape != reference.actual.shape or values.min() < 0 or values.max() >= expected_count:
            raise ValueError(f"Bad sweep predictions in {path} for {key}")
    return results


def rotation_outage_effects(dataset, reference, rotation, sweeps, draws):
    visible = np.ones(len(reference.actual), dtype=bool)
    classes = len(CLASSES[dataset])

    def score(labels):
        values, samples = score_matrices(participants_cm(reference, labels, visible, classes), draws)
        return values["macro_f1"], samples

    rows = []
    for model in MODELS:
        clean_labels = sweeps["sample_gap", 0, model]
        if not np.array_equal(rotation[0, model], clean_labels):
            raise ValueError(f"Rotation and outage sweeps use different clean labels for {dataset}/{model}.")
        clean, clean_draws = score(clean_labels)
        outage, outage_draws = score(sweeps["rotation_with_outage", 0, model])
        metadata = info_for(dataset, "clean", model, [42], reference)
        for level in SWEEPS["rotation_with_outage"][1:]:
            rotated, rotation_draws = score(rotation[level, model])
            joint, joint_draws = score(sweeps["rotation_with_outage", level, model])
            low, high = interval(100 * (joint_draws - rotation_draws - outage_draws + clean_draws))
            rows.append({**metadata, "rotation_degrees": level, "joint_macro_f1": joint, "rotation_macro_f1": rotated,
                         "outage_macro_f1": outage, "clean_macro_f1": clean,
                         "interaction_pp": 100 * (joint - rotated - outage + clean), "low_pp": low, "high_pp": high})
    return rows


def sweep_results(groups, root, draws):
    estimates, effects, class_scores, interactions = [], [], [], []
    for dataset in DATA_NAMES:
        reference = groups[dataset, "clean", PRIMARY, 42]
        raw = sweep_predictions(root, dataset, reference)
        visible = np.ones(len(reference.actual), dtype=bool)
        labels = CLASSES[dataset]
        classes = len(labels)
        clean_matrices = {model: participants_cm(reference, raw["sample_gap", 0, model], visible, classes)
                          for model in MODELS}
        clean_scores = {model: score_matrices(matrix, draws) for model, matrix in clean_matrices.items()}
        clean_activity = {model: activity_scores(matrix.sum(0), labels) for model, matrix in clean_matrices.items()}
        rotation = stress_predictions(root, dataset, "clean", reference)
        if not rotation:
            raise ValueError(f"Missing rotation-only controls to compare with joint stress on {dataset}.")
        interactions.extend(rotation_outage_effects(dataset, reference, rotation, raw, draws))
        for family, levels in SWEEPS.items():
            for level in levels:
                ref = raw[family, level, PRIMARY]
                geom, geometry_draws = score_matrices(participants_cm(reference, ref, visible, classes), draws)
                for model in MODELS:
                    same = raw[family, level, model]
                    if level == 0 and family != "rotation_with_outage":
                        clean = groups[dataset, "clean", model, 42].by_condition["clean"]
                        if not np.array_equal(same, clean):
                            raise ValueError(f"Zero-stress replay differs for {dataset}/{model}/{family}")
                    matrix = participants_cm(reference, same, visible, classes)
                    values, samples = score_matrices(matrix, draws)
                    low, high = interval(samples)
                    clean_values, clean_draws = clean_scores[model]
                    decline_low, decline_high = interval(100 * (samples - clean_draws))
                    meta = info_for(dataset, "clean", model, [42], reference)
                    dose = level / 100 if family == "gaussian_noise" else level
                    estimates.append({**meta, "family": family, "level": dose, **values, "low": low, "high": high,
                                      "clean_macro_f1": clean_values["macro_f1"],
                                      "change_from_clean_pp": 100 * (values["macro_f1"] - clean_values["macro_f1"]),
                                      "change_low_pp": decline_low, "change_high_pp": decline_high})
                    if level == levels[-1]:
                        for row, clean in zip(activity_scores(matrix.sum(0), labels), clean_activity[model], strict=True):
                            delta = None if row["f1"] is None or clean["f1"] is None else 100 * (row["f1"] - clean["f1"])
                            class_scores.append({**meta, "family": family, "level": dose, **row,
                                                 "clean_f1": clean["f1"], "f1_change_pp": delta})
                    if model != PRIMARY:
                        low_pp, high_pp = interval(100 * (geometry_draws - samples))
                        effects.append({**meta, "family": family, "level": dose,
                                        "delta_pp": 100 * (geom["macro_f1"] - values["macro_f1"]),
                                        "low_pp": low_pp, "high_pp": high_pp})
    write_table(root, "Robustness/Stress_Sweeps.csv", estimates)
    write_table(root, "Robustness/Stress_Paired_Effects.csv", effects)
    write_table(root, "Robustness/Rotation_Outage_Interaction.csv", interactions)
    write_table(root, "Diagnostics/Stress_Per_Class.csv", class_scores)
    return estimates


def confidence_scores(confidence, correct, count=10):
    """Equal-width top-label confidence calibration using the stored quantized confidence."""
    which = np.minimum((confidence * count).astype(int), count - 1)
    bins = []
    for idx in range(count):
        active = which == idx
        support = int(active.sum())
        if support:
            bins.append({"bin_lower": idx/count, "bin_upper": (idx+1)/count, "windows": support,
                         "accuracy": float(correct[active].mean()), "confidence": float(confidence[active].mean())})
    calibration = sum(b["windows"] * abs(b["accuracy"] - b["confidence"]) for b in bins) / len(correct)
    summary = {"windows": len(correct), "accuracy": float(correct.mean()), "mean_confidence": float(confidence.mean()),
               "top_label_ece": float(calibration), "top_label_brier": float(np.square(confidence - correct).mean())}
    return summary, bins


def risk_at_thresholds(confidence, correct):
    rows = []
    for threshold in CONFIDENCE_LEVELS:
        retained = confidence >= threshold
        support = int(retained.sum())
        rows.append({"threshold": threshold, "accepted_windows": support, "coverage": support / len(correct),
                     "error_rate_accepted": float(1 - correct[retained].mean()) if support else None})
    return rows


def confidence_masks(data, regime):
    if regime == "clean":
        return {"clean": np.ones(len(data.actual), dtype=bool)}
    # Slow and fast share the same nonwalking windows. Count each window once for the union.
    return {**pace_masks(data.pace), "unseen_pace_union": data.pace != 1}


def agreement_scores(actual, first, second):
    a, b = first == actual, second == actual
    return {"windows": len(actual), "same_prediction": float((first == second).mean()),
            "both_correct": float((a & b).mean()), "one_correct": float((a ^ b).mean()),
            "neither_correct": float((~a & ~b).mean())}


def seed_agreement(groups, root):
    rows = []
    for dataset, regime in TASKS:
        seeds = matched_seed_ids(groups, dataset, regime)
        for model in MODELS:
            for first, second in combinations(seeds, 2):
                a, b = (groups[dataset, regime, model, seed] for seed in (first, second))
                meta = info_for(dataset, regime, model, [first, second], a)
                for name, mask in confidence_masks(a, regime).items():
                    scores = agreement_scores(a.actual[mask], a.by_condition["clean"][mask], b.by_condition["clean"][mask])
                    rows.append({**meta, "condition": name, **scores})
    write_table(root, "Diagnostics/Seed_Agreement.csv", rows)
    return rows


def calibration_results(groups, root):
    summary, reliability, selective = [], [], []
    for dataset, regime in TASKS:
        for seed in matched_seed_ids(groups, dataset, regime):
            for model in MODELS:
                data = groups[dataset, regime, model, seed]
                files = [root / "Runs" / dataset / regime / model / f"fold{f:02d}_seed{seed}" / "predictions.npz"
                         for f in range(FOLDS[dataset])]
                pieces = []
                for path in files:
                    with np.load(path, allow_pickle=False) as raw:
                        clean = raw["conditions"].tolist().index("clean")
                        pieces.append(raw["confidence_u8"][clean])
                confidence = np.concatenate(pieces).astype(float) / 255
                correct = data.by_condition["clean"] == data.actual
                for condition, mask in confidence_masks(data, regime).items():
                    meta = {**info_for(dataset, regime, model, [seed], data), "condition": condition}
                    diagnostic, bins = confidence_scores(confidence[mask], correct[mask])
                    summary.append({**meta, **diagnostic})
                    reliability.extend({**meta, **row} for row in bins)
                    selective.extend({**meta, **row} for row in risk_at_thresholds(confidence[mask], correct[mask]))
    for filename, rows in (
        ("Diagnostics/Calibration.csv", summary), ("Diagnostics/Reliability_Bins.csv", reliability),
        ("Diagnostics/Risk_Coverage.csv", selective),
    ):
        write_table(root, filename, rows)
    return summary, selective


def make_expanded_results(root, draws=1500):
    groups, _, _, _ = collect(root)
    benchmark = matched_results(groups, root, draws)
    stress = sweep_results(groups, root, draws)
    calibration, selective = calibration_results(groups, root)
    agreement = seed_agreement(groups, root)
    audit = {"matched_benchmark_rows": len(benchmark), "extra_stress_rows": len(stress),
             "calibration_rows": len(calibration), "risk_coverage_rows": len(selective),
             "seed_agreement_rows": len(agreement), "bootstrap_draws": draws}
    write_json(root / "Execution" / "Expanded_Audit.json", audit)
    print(json.dumps(audit, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("Results"))
    p.add_argument("--draws", type=int, default=1500)
    args = p.parse_args()
    if args.draws < 1:
        p.error("draws must be positive")
    make_expanded_results(args.results, args.draws)


if __name__ == "__main__":
    main()
