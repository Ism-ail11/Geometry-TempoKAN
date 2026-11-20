import unittest

import numpy as np
import torch

from Experiments.analyze import Predictions, macro_f1_array
from Experiments.expanded import (activity_scores, agreement_scores, confidence_masks, confidence_scores, matched_seed_ids,
                                  pooled_matrices, risk_at_thresholds)
from Experiments.registry import MODELS, PRIMARY
from Experiments.stress_sweeps import burst_gap, outage_mask, perturbation


class SensorSweepTests(unittest.TestCase):
    def test_gap_interpolates_the_internal_samples(self):
        x = torch.arange(7).square().float().reshape(1, 7, 1)
        result = burst_gap(x, 2, torch.tensor([0.5]))
        torch.testing.assert_close(result.flatten(), torch.tensor([0., 1., 4., 11., 18., 25., 36.]))

    def test_gap_zero_preserves_all_channels_and_windows(self):
        windows = torch.arange(4 * 128 * 6).float().reshape(4, 128, 6)
        self.assertTrue(torch.equal(burst_gap(windows, 0, torch.zeros(4)), windows))

    def test_gap_uses_exactly_the_requested_length_at_extreme_offsets(self):
        curve = torch.arange(128).square().float().reshape(1, 128, 1).repeat(2, 1, 3)
        changed = burst_gap(curve, 32, torch.tensor([0., 1.]))
        differences = changed != curve
        torch.testing.assert_close(differences.sum(1), torch.full((2, 3), 32))
        self.assertFalse(differences[:, [0, 127]].any())

    def test_outage_removes_one_contiguous_sensor_group_per_person(self):
        people = np.array([21, 16, 21, 99])
        for dataset, channels, width in (("pamap2", 18, 6), ("selfback", 6, 3), ("uci_har", 6, 3)):
            with self.subTest(dataset=dataset):
                mask = outage_mask(people, channels, dataset).reshape(len(people), -1, width)
                torch.testing.assert_close(mask.sum((1, 2)), torch.full((len(people),), width))
                self.assertTrue(torch.equal(mask.any(-1), mask.all(-1)))
                self.assertTrue(torch.equal(mask[0], mask[2]))
                shuffled = outage_mask(people[::-1], channels, dataset).reshape(mask.shape)
                self.assertTrue(torch.equal(mask.flip(0), shuffled))

    def test_gaussian_levels_reuse_the_same_draw_and_training_scale(self):
        x = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
        noise = torch.tensor([[[1., 1.], [-1., -1.], [2., 2.], [-2., -2.]]])
        std = torch.tensor([1., 3.])
        zero, low, high = [
            perturbation(x, np.array([3]), "selfback", "gaussian_noise", level,
                         None, std, None, noise, None) for level in (0, 10, 20)
        ]
        torch.testing.assert_close(zero, x)
        torch.testing.assert_close(low - x, 0.1 * std * noise)
        torch.testing.assert_close(high - x, 2 * (low - x))

    def test_rotation_and_outage_keep_the_masked_values_at_training_means(self):
        x = torch.arange(1., 25.).reshape(1, 4, 6)
        missing = torch.tensor([[[True, True, True, False, False, False]]])
        mean = torch.arange(-6., 0.)
        at_zero, at_angle = [
            perturbation(x, np.array([3]), "selfback", "rotation_with_outage", angle,
                         mean, None, None, None, missing) for angle in (0, 60)
        ]
        expected = mean[:3].expand(1, 4, 3)
        torch.testing.assert_close(at_zero[..., :3], expected)
        torch.testing.assert_close(at_angle[..., :3], expected)
        torch.testing.assert_close(at_zero[..., 3:], x[..., 3:])
        self.assertFalse(torch.allclose(at_angle[..., 3:], x[..., 3:]))
        torch.testing.assert_close(at_angle[..., 3:].norm(dim=-1), x[..., 3:].norm(dim=-1))


class EvidenceTests(unittest.TestCase):
    def test_activity_metrics_distinguish_zero_recall_from_undefined_precision(self):
        result = activity_scores(np.array([[1.5, 0], [2.5, 0]]), ("Walking", "Jogging"))
        self.assertEqual(result[1]["recall"], 0)
        self.assertEqual(result[1]["f1"], 0)
        self.assertIsNone(result[1]["precision"])
        self.assertAlmostEqual(result[0]["precision"], 1.5 / 4)

    def test_seed_agreement_distinguishes_correctness_from_prediction_matches(self):
        scores = agreement_scores(np.array([0, 0, 0, 0]), np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1]))
        self.assertEqual(scores, {"windows": 4, "same_prediction": 0.5, "both_correct": 0.25,
                                  "one_correct": 0.25, "neither_correct": 0.5})

    def test_matched_seed_set_excludes_runs_missing_comparators(self):
        groups = {("uci_har", "clean", model, seed): None for model in MODELS for seed in (42, 77)}
        groups["uci_har", "clean", PRIMARY, 123] = None
        groups["selfback", "clean", PRIMARY, 17] = None
        self.assertEqual(matched_seed_ids(groups, "uci_har", "clean"), [42, 77])

    def test_participant_confusions_are_pooled_before_macro_f1(self):
        actual = np.array([0, 0, 0, 1, 1, 1])
        keys = (np.arange(6), np.array([1, 1, 1, 1, 2, 2]), actual, np.full(6, -1))
        examples = [Predictions(*keys, {"clean": p}) for p in (np.zeros(6, dtype=int), actual)]
        observed = pooled_matrices(examples, "clean", 2)
        np.testing.assert_allclose(observed, np.array([[[3, 0], [.5, .5]], [[0, 0], [1, 1]]]))
        self.assertAlmostEqual(float(macro_f1_array(observed.sum(0))), (0.8 + 2 / 3) / 2)

    def test_top_label_calibration_uses_window_weights_and_includes_confidence_one(self):
        confidence = np.array([0.55, 0.58, 0.90, 1.0])
        correct = np.array([True, False, False, True])
        values, bins = confidence_scores(confidence, correct)
        self.assertEqual([b["windows"] for b in bins], [2, 2])
        self.assertEqual(bins[-1]["bin_upper"], 1)
        self.assertAlmostEqual(values["top_label_ece"], 0.2575)
        self.assertAlmostEqual(values["top_label_brier"], 0.337225)
        self.assertAlmostEqual(values["accuracy"], 0.5)

    def test_coverage_is_monotone_and_empty_acceptance_has_no_error_estimate(self):
        rows = risk_at_thresholds(np.array([.50, .59, .75]), np.array([True, False, True]))
        self.assertEqual([r["accepted_windows"] for r in rows], [3, 1, 1, 0, 0, 0])
        self.assertAlmostEqual(rows[0]["error_rate_accepted"], 1 / 3)
        self.assertEqual(rows[1]["error_rate_accepted"], 0.0)
        self.assertIsNone(rows[-1]["error_rate_accepted"])

    def test_unseen_pace_confidence_reuses_nonwalking_only_once(self):
        item = Predictions(np.arange(5), np.array([1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1]),
                           np.array([0, 1, 2, -1, -1]), {})
        masks = confidence_masks(item, "pace")
        self.assertEqual(masks["pace_slow"].sum(), 3)
        self.assertEqual(masks["pace_fast"].sum(), 3)
        np.testing.assert_array_equal(masks["unseen_pace_union"], np.array([True, False, True, True, True]))
