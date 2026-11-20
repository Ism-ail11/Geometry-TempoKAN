import os
from pathlib import Path
import unittest

import numpy as np
import torch

from Datasets.prepare import Windows, load_prepared
from Datasets.protocol import Split, context_neighbors, normalize_split, pace_masks, restrict_pace, subject_splits
from Experiments.registry import FOLDS, TASKS
from Experiments.stress import rotated_test


class DatasetTests(unittest.TestCase):
    @staticmethod
    def example():
        n = 12
        signal = np.arange(n, dtype=np.float32)[:, None, None] * np.ones((n, 128, 6), dtype=np.float32)
        return Windows(signal, np.arange(n) % 4, np.repeat(np.arange(4), 3), np.tile([-1, 1, 2], 4),
                       np.repeat([7, 8, 9, 10], 3), np.tile([False], n))

    def test_normalization_uses_training_only(self):
        data = self.example()
        split = Split(np.array([0, 2]), np.array([6]), np.array([11]), 0)
        parts, mean, std = normalize_split(data, split)
        np.testing.assert_array_equal(mean, np.ones(6))
        np.testing.assert_array_equal(std, np.ones(6))
        np.testing.assert_array_equal(parts[2], np.full((1,128,6), 10))

    def test_unseen_pace_excludes_slow_and_fast_from_training_and_validation(self):
        data = self.example()
        full = Split(np.arange(6), np.arange(6,9), np.arange(9,12), 0)
        fold = restrict_pace(full, data, "pace")
        self.assertTrue(set(data.pace[fold.train]) <= {-1, 1})
        self.assertTrue(set(data.pace[fold.val]) <= {-1, 1})
        np.testing.assert_array_equal(full.test, fold.test)

    def test_pace_conditions_share_exactly_the_same_nonwalking_examples(self):
        labels = np.array([-1, 0, 1, 2, -1])
        masks = pace_masks(labels)
        for key, code in (("pace_slow",0),("pace_medium",1),("pace_fast",2)):
            np.testing.assert_array_equal(masks[key], (labels == -1) | (labels == code))

    def test_gait_neighbors_stay_in_recording_and_index_subset(self):
        data = self.example()
        neighbors = context_neighbors(data.session, np.array([0, 1, 3, 4, 5, 9, 11]))
        expected = [[-1,1],[0,-1],[-1,3],[2,4],[3,-1],[-1,-1],[-1,-1]]
        np.testing.assert_array_equal(neighbors, expected)

    def test_rotation_is_reproducible_per_person(self):
        raw = torch.arange(768, dtype=torch.float32).reshape(1,128,6).repeat(3,1,1) / 130
        out = rotated_test(raw, np.array([3, 5, 3]), "selfback", 30)
        torch.testing.assert_close(out[0], out[2])
        self.assertGreater(float((out[0]-out[1]).abs().mean()), 0.1)

    def test_released_fold_participants_and_window_counts(self):
        cache = Path(os.environ.get("GEOMETRY_TEST_DATA","Datasets/Cache"))
        if not all((cache/"prepared"/f"{ds}.npz").is_file() for ds in FOLDS):
            self.skipTest("Prepare the three datasets to exercise the real fold checks.")
        lengths = {"uci_har":10299,"pamap2":14904,"selfback":35623}
        for ds in FOLDS:
            data = load_prepared(ds,cache)
            self.assertEqual(len(data.x), lengths[ds])
            folds = list(subject_splits(ds,data))
            self.assertEqual(len(folds),FOLDS[ds])
            for regime in (r for d,r in TASKS if ds==d):
                observed = set()
                for fold in folds:
                    s = restrict_pace(fold,data,regime)
                    a,b,c=(set(data.subject[indices]) for indices in (s.train,s.val,s.test))
                    self.assertFalse(a&b or a&c or b&c or c&observed)
                    observed.update(c)
                self.assertEqual(len(observed),{"uci_har":9,"pamap2":8,"selfback":33}[ds])
