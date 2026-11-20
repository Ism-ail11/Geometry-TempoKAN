import json
from pathlib import Path
import unittest

import numpy as np
import torch

from Experiments.analyze import Predictions, check_aligned, endpoint, macro_f1_array, score_matrices
from Experiments.registry import BASELINES, FOLDS, MODELS, TASKS, make_model
from Experiments.run import make_run, validate_plan
from Experiments.train import Run


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_baselines_accept_both_sensor_layouts_and_train(self):
        for model in BASELINES:
            for channels, count in ((6,7),(18,12)):
                run = make_run("selfback","clean",model,0)
                obj = make_model(model,channels,count,run.width)
                scores = obj(torch.randn(3,128,channels))
                self.assertEqual(scores.shape,(3,count))
                scores.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in obj.parameters()))

    def test_plan_has_every_matched_seed_model_fold(self):
        rows = [Run(**row) for row in json.loads(Path("Experiments/Plans/All.json").read_text())]
        validate_plan(rows)
        for ds, regime in TASKS:
            for model in MODELS:
                for seed in (42, 77):
                    actual = [r.fold for r in rows if (r.dataset, r.regime, r.model, r.seed) == (ds, regime, model, seed)]
                    self.assertEqual(sorted(actual), list(range(FOLDS[ds])))

    def test_multiple_copies_of_a_fit_are_rejected(self):
        row = make_run("uci_har","clean","geometry_kan",0)
        with self.assertRaises(ValueError):
            validate_plan([row,row])

    def test_macro_f1_uses_observed_classes(self):
        matrix = np.array([[4,1,0],[1,4,0],[0,0,0]])
        self.assertAlmostEqual(float(macro_f1_array(matrix)),0.8)

    def test_bootstrap_is_deterministic_and_paired(self):
        a = np.array([[[9,1],[2,8]],[[5,5],[1,9]],[[6,4],[3,7]]])
        b = np.array([[[8,2],[3,7]],[[4,6],[2,8]],[[5,5],[4,6]]])
        value_a,sample_a=score_matrices(a,50)
        value_b,sample_b=score_matrices(b,50)
        np.testing.assert_array_equal(sample_a,score_matrices(a,50)[1])
        self.assertGreater(value_a["macro_f1"],value_b["macro_f1"])
        self.assertTrue((sample_a > sample_b).all())

    def test_pace_endpoint_uses_both_unseen_speeds(self):
        data = Predictions(np.arange(4),np.array([0,0,1,1]),np.array([0,1,0,1]),np.array([0,-1,2,-1]),
                           {"clean":np.array([0,1,1,1])})
        result,_ = endpoint(data,"pace",2,50)
        self.assertEqual(result["macro_f1"],0.7)

    def test_alignment_detects_changed_test_windows(self):
        a = Predictions(np.array([0,1]),np.array([0,1]),np.array([0,1]),np.array([-1,-1]),{})
        b = Predictions(np.array([0,2]),a.subject,a.actual,a.pace,{})
        with self.assertRaises(ValueError):
            check_aligned(a,b,"fixture")
