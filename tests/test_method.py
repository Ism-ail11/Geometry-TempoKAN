import unittest

import torch

from Datasets.prepare import CLASSES
from Experiments.registry import GEOMETRY, make_model, parameter_count
from Experiments.train import geometry_loss
from Proposed_Method.layers import cubic_basis, spline_features
from Proposed_Method.model import rotate_measurements


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def case(self, dataset, model="geometry_kan"):
        channels = 18 if dataset == "pamap2" else 6
        labels = CLASSES[dataset]
        rng = torch.Generator().manual_seed(217)
        x = 0.4 * torch.randn((7, 128, channels), generator=rng)
        mean = torch.randn(channels, generator=rng) * 1.5
        std = 0.7 + torch.rand(channels, generator=rng)
        obj = make_model(model, channels, len(labels))
        obj.calibrate(mean, std, dataset, labels)
        return obj, x, rng

    def test_motion_logits_are_invariant_under_colocated_proper_rotations(self):
        for dataset, unit in (("uci_har", 6), ("pamap2", 6), ("selfback", 3)):
            obj, x, rng = self.case(dataset)
            raw = x * obj.std_signal + obj.mean_signal
            units = (len(x), x.shape[-1] // unit)
            axes = torch.randn((*units, 3), generator=rng)
            angle = 4 * torch.rand(units, generator=rng) - 2
            rotated = rotate_measurements(raw, axes, angle, unit)
            torch.testing.assert_close(raw.reshape(*raw.shape[:2], -1, 3).norm(dim=-1),
                                       rotated.reshape(*raw.shape[:2], -1, 3).norm(dim=-1))
            shifted = (rotated - obj.mean_signal) / obj.std_signal
            obj.eval()
            torch.testing.assert_close(obj.signal_components(shifted)[0], obj.signal_components(x)[0],
                                       atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(obj.motion_logits(shifted), obj.motion_logits(x), atol=1e-5, rtol=1e-4)
            control, _, _ = self.case(dataset, "geometry_axes")
            self.assertGreater(float((control.signal_components(shifted)[0] -
                                      control.signal_components(x)[0]).abs().max()), 0.01)

    def test_updates_preserve_both_group_probabilities(self):
        obj, x, _ = self.case("selfback")
        output, base, _, _ = obj.forward_views(x, return_posture=True)
        for mask in (obj.moving, ~obj.moving):
            torch.testing.assert_close(output.softmax(-1)[:, mask].sum(1), base.softmax(-1)[:, mask].sum(1))

    def test_sensor_loss_preserves_finite_outputs(self):
        obj, x, _ = self.case("selfback")
        x[:, :, :3] = 0
        signal, _, _, _ = obj.signal_components(x)
        self.assertLess(float(signal[:, :, ::2].abs().max()), 1.01e-4)
        self.assertTrue(torch.isfinite(obj(x)).all())

    def test_ablation_parameters_match_expected_changes(self):
        sizes = {name: parameter_count(self.case("pamap2", name)[0]) for name in GEOMETRY}
        self.assertEqual(sizes["geometry_kan"], sizes["geometry_axes"])
        self.assertEqual(sizes["geometry_kan"], sizes["geometry_no_rotation"])
        self.assertLess(sizes["geometry_no_posture"], sizes["geometry_kan"])
        self.assertLess(abs(sizes["geometry_mlp"] / sizes["geometry_kan"] - 1), 0.001)

    def test_losses_produce_finite_gradients_for_all_geometry_controls(self):
        for name in GEOMETRY:
            obj, x, rng = self.case("pamap2", name)
            loss = geometry_loss(obj, x, torch.arange(len(x)), torch.ones(12), rng)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in obj.parameters()), name)

    def test_spline_values_and_gradients_match_torch_basis(self):
        values = torch.linspace(-2.9, 2.9, 53).reshape(1, -1).requires_grad_()
        centers = torch.arange(8).float() - 3.5
        fast = spline_features(values, centers)
        expected = cubic_basis(values, centers)
        torch.testing.assert_close(fast, expected)
        fast_grad, = torch.autograd.grad(fast.square().sum(), values)
        expected_grad, = torch.autograd.grad(expected.square().sum(), values)
        torch.testing.assert_close(fast_grad, expected_grad, atol=2e-6, rtol=2e-5)
