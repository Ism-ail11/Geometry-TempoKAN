"""Gravity-referenced harmonic motion and posture-specific spline corrections."""

import torch
from torch import nn
from torch.nn import functional as F

from .harmonics import HarmonicMotionReadout
from .layers import MatchedDenseConvBlock, SplineLinear, kinematic_features, mean_patches, moment_pool


def conditional_correction(elapsed, correction, gate, moving):
    """Change activity odds within the moving group, preserving its probability mass."""
    candidate = elapsed + gate * correction
    floor = torch.finfo(elapsed.dtype).min
    left = elapsed.masked_fill(~moving, floor).logsumexp(-1, keepdim=True)
    right = candidate.masked_fill(~moving, floor).logsumexp(-1, keepdim=True)
    return torch.where(moving, candidate + left - right, elapsed)


def motion_geometry(centered, gravity, radial, present, *, aligned=True):
    """Lagged horizontal dot and signed area features within each triad."""
    if aligned:
        projected = (centered * gravity[:, None]).sum(-1, keepdim=True)
        plane = centered - projected * gravity[:, None]
        normal = gravity
    else:
        normal = torch.zeros_like(gravity)
        normal[..., 2] = 1
        plane = centered * centered.new_tensor([1, 1, 0])
    denominator = plane.square().sum(-1).mean(1, keepdim=True) + 0.0125 ** 2
    aligned_terms = []
    for lag in (3, 10):
        previous = torch.cat((plane[:, :1].expand(-1, lag, -1, -1), plane[:, :-lag]), dim=1)
        alignment = (previous * plane).sum(-1)
        aligned_terms.append(torch.tanh(alignment / (2 * denominator)))
        if lag == 3:
            area = (torch.linalg.cross(previous, plane, dim=-1) * normal[:, None]).sum(-1)
            aligned_terms.append(torch.tanh(area / denominator))
    return torch.cat((radial, *(term * present[:, None] for term in aligned_terms)), dim=-1)


def posture_geometry(center, centered, gravity, present, scale):
    """Absolute gravity bearing with per-axis micro-motion and covariance."""
    amplitude = (centered.square().mean(dim=1) + 1e-10).sqrt()
    micro = (centered.diff(dim=1).square().mean(dim=1) + 1e-10).sqrt()
    # Stabilized covariance ratios preserve which axes move together.
    unit = centered / (amplitude[:, None] + 0.015)
    a, b, c = unit.unbind(-1)
    rho = torch.stack(((a * b).mean(1), (a * c).mean(1), (b * c).mean(1)), dim=-1)
    # Two halves distinguish drift from a stable posture.
    first, second = centered.chunk(2, dim=1)
    drift = torch.tanh((first.mean(1) - second.mean(1)) * 8)
    feature = torch.cat(((center / scale[:, None]).tanh(), gravity,
                         (12 * amplitude).log1p(), (35 * micro).log1p(), rho, drift), dim=-1)
    return (feature * present[..., None]).flatten(1)


class GeometryKAN(nn.Module):
    """Factorized specialist heads on the same elapsed-time backbone."""

    def __init__(self, channels, classes, width=24, stride=4, *, aligned=True, spline=True, posture=True):
        groups = channels // 3
        harmonic = HarmonicMotionReadout(5 * groups, max(4, width // 2), classes, spline=spline)
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.stride = stride
        self.register_buffer("mean_signal", torch.zeros(channels))
        self.register_buffer("std_signal", torch.ones(channels))
        self.register_buffer("gravity_parent", torch.arange(groups))
        self.register_buffer("signal_scale", torch.ones(groups))
        self.register_buffer("accel_weight", torch.ones(groups) / groups)
        self.register_buffer("moving", torch.ones(classes, dtype=torch.bool))
        self.specialist = harmonic
        self.elapsed_stem = nn.Sequential(nn.Linear(channels * stride + 2 * groups, width),
                                          nn.LayerNorm(width), nn.SiLU())
        self.elapsed_body = self.encoder(width, (1, 2, 4))
        pose_count = 3 * channels + 15 * groups
        self.posture = nn.Sequential(nn.LayerNorm(pose_count), nn.Linear(pose_count, 2 * width),
                                     nn.SiLU(), nn.Linear(2 * width, width))
        self.base = nn.Sequential(nn.LayerNorm(4 * width), nn.Linear(4 * width, 2 * width),
                                  nn.SiLU(), nn.Dropout(0.12), nn.Linear(2 * width, classes))
        self.reliability = nn.Linear(4, 1)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)
        self.aligned = aligned
        self.has_posture = posture
        if posture:
            inputs = 18 * groups
            if spline:
                projection = SplineLinear(inputs, width)
            else:
                middle = round(9 * inputs * width / (inputs + 1 + width))
                projection = nn.Sequential(nn.Linear(inputs, middle), nn.SiLU(), nn.Linear(middle, width))
            self.pose_specialist = nn.Sequential(projection, nn.LayerNorm(width), nn.SiLU(),
                                                 nn.Dropout(0.08), nn.Linear(width, classes))

    @staticmethod
    def encoder(width, dilations):
        return nn.Sequential(*(MatchedDenseConvBlock(width, dilation) for dilation in dilations),
                             nn.LayerNorm(width))

    @torch.no_grad()
    def calibrate(self, mean, std, dataset, classes):
        self.mean_signal.copy_(torch.as_tensor(mean))
        self.std_signal.copy_(torch.as_tensor(std))
        both = dataset != "selfback"
        group = torch.arange(self.groups)
        accel = (group % 2 == 0) if both else torch.ones_like(group, dtype=torch.bool)
        self.accel_weight.copy_(accel.float() / accel.sum())
        self.gravity_parent.copy_((group // 2) * 2 if both else group)
        scale = torch.as_tensor(std).reshape(-1, 3).square().mean(-1).sqrt().clamp_min(1e-4)
        scale[accel] = 9.80665 if dataset == "pamap2" else 1.0
        self.signal_scale.copy_(scale)
        self.moving.copy_(torch.tensor([label.lower() not in {"sitting", "standing", "lying"} for label in classes]))

    def sensor_frame(self, x):
        n, length, _ = x.shape
        raw = (x * self.std_signal + self.mean_signal).reshape(n, length, self.groups, 3)
        present = x.reshape_as(raw).abs().amax(dim=(1, 3)) > 1e-7
        center = raw.mean(dim=1)
        centered = (raw - center[:, None]) / self.signal_scale[None, None, :, None]
        gravity = F.normalize(center[:, self.gravity_parent], dim=-1, eps=1e-5)
        gravity = gravity * present[:, self.gravity_parent, None]
        return center, centered, gravity, present

    def signal_views(self, center, centered, gravity, present):
        vertical = (centered * gravity[:, None]).sum(-1)
        lateral = (centered - vertical[..., None] * gravity[:, None]).square().sum(-1).add(1e-8).sqrt()
        radial = torch.cat((vertical, lateral), dim=-1)
        dyn = self.motion_channels(centered, gravity, radial, present)
        magnitude = (vertical.square() + lateral.square()).mean(dim=1).add(1e-8).sqrt()
        active_rms = (magnitude * self.accel_weight).sum(-1, keepdim=True)
        reliab = torch.cat((active_rms.log1p(), magnitude.amax(-1, keepdim=True).log1p(),
                            present.float().mean(-1, keepdim=True),
                            radial.diff(dim=1).abs().mean(dim=(1, 2)).view(len(centered), 1).log1p()), dim=1)
        motion = active_rms.square() / (active_rms.square() + (0.055 ** 2))
        gate = motion * (0.18 + 0.72 * self.reliability(reliab).sigmoid())
        pose = torch.cat((gravity.flatten(1), (center / self.signal_scale[:, None]).flatten(1),
                          (magnitude * present)), dim=-1)
        return dyn, radial, gate, pose

    def signal_components(self, x):
        return self.signal_views(*self.sensor_frame(x))

    def elapsed_logits(self, x, radial, pose):
        patches = x.reshape(len(x), -1, self.stride * self.channels)
        elapsed = self.elapsed_stem(torch.cat((patches, mean_patches(radial, self.stride)), dim=-1))
        elapsed = moment_pool(self.elapsed_body(elapsed))
        pose = self.posture(torch.cat((pose, kinematic_features(x)), dim=1))
        return self.base(torch.cat((elapsed, pose), dim=1))

    def motion_logits(self, x):
        dynamics, _, _, _ = self.signal_components(x)
        return self.specialist(dynamics)

    def forward(self, x):
        return self.forward_views(x)[0]

    def motion_channels(self, centered, gravity, radial, present):
        return motion_geometry(centered, gravity, radial, present, aligned=self.aligned)

    def forward_views(self, x, *, return_posture=False):
        frame = self.sensor_frame(x)
        dynamics, radial, gate, pose = self.signal_views(*frame)
        base = self.elapsed_logits(x, radial, pose)
        motion = self.specialist(dynamics)
        fused = conditional_correction(base, motion, gate, self.moving)
        correction = None
        if self.has_posture:
            correction = self.pose_specialist(posture_geometry(*frame, self.signal_scale))
            stationary_gate = 0.8 - 0.45 * gate
            fused = conditional_correction(fused, correction, stationary_gate, ~self.moving)
        if return_posture:
            return fused, base, motion, correction
        return fused, base, motion


def rotate_measurements(raw, axes, angles, group):
    """Apply one proper rotation per placement to co-located inertial triads."""
    n, length, channels = raw.shape
    v = raw.reshape(n, length, channels // group, group // 3, 3)
    axis = F.normalize(axes, dim=-1, eps=1e-7)[:, None, :, None]
    sine = angles.sin()[:, None, :, None, None]
    cosine = angles.cos()[:, None, :, None, None]
    rotated = v * cosine + torch.linalg.cross(axis, v, dim=-1) * sine
    rotated = rotated + (v * axis).sum(-1, keepdim=True) * axis * (1 - cosine)
    return rotated.reshape_as(raw)


def augment_mount(x, mean, std, dataset, generator):
    group = 3 if dataset == "selfback" else 6
    shape = (len(x), x.shape[-1] // group)
    axes = torch.randn((*shape, 3), generator=generator, device=x.device, dtype=x.dtype)
    amount = torch.rand(shape, generator=generator, device=x.device, dtype=x.dtype).mul(2).sub(1)
    active = torch.rand((len(x), 1), generator=generator, device=x.device) < 0.7
    angles = amount * active * 0.27
    return (rotate_measurements(x * std + mean, axes, angles, group) - mean) / std
