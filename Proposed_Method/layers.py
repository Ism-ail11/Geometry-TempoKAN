"""Spline edges and elapsed-time layers for Geometry TempoKAN."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .native import CPUTempoCubic, KERNEL


def cubic_basis(x, centers):
    """Cubic cardinal B-splines with a tail handled by the separate linear branch."""
    coordinate = x.clamp(-3.0, 3.0).unsqueeze(-1) * ((len(centers) - 3) / 6)
    distance = (coordinate - centers).abs()
    near = (1 - distance).clamp_min(0)
    far = (2 - distance).clamp_min(0)
    return (far * far * far - 4 * near * near * near) / 6


def spline_features(x, centers, with_base=False):
    if KERNEL is not None and x.device.type == "cpu" and x.dtype == torch.float32:
        return CPUTempoCubic.apply(x, len(centers) - 3, with_base)
    basis = cubic_basis(x, centers)
    return torch.cat((x.unsqueeze(-1), basis), dim=-1) if with_base else basis


class SplineLinear(nn.Module):
    """Sum of independent, learned univariate edge functions."""

    def __init__(self, inputs, outputs, grid=5):
        super().__init__()
        self.register_buffer("centers", torch.arange(grid + 3, dtype=torch.float32) - 1 - grid / 2)
        self.base = nn.Linear(inputs, outputs)
        self.spline = nn.Linear(inputs * (grid + 3), outputs, bias=False)
        nn.init.normal_(self.spline.weight, std=0.035 / math.sqrt(inputs))

    def forward(self, x):
        return self.base(x) + self.spline(spline_features(x, self.centers).flatten(-2))


class MatchedDenseConvBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Conv1d(width, width, 3, dilation=dilation, padding=dilation)

    def forward(self, x):
        z = self.norm(x).transpose(1, 2).contiguous()
        return x + F.silu(self.linear(z)).transpose(1, 2)


def mean_patches(x, stride):
    if x.shape[1] % stride:
        raise ValueError("The window length must be divisible by the token stride")
    return x.reshape(x.shape[0], -1, stride, x.shape[-1]).mean(dim=2) if stride > 1 else x


def moment_pool(x):
    mean = x.mean(dim=1)
    deviation = (x.var(dim=1, unbiased=False) + 1e-5).sqrt()
    return torch.cat((mean, deviation, x.amax(dim=1)), dim=-1)


KINEMATIC_BANDS = ((0.35, 1.15), (1.15, 2.35), (2.35, 3.9), (3.9, 6.25), (6.25, 10), (10, 20))


def kinematic_features(x):
    """Axis-wise pose statistics and spectra of standardized triad magnitudes."""
    batch, length, channels = x.shape
    triad = x.reshape(batch, length, channels // 3, 3)
    spatial = triad.square().sum(dim=-1).add(1e-8).sqrt()
    varied = spatial - spatial.mean(dim=1, keepdim=True)
    taper = torch.hann_window(length, dtype=x.dtype, device=x.device)
    frequency = torch.fft.rfftfreq(length, d=1 / 50, device=x.device)
    power = torch.fft.rfft(varied * taper.view(1, -1, 1), dim=1).abs().square() / length
    band = torch.stack([((frequency >= low) & (frequency < high)) for low, high in KINEMATIC_BANDS], dim=1)
    energy = (power.transpose(1, 2) @ band.to(x.dtype)).log1p().flatten(1)
    triad_time = torch.cat((spatial.mean(dim=1), varied.square().mean(dim=1).add(1e-8).sqrt()), dim=1)
    activity = x.diff(dim=1).abs().mean(dim=1)
    return torch.cat((x.mean(dim=1), x.std(dim=1, unbiased=False), activity, triad_time, energy), dim=1)
