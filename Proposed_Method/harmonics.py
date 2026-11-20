"""Fixed-frequency harmonic features for the moving-activity classifier."""

import math

import torch
from torch import nn

from .layers import SplineLinear


def harmonic_descriptors(real, imag):
    """Return power and two phase-invariant, amplitude-stabilized products."""
    power = real.square() + imag.square()
    a, b, c = real.unbind(-1)
    d, e, f = imag.unbind(-1)
    p, q, r = power.unbind(-1)
    square_real, square_imag = a.square() - d.square(), 2 * a * d
    pair_real, pair_imag = a * b - d * e, a * e + b * d
    norm_square = (p.square() * q + 0.004).rsqrt()
    norm_pair = (p * q * r + 0.004).rsqrt()
    phase = torch.stack(((square_real * b + square_imag * e) * norm_square,
                         (square_imag * b - square_real * e) * norm_square,
                         (pair_real * c + pair_imag * f) * norm_pair,
                         (pair_imag * c - pair_real * f) * norm_pair), -1)
    relative_power = power / (0.35 + power.sum(-1, keepdim=True))
    return power, torch.cat((2 * relative_power, phase), -1)


class HarmonicMotionReadout(nn.Module):
    """Pool normalized harmonic shape over candidate fundamental frequencies."""

    def __init__(self, dynamics, width, classes, spline=True, length=128):
        super().__init__()
        self.length = length
        frequencies = torch.linspace(0.70, 3.10, 17)
        harmonic = frequencies[:, None] * torch.arange(1, 4)[None]
        time = (torch.arange(length // 2) * 2 + 0.5) / 50
        angle = 2 * math.pi * time[:, None, None] * harmonic
        taper = torch.hann_window(len(time), periodic=False)
        basis = torch.stack((angle.cos(), -angle.sin()), -1)
        basis = (basis * (taper / taper.sum())[:, None, None, None]).flatten(1)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.frequency_prior = nn.Parameter(torch.zeros(len(frequencies)))
        inputs = dynamics * 8
        if spline:
            self.project = SplineLinear(inputs, 2 * width)
        else:
            middle = round(9 * inputs * 2 * width / (inputs + 1 + 2 * width))
            self.project = nn.Sequential(nn.Linear(inputs, middle), nn.SiLU(), nn.Linear(middle, 2 * width))
        self.head = nn.Sequential(nn.LayerNorm(2 * width), nn.SiLU(), nn.Dropout(0.08),
                                  nn.Linear(2 * width, classes))

    def spectrum(self, dynamics):
        batch, samples, channels = dynamics.shape
        if samples != self.length:
            raise ValueError(f"Cadence harmonics require {self.length} samples, received {samples}.")
        x = dynamics.reshape(batch, samples // 2, 2, channels).mean(2).transpose(1, 2)
        centered = x - x.mean(-1, keepdim=True)
        energy = centered.square().mean(-1, keepdim=True)
        normalized = centered * (energy + 1e-5).rsqrt()
        real, imag = (normalized @ self.basis).reshape(batch, channels, -1, 3, 2).unbind(-1)
        _, features = harmonic_descriptors(real, imag)
        summary = torch.log1p(6 * (energy + 1e-8).sqrt())
        return features, summary

    def forward(self, dynamics):
        features, summary = self.spectrum(dynamics)
        weights = self.frequency_prior[None].expand(len(dynamics), -1).softmax(-1)
        pooled = (features * weights[:, None, :, None]).sum(-2)
        return self.head(self.project(torch.cat((pooled, summary), -1).flatten(1)))
