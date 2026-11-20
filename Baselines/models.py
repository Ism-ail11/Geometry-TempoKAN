"""Independent HAR comparators using the shared window and class interface."""

import torch
from torch import nn
from torch.nn import functional as F

from Proposed_Method.layers import MatchedDenseConvBlock, SplineLinear, moment_pool, spline_features


class SplineProjection(SplineLinear):
    """Cubic spline edges with the SiLU base used in the comparator families."""

    def forward(self, x):
        return self.base(F.silu(x)) + self.spline(spline_features(x, self.centers).flatten(-2))


class KANHAR(nn.Module):
    """Two KAN layers adapted from Alikhani's KAN-HAR to multichannel windows."""

    def __init__(self, channels, classes, width):
        super().__init__()
        self.extract = SplineProjection(128 * channels, width)
        self.classify = SplineProjection(width, classes)

    def forward(self, x):
        return self.classify(self.extract(x.flatten(1)))


class MixerBlock(nn.Module):
    def __init__(self, patches, width):
        super().__init__()
        self.norm_time = nn.LayerNorm(width)
        self.norm_channels = nn.LayerNorm(width)
        self.time = nn.Sequential(nn.Linear(patches, 2 * patches), nn.GELU(), nn.Linear(2 * patches, patches))
        self.channels = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x):
        x = x + self.time(self.norm_time(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channels(self.norm_channels(x))


class ArctanEdges(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.rate = nn.Parameter(torch.ones(outputs, inputs))
        self.scale = nn.Parameter(torch.empty(outputs, inputs))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.xavier_uniform_(self.scale)

    def forward(self, x):
        return (torch.atan(x.unsqueeze(-2) * self.rate) * self.scale).sum(dim=-1) + self.bias


class KANMLPMixer(nn.Module):
    """Spline patch embedding, MLP mixing, and an arctangent-edge classifier."""

    def __init__(self, channels, classes, width):
        super().__init__()
        self.patch = 16
        patches = 128 // self.patch
        self.embed = SplineProjection(self.patch * channels, width)
        self.mix = nn.Sequential(MixerBlock(patches, width), MixerBlock(patches, width), nn.LayerNorm(width))
        self.classify = nn.Sequential(nn.Dropout(0.12), ArctanEdges(width, classes))

    def forward(self, x):
        patches = x.reshape(len(x), x.shape[1] // self.patch, -1)
        return self.classify(self.mix(self.embed(patches)).mean(dim=1))


class ResidualKANFeatures(nn.Module):
    """Channel-wise and cross-channel KAN features following the RL-KAN FE design."""

    def __init__(self, channels, classes):
        super().__init__()
        self.patch = 16
        self.per_channel = SplineProjection(self.patch, 5)
        self.across_channels = SplineProjection(5 * channels, 10)
        self.classify = nn.Sequential(nn.Flatten(), nn.Linear(8 * (5 * channels + 10), 256),
                                      nn.ReLU(), nn.Dropout(0.12), nn.Linear(256, 128), nn.ReLU(),
                                      nn.Linear(128, classes))

    def forward(self, x):
        grouped = x.reshape(len(x), -1, self.patch, x.shape[-1]).transpose(-2, -1).contiguous()
        per_channel = self.per_channel(grouped).flatten(2)
        return self.classify(torch.cat((per_channel, self.across_channels(per_channel)), dim=-1))


class TemporalCNN(nn.Module):
    """Dilated residual temporal CNN with an elapsed-time readout."""

    def __init__(self, channels, classes, width, stride):
        super().__init__()
        self.stride = stride
        self.stem = nn.Sequential(nn.Linear(channels * stride, width), nn.LayerNorm(width), nn.SiLU())
        self.encoder = nn.Sequential(*(MatchedDenseConvBlock(width, rate) for rate in (1, 2, 4, 8)),
                                     nn.LayerNorm(width))
        self.classify = nn.Sequential(nn.LayerNorm(3 * width), nn.Linear(3 * width, width), nn.SiLU(),
                                      nn.Dropout(0.12), nn.Linear(width, classes))

    def forward(self, x):
        patches = x.reshape(len(x), -1, x.shape[-1] * self.stride)
        return self.classify(moment_pool(self.encoder(self.stem(patches))))
