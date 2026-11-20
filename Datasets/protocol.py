"""Reproducible participant-disjoint partitions and input perturbations."""

from dataclasses import dataclass

import numpy as np
import torch


SPLIT_SEED = 20261119
CONDITIONS = ("clean", "warp_mild", "warp_strong", "warp_asymmetric", "noise", "gain", "gap", "sensor_out")
PACE_CONDITIONS = ("pace_medium", "pace_slow", "pace_fast")


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    fold: int


def subject_splits(dataset, data):
    rng = np.random.default_rng(SPLIT_SEED)
    if dataset == "uci_har":
        official_training = np.unique(data.subject[~data.official_test])
        val_people = rng.permutation(official_training)[:4]
        val = np.isin(data.subject, val_people)
        yield Split(np.flatnonzero(~data.official_test & ~val), np.flatnonzero(val),
                    np.flatnonzero(data.official_test), 0)
        return
    people = np.unique(data.subject)
    if dataset == "pamap2":
        for fold, person in enumerate(people):
            val_person = people[(fold + 3) % len(people)]
            yield Split(np.flatnonzero(~np.isin(data.subject, (person, val_person))),
                        np.flatnonzero(data.subject == val_person), np.flatnonzero(data.subject == person), fold)
        return
    groups = np.array_split(rng.permutation(people), 5)
    for fold, test_people in enumerate(groups):
        val_people = groups[(fold + 1) % 5][:3]
        val = np.isin(data.subject, val_people)
        test = np.isin(data.subject, test_people)
        yield Split(np.flatnonzero(~(val | test)), np.flatnonzero(val), np.flatnonzero(test), fold)


def restrict_pace(split, data, regime):
    if regime in ("clean", "smoke"):
        return split
    if regime != "pace":
        raise ValueError(f"Unknown training regime {regime}")
    allowed = (data.pace == -1) | (data.pace == 1)
    return Split(split.train[allowed[split.train]], split.val[allowed[split.val]], split.test, split.fold)


def pace_masks(pace):
    return {name: (pace == -1) | (pace == code) for name, code in zip(PACE_CONDITIONS, (1, 0, 2))}


def normalize_split(data, split):
    training = data.x[split.train]
    mean = training.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = training.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    return [((part - mean) / std).astype(np.float32, copy=False)
            for part in (training, data.x[split.val], data.x[split.test])], mean, std


def resample_batch(x, coordinates):
    coordinates = coordinates.expand(len(x), -1)
    last = x.shape[1] - 1
    fractional = coordinates.clamp(0, 1) * last
    left = fractional.floor().long().clamp_max(last - 1)
    right = left + 1
    fraction = fractional - left
    low = x.gather(1, left.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    high = x.gather(1, right.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return torch.lerp(low, high, fraction.unsqueeze(-1))


def uniform_grid(x):
    return torch.linspace(0, 1, x.shape[1], dtype=x.dtype, device=x.device).view(1, -1)


def random_warp(x, severity, rng, active=None, higher_harmonic=0.0):
    """Smooth monotone time reparameterization, shared across channels."""
    count = x.shape[0]
    base = uniform_grid(x)
    phase = torch.rand(count, 1, device=x.device, generator=rng) * (2 * torch.pi)
    direction = torch.rand(count, 1, device=x.device, generator=rng).mul(2).sub(1).sign()
    a = severity * direction
    if active is not None:
        a = a * active
    angle = 2 * torch.pi * base
    allocation = base + a * (torch.sin(angle + phase) - torch.sin(phase)) / (2 * torch.pi)
    if higher_harmonic:
        allocation = allocation + higher_harmonic * a * torch.sin(2 * angle) / (4 * torch.pi)
    return resample_batch(x, allocation)


def augment_training(x, rng, policy="shared", dataset=None):
    if policy == "none":
        return x
    count = len(x)
    unit = torch.rand(count, 1, device=x.device, generator=rng)
    x = random_warp(x, 0.38, rng, active=(unit < 0.48))
    variation = torch.randn(count, 1, 1, device=x.device, generator=rng).mul(0.045).add(1)
    center = x.mean(dim=1, keepdim=True)
    noise = torch.randn(x.shape, device=x.device, generator=rng, dtype=x.dtype).mul(0.018)
    x = (x - center) * variation + center + noise
    if policy == "motion_sensor":
        group = 6 if dataset == "pamap2" else 3
        probability = 0.18
        active = torch.rand(count, 1, 1, device=x.device, generator=rng) < probability
        x = torch.where(active, drop_sensor(x, group, rng), x)
    return x


def context_neighbors(session, train_indices):
    """Resolve adjacent training windows within the same contiguous labeled recording."""
    position = {int(source): local for local, source in enumerate(train_indices)}
    neighbors = np.full((len(train_indices), 2), -1, dtype=int)
    for local, source in enumerate(train_indices):
        for column, raw in enumerate((source - 1, source + 1)):
            if raw in position and session[raw] == session[source]:
                neighbors[local, column] = position[raw]
    return neighbors


def augment_pace_context(x, pool, rows, labels, neighbors, rng):
    """Resample a centered window from the same training walking or jogging recording."""
    window = x.shape[1]
    matched = neighbors[rows]
    walking = labels[rows] == 0
    jogging = labels[rows] == 3
    threshold = 0.82 * walking.float() + 0.58 * jogging.float()
    selected = (matched >= 0).all(dim=1) & (torch.rand(len(x), generator=rng, device=x.device) < threshold)
    if not selected.any():
        return x
    previous, following = matched[selected].unbind(dim=1)
    middle = pool[rows[selected]]
    context = torch.cat((pool[previous, :window//2], middle, pool[following, -window//2:]), dim=1)
    unit = torch.rand(len(context), 1, device=x.device, generator=rng)
    walk_speed = torch.where(unit < 0.5, 0.65 + 0.36 * unit, 0.93 + 0.68 * unit)
    jog_speed = 0.72 + 0.54 * unit
    speed = torch.where(walking[selected, None], walk_speed, jog_speed)
    timeline = torch.linspace(-0.5, 0.5, window, dtype=x.dtype, device=x.device).view(1, -1)
    coords = 0.5 + timeline * speed * (window - 1) / (context.shape[1] - 1)
    sampled = resample_batch(context, coords)
    center = sampled.mean(dim=1, keepdim=True)
    exponent = 0.35 + 1.50 * torch.rand(speed.shape, device=x.device, generator=rng)
    gain = torch.randn(speed.shape, device=x.device, generator=rng).mul(0.19).exp()
    response = (speed.pow(exponent) * gain).clamp(0.48, 2.2).unsqueeze(-1)
    x = x.clone()
    x[selected] = center + response * (sampled - center)
    return x


def perturb(x, condition, dataset, seed):
    rng = torch.Generator(device=x.device).manual_seed(seed)
    n, steps, channels = x.shape
    if condition == "clean":
        return x
    if condition in ("warp_mild", "warp_strong", "warp_asymmetric"):
        severity = {"warp_mild": 0.30, "warp_strong": 0.65, "warp_asymmetric": 0.55}[condition]
        harmonic = 0.35 if condition == "warp_asymmetric" else 0.0
        return random_warp(x, severity, rng, higher_harmonic=harmonic)
    if condition == "noise":
        return x + torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=rng).mul(0.15)
    if condition == "gain":
        gain = torch.rand(n, 1, channels, dtype=x.dtype, device=x.device, generator=rng).mul(0.5).add(0.75)
        bias = torch.randn(n, 1, channels, dtype=x.dtype, device=x.device, generator=rng).mul(0.12)
        return x * gain + bias
    if condition == "gap":
        width = steps // 6
        start = torch.randint(1, steps - width - 1, (n, 1), device=x.device, generator=rng)
        end = start + width
        timeline = torch.arange(steps, device=x.device).view(1, -1)
        inside = (timeline >= start) & (timeline < end)
        left = x.gather(1, (start - 1).unsqueeze(-1).expand(-1, -1, channels))
        right = x.gather(1, end.unsqueeze(-1).expand(-1, -1, channels))
        fill = torch.lerp(left, right, ((timeline - start + 1) / (width + 1)).unsqueeze(-1))
        return torch.where(inside.unsqueeze(-1), fill, x)
    if condition == "sensor_out":
        group = 6 if dataset == "pamap2" else 3
        return drop_sensor(x, group, rng)
    raise ValueError(f"Unknown test perturbation {condition}")


def drop_sensor(x, group, rng):
    count, _, channels = x.shape
    dropped = torch.randint(channels // group, (count, 1, 1), device=x.device, generator=rng)
    which = torch.arange(channels, device=x.device).view(1, 1, -1) // group
    return x.masked_fill(which == dropped, 0)
