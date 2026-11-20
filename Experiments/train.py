"""Train, checkpoint, and evaluate one resumable subject-disjoint run."""

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import random
import time
from uuid import uuid4

from filelock import FileLock
import numpy as np
import torch
from torch.nn import functional as F

from Datasets.prepare import CLASSES, load_prepared
from .registry import GEOMETRY, make_model, parameter_count
from Proposed_Method.native import KERNEL
from Proposed_Method.model import augment_mount
from Datasets.protocol import (CONDITIONS, augment_pace_context, augment_training, context_neighbors, normalize_split,
                       pace_masks, perturb, random_warp, restrict_pace, subject_splits)


@dataclass(frozen=True)
class Run:
    dataset: str
    regime: str
    model: str
    fold: int
    seed: int
    epochs: int
    batch: int = 192
    width: int = 16
    stride: int = 4
    learning_rate: float = 0.0025
    augmentation: str = "shared"
    patience: int = 5
    validation_warp_weight: float = 0.20
    profile: str = "standard"
    cap: int = 0

    @property
    def identifier(self):
        return f"{self.dataset}/{self.regime}/{self.model}/fold{self.fold:02d}_seed{self.seed}"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def confusion(y, pred, labels):
    return np.bincount(labels * y.astype(np.int64) + pred, minlength=labels**2).reshape(labels, labels)


def metrics_from_cm(cm, absent="observed"):
    actual = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    hit = cm.diagonal()
    observed = actual > 0
    denominator = actual + predicted
    f1 = np.divide(2 * hit, denominator, out=np.zeros_like(hit, dtype=float), where=denominator > 0)
    selected = observed if absent == "observed" else np.ones(len(actual), dtype=bool)
    recall = np.divide(hit, actual, out=np.zeros_like(hit, dtype=float), where=observed)
    return {"macro_f1": float(f1[selected].mean()), "accuracy": float(hit.sum() / actual.sum()),
            "balanced_accuracy": float(recall[observed].mean()), "support": int(actual.sum())}


def seed_process(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@lru_cache(maxsize=1)
def dataset_cached(name, root):
    return load_prepared(name, root)


@torch.inference_mode()
def infer(model, x, batch, dataset, condition="clean", salt=0):
    model.eval()
    logits = []
    for number, inputs in enumerate(x.split(batch)):
        inputs = perturb(inputs, condition, dataset, 90_007 + 181 * number + salt)
        logits.append(model(inputs).detach().cpu())
    probabilities = torch.cat(logits, dim=0).softmax(dim=-1)
    conf, pred = probabilities.max(dim=-1)
    return pred.numpy().astype(np.int8), conf.numpy()


def validation_score(model, val_x, val_y, count, run):
    clean = infer(model, val_x, run.batch, run.dataset)[0]
    score = metrics_from_cm(confusion(val_y, clean, count))["macro_f1"]
    if run.validation_warp_weight:
        shifted = infer(model, val_x, run.batch, run.dataset, "warp_strong", salt=run.fold)[0]
        warped = metrics_from_cm(confusion(val_y, shifted, count))["macro_f1"]
    else:
        warped = score
    weight = run.validation_warp_weight
    return (1 - weight) * score + weight * warped, score, warped


def cap_indices(indices, count, seed):
    if not count or len(indices) <= count:
        return indices
    return np.sort(np.random.default_rng(seed).choice(indices, count, replace=False))


def evaluate_checkpoint(model, x, data, idx, run, metrics_file):
    labels = len(CLASSES[run.dataset])
    actual = data.y[idx].astype(np.int8)
    names = ["clean"] if run.regime in ("pace", "smoke") else list(CONDITIONS)
    prediction, confidence, estimates = [], [], []
    for condition in names:
        pred, conf = infer(model, x, run.batch, run.dataset, condition, salt=run.fold)
        prediction.append(pred)
        confidence.append(np.rint(255 * conf).astype(np.uint8))
        estimates.append(dict(condition=condition, **metrics_from_cm(confusion(actual, pred, labels))))
    if run.regime == "pace":
        for name, keep in pace_masks(data.pace[idx]).items():
            values = np.array(prediction[0])[keep]
            estimates.append(dict(condition=name, **metrics_from_cm(confusion(actual[keep], values, labels))))
    temporary = metrics_file.with_name(metrics_file.stem + ".part.npz")
    np.savez_compressed(temporary, conditions=np.array(names), pred=np.stack(prediction),
                        confidence_u8=np.stack(confidence), actual=actual, subject=data.subject[idx],
                        pace=data.pace[idx], index=idx.astype(np.int32))
    temporary.replace(metrics_file)
    return estimates


def geometry_loss(model, x, target, class_weights, generator):
    fused, elapsed, correction, posture = model.forward_views(x, return_posture=True)
    loss = F.cross_entropy(fused, target, weight=class_weights, label_smoothing=0.015)
    loss = loss + 0.26 * F.cross_entropy(elapsed, target, weight=class_weights, label_smoothing=0.015)
    moving = model.moving[target]
    if moving.any():
        # Movement logits model the activity conditioned on it belonging to the moving group.
        conditional = correction.masked_fill(~model.moving, -1e4)
        loss = loss + 0.23 * F.cross_entropy(conditional[moving], target[moving], weight=class_weights)
    if posture is not None and (~moving).any():
        conditional = posture.masked_fill(model.moving, -1e4)
        loss = loss + 0.27 * F.cross_entropy(conditional[~moving], target[~moving], weight=class_weights)
    selected = moving.nonzero(as_tuple=True)[0][::4]
    if len(selected):
        warped = random_warp(x[selected], 0.58, generator, higher_harmonic=0.3)
        changed = model.motion_logits(warped).masked_fill(~model.moving, -1e4)
        ideal = correction[selected].detach().masked_fill(~model.moving, -1e4)
        loss = loss + 0.17 * F.kl_div(changed.log_softmax(-1), ideal.softmax(-1), reduction="batchmean")
    return loss


def execute(run, data_dir, output, device_name="cpu", threads=1):
    run_dir = Path(output) / "Runs" / run.identifier
    run_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run_dir / ".fit.lock")):
        return execute_locked(run, data_dir, run_dir, device_name, threads)


def execute_locked(run, data_dir, run_dir, device_name, threads):
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    device = torch.device(device_name)
    done = run_dir / "run.json"
    predictions_file = run_dir / "predictions.npz"
    weights_file = run_dir / "checkpoint.pt"
    if done.is_file() and predictions_file.is_file() and weights_file.is_file():
        previous = json.loads(done.read_text())
        if previous["config"] != asdict(run):
            raise ValueError(f"Run configuration changed in {run_dir}. Use a separate output directory.")
        return f"cached {run.identifier}"

    data = dataset_cached(run.dataset, str(Path(data_dir).resolve()))
    split = restrict_pace(list(subject_splits(run.dataset, data))[run.fold], data, run.regime)
    if run.cap:
        split = type(split)(cap_indices(split.train, run.cap, run.seed),
                            cap_indices(split.val, min(run.cap, 256), 111),
                            cap_indices(split.test, min(run.cap, 256), 112), split.fold)
    arrays, mean, std = normalize_split(data, split)
    train_x, val_x, test_x = [torch.from_numpy(a).to(device) for a in arrays]
    train_y = torch.as_tensor(data.y[split.train].astype(np.int64), device=device)
    neighbors = (torch.as_tensor(context_neighbors(data.session, split.train), device=device)
                 if run.augmentation == "gait_shift" else None)
    val_y = data.y[split.val]
    count = len(CLASSES[run.dataset])
    seed_process(run.seed + 31 * run.fold, device)
    model = make_model(run.model, data.x.shape[-1], count, run.width, run.stride).to(device)
    geometry = run.model in GEOMETRY
    if geometry:
        model.calibrate(mean, std, run.dataset, CLASSES[run.dataset])
    generator = torch.Generator(device=device).manual_seed(run.seed * 103 + run.fold * 19 + 23)
    if run.model != "geometry_no_rotation":
        mount_rng = torch.Generator(device=device).manual_seed(run.seed * 271 + 17 * run.fold)
    else:
        mount_rng = None
    mount_mean = torch.as_tensor(mean, device=device)
    mount_std = torch.as_tensor(std, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=run.learning_rate, weight_decay=0.0008, foreach=False)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, run.epochs, eta_min=0.00045)
    freq = torch.bincount(train_y, minlength=count).float().clamp_min(1)
    class_weights = freq.rsqrt()
    class_weights = class_weights / class_weights.mean()
    best, best_epoch, best_weights, history = -1.0, 0, None, []
    start = time.perf_counter()
    for epoch in range(1, run.epochs + 1):
        round_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        order = torch.randperm(len(train_x), generator=generator, device=device)
        for row in order.split(run.batch):
            x = train_x[row]
            if neighbors is not None:
                x = augment_pace_context(x, train_x, row, train_y, neighbors, generator)
            if mount_rng is not None:
                x = augment_mount(x, mount_mean, mount_std, run.dataset, mount_rng)
            x = augment_training(x, generator, run.augmentation, run.dataset)
            opt.zero_grad(set_to_none=True)
            if geometry:
                loss = geometry_loss(model, x, train_y[row], class_weights, generator)
            else:
                loss = F.cross_entropy(model(x), train_y[row], weight=class_weights, label_smoothing=0.015)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            epoch_loss += float(loss.detach()) * row.numel()
        val, val_clean, val_warp = validation_score(model, val_x, val_y, count, run)
        history.append({"epoch": epoch, "train_loss": epoch_loss / len(train_x), "validation_score": val,
                        "validation_macro_f1": val_clean, "validation_warp_macro_f1": val_warp,
                        "lr": float(opt.param_groups[0]["lr"]), "seconds": time.perf_counter() - round_start})
        schedule.step()
        if val > best + 0.0001:
            best_epoch, best = epoch, val
            best_weights = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        if epoch - best_epoch >= run.patience:
            break

    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_weights)
    estimates = evaluate_checkpoint(model, test_x, data, split.test, run, predictions_file)
    timing = []
    with torch.inference_mode():
        sample = test_x[:1]
        for _ in range(4):
            model(sample)
        for _ in range(19):
            before = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timing.append((time.perf_counter() - before) * 1000)
    checkpoint = {
        "state_dict": best_weights, "config": asdict(run), "labels": CLASSES[run.dataset],
        "normalization": {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std)},
        "input_channels": data.x.shape[-1], "best_epoch": best_epoch,
    }
    tmp = weights_file.with_suffix(".pt.part")
    torch.save(checkpoint, tmp)
    tmp.replace(weights_file)
    write_json(done, {
        "config": asdict(run), "params": parameter_count(model), "best_epoch": best_epoch,
        "training_seconds": train_seconds, "latency_ms_batch1": float(np.median(timing)), "history": history,
        "metrics": estimates, "counts": {role: len(getattr(split, role)) for role in ("train", "val", "test")},
        "subjects": {role: sorted(np.unique(data.subject[getattr(split, role)]).tolist())
                     for role in ("train", "val", "test")},
        "device": str(device), "threads": threads, "host_environment": platform.platform(),
        "spline_backend": "native_cpu" if KERNEL is not None and device.type == "cpu" else "pytorch",
        "pytorch_version": torch.__version__, "process_id": os.getpid(),
    })
    return (f"{run.identifier:<58} epoch {best_epoch:2d}/{len(history):<2d} "
            f"val={best:.3f} test={estimates[0]['macro_f1']:.3f} train={train_seconds:.1f}s")
