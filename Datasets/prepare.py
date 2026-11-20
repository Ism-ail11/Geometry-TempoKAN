"""Download, decode, and segment the three original UCI releases."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from filelock import FileLock
import numpy as np
from scipy.signal import resample_poly


RELEASES = {
    "uci_har": ("240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip", "10.24432/C54S4K"),
    "pamap2": ("231/pamap2%2Bphysical%2Bactivity%2Bmonitoring.zip", "10.24432/C5NW2H"),
    "selfback": ("521/selfback.zip", "10.24432/C5SK65"),
}
UCI_ROOT = "https://archive.ics.uci.edu/static/public/"
SAMPLE_RATE = 50
WINDOW = 128
STRIDE = 64
CACHE_VERSION = 1
CLASSES = {
    "uci_har": ("Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Lying"),
    "pamap2": ("Lying", "Sitting", "Standing", "Walking", "Running", "Cycling", "Nordic walking",
               "Upstairs", "Downstairs", "Vacuum cleaning", "Ironing", "Rope jumping"),
    "selfback": ("Walking", "Upstairs", "Downstairs", "Jogging", "Standing", "Sitting", "Lying"),
}
PACES = {-1: "nonwalking", 0: "slow", 1: "medium", 2: "fast"}


@dataclass(frozen=True)
class Windows:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    pace: np.ndarray
    session: np.ndarray
    official_test: np.ndarray

    def __post_init__(self):
        n = len(self.x)
        if self.x.ndim != 3 or self.x.shape[1] != WINDOW or n == 0:
            raise ValueError(f"Expected nonempty N × {WINDOW} × C data, got {self.x.shape}")
        if any(a.shape != (n,) for a in (self.y, self.subject, self.pace, self.session, self.official_test)):
            raise ValueError("Window metadata does not match signal dimensions")


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
        return digest.hexdigest()


def download(name, raw_dir):
    destination = Path(raw_dir) / f"{name}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination.with_suffix(".lock"))):
        return download_locked(name, destination)


def download_locked(name, destination):
    if destination.is_file():
        with ZipFile(destination) as archive:
            if not archive.namelist():
                raise ValueError(f"Empty manually supplied archive {destination}")
        print(f"{name}: already downloaded ({destination.stat().st_size:,} bytes)", flush=True)
        return destination
    candidates = [UCI_ROOT + RELEASES[name][0]]
    if name == "pamap2":
        candidates.append("https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip")
    if name == "uci_har":
        candidates.append("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip")
    temporary = destination.with_suffix(".zip.part")
    problems = []
    for attempt in range(3):
        for url in candidates:
            try:
                req = Request(url, headers={"User-Agent": "Geometry TempoKAN benchmark/1.0"})
                with urlopen(req, timeout=90) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=2**20)
                with ZipFile(temporary) as archive:
                    if not archive.namelist():
                        raise ValueError("Download contained an empty archive")
                temporary.replace(destination)
                destination.with_suffix(".source.txt").write_text(url + "\n", encoding="utf-8")
                print(f"{name}: downloaded {destination.stat().st_size:,} bytes", flush=True)
                return destination
            except (OSError, ValueError, RuntimeError, BadZipFile) as error:
                problems.append(f"{url} ({error})")
                temporary.unlink(missing_ok=True)
        time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {name}. Place its release at {destination}. " + " | ".join(problems))


def inner_archive(archive, required, depth=0):
    if any(required(entry.filename) for entry in archive.infolist()):
        return archive
    if depth >= 3:
        raise ValueError("Expected sensor data in the release archive")
    for entry in archive.infolist():
        if entry.filename.lower().endswith(".zip") and not entry.filename.startswith("__MACOSX"):
            nested = ZipFile(BytesIO(archive.read(entry)))
            try:
                return inner_archive(nested, required, depth + 1)
            except ValueError:
                nested.close()
    raise ValueError("Expected sensor data in the release archive")


def read_array(archive, basename, dtype=np.float32):
    matches = [file for file in archive.namelist() if file.replace("\\", "/").endswith(basename)]
    if len(matches) != 1:
        raise ValueError(f"Expected one file ending in {basename}, got {matches}")
    return np.loadtxt(BytesIO(archive.read(matches[0])), dtype=dtype)


def load_uci_har(path):
    with ZipFile(path) as release:
        archive = inner_archive(release, lambda name: name.endswith("total_acc_x_train.txt"))
        arrays = []
        for partition in ("train", "test"):
            channels = [
                read_array(archive, f"/{partition}/Inertial Signals/{signal}_{axis}_{partition}.txt")
                for signal in ("total_acc", "body_gyro") for axis in "xyz"
            ]
            x = np.stack(channels, axis=-1)
            y = read_array(archive, f"/{partition}/y_{partition}.txt", np.int16) - 1
            subject = read_array(archive, f"/{partition}/subject_{partition}.txt", np.int16)
            arrays.append((x, y, subject, np.full(len(x), -1, np.int8),
                           np.full(len(x), -1, np.int32), np.full(len(x), partition == "test")))
        result = Windows(*(np.concatenate(block) for block in zip(*arrays)))
    if (len(result.x), len(np.unique(result.subject))) != (10299, 30):
        raise ValueError("UCI HAR counts did not match the original release")
    return result


def frame_signal(signal, stride=STRIDE):
    if len(signal) < WINDOW:
        return np.empty((0, WINDOW, signal.shape[1]), np.float32)
    view = np.lib.stride_tricks.sliding_window_view(signal, WINDOW, axis=0)[::stride]
    return np.ascontiguousarray(view.swapaxes(1, 2), dtype=np.float32)


def downsample_100(signal):
    return np.asarray(resample_poly(signal, up=1, down=2, axis=0, padtype="line"), dtype=np.float32)


def interpolate_short_gaps(values, max_gap=10):
    """Fill bounded interior gaps and leave long or unbounded gaps invalid."""
    values = values.copy()
    for channel in range(values.shape[1]):
        signal = values[:, channel]
        missing = ~np.isfinite(signal)
        transitions = np.diff(np.r_[False, missing, False].astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        for start, stop in zip(starts, stops):
            if start and stop < len(signal) and stop - start <= max_gap:
                signal[start:stop] = np.linspace(signal[start - 1], signal[stop], stop - start + 2)[1:-1]
    return values


def pack_sequences(sequences):
    arrays = []
    for signal, label, subject, pace, session in sequences:
        windows = frame_signal(downsample_100(signal))
        if len(windows):
            count = len(windows)
            arrays.append((windows, np.full(count, label, np.int16), np.full(count, subject, np.int16),
                           np.full(count, pace, np.int8), np.full(count, session, np.int32),
                           np.zeros(count, dtype=bool)))
    if not arrays:
        raise ValueError("No eligible labeled windows in the dataset")
    return Windows(*(np.concatenate(block) for block in zip(*arrays)))


def load_pamap2(path):
    label_map = {label: order for order, label in enumerate((1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24))}
    columns = [start + offset for start in (3, 20, 37) for offset in (1, 2, 3, 7, 8, 9)]
    protocol_file = re.compile(r"(?:^|/)Protocol/subject(10[1-8])\.dat$", re.I)

    def sequences(archive):
        session = 0
        for name in sorted(archive.namelist()):
            match = protocol_file.search(name.replace("\\", "/"))
            if not match:
                continue
            subject = int(match[1])
            with archive.open(name) as stream:
                raw = np.loadtxt(stream, usecols=(0, 1, *columns))
            labels = raw[:, 1].astype(np.int16)
            boundaries = np.r_[0, np.flatnonzero((np.diff(labels) != 0) | (np.diff(raw[:, 0]) > 0.015)) + 1,
                               len(labels)]
            for begin, end in zip(boundaries[:-1], boundaries[1:]):
                label = int(labels[begin])
                if label not in label_map or end - begin < 2 * WINDOW - 1:
                    continue
                signal = interpolate_short_gaps(raw[begin:end, 2:].astype(np.float32))
                finite = np.isfinite(signal).all(axis=1)
                edges = np.diff(np.r_[False, finite, False].astype(np.int8))
                for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    if right - left >= 2 * WINDOW - 1:
                        yield signal[left:right], label_map[label], subject, -1, session
                        session += 1

    with ZipFile(path) as release:
        archive = inner_archive(release, lambda name: protocol_file.search(name))
        result = pack_sequences(sequences(archive))
    if set(result.subject) != set(range(101, 109)) or set(result.y) != set(range(12)):
        raise ValueError("PAMAP2 protocol participant or class coverage did not match")
    return result


SELFBACK_LABELS = {
    "walkslow": (0, 0), "walkmod": (0, 1), "walkfast": (0, 2), "upstairs": (1, -1),
    "downstairs": (2, -1), "jogging": (3, -1), "standing": (4, -1), "sitting": (5, -1),
    "lying": (6, -1),
}


def selfback_metadata(name):
    path = PurePosixPath(name.replace("\\", "/").lower())
    if any(part.startswith(".") or part == "__macosx" for part in path.parts):
        return None
    if "wt" not in path.parts or path.suffix not in ("", ".csv", ".txt", ".dat"):
        return None
    stem = path.stem
    aliases = {
        "walkslow": ("walkslow", "walkingslow", "slowwalking", "slowwalk"),
        "walkmod": ("walkmod", "walkmedium", "walkingmedium", "walkmoderate", "walknormal"),
        "walkfast": ("walkfast", "walkingfast", "fastwalking", "fastwalk"),
        "upstairs": ("upstairs", "stairsup", "stairascent"),
        "downstairs": ("downstairs", "stairsdown", "stairdescent"),
        "jogging": ("jogging", "jog", "running"),
        "standing": ("standing", "stand"), "sitting": ("sitting", "sit"),
        "lying": ("lying", "lie", "lay"),
    }
    token = re.sub(r"[^a-z]", "", stem)
    subject_ids = re.findall(r"\d+", stem)
    for activity, terms in aliases.items():
        if any(term in token for term in terms) and len(subject_ids) == 1:
            label, pace = SELFBACK_LABELS[activity]
            return int(subject_ids[0]), label, pace
    return None


def parse_selfback_values(content):
    first_line = next(line for line in content.splitlines() if line.strip())
    delimiter = "," if b"," in first_line else (";" if b";" in first_line else None)
    probe = first_line.decode("utf-8-sig")
    fields = probe.split(delimiter)
    skiprows = 0
    try:
        [float(field) for field in fields]
    except ValueError:
        if any(re.search(r"[a-z]", field, re.I) for field in fields):
            skiprows = 1
    data = np.loadtxt(BytesIO(content), delimiter=delimiter, dtype=np.float64, skiprows=skiprows, ndmin=2)
    if data.shape[1] not in (6, 7):
        raise ValueError(f"Expected six synchronized axes and optional timestamp. Got {data.shape}")
    if not np.isfinite(data[:, -6:]).all():
        raise ValueError("Unexpected invalid acceleration values in SELFBACK")
    return data


def load_selfback(path):
    def sequences(archive):
        session = 0
        for name in sorted(archive.namelist()):
            meta = selfback_metadata(name)
            if meta is None:
                continue
            subject, label, pace = meta
            data = parse_selfback_values(archive.read(name))
            if data.shape[1] == 7:
                timestamps = data[:, 0]
                steps = np.diff(timestamps)
                positive = steps[steps > 0]
                if len(positive):
                    step = np.median(positive)
                    boundaries = np.r_[0, np.flatnonzero((steps > 1.5 * step) | (steps <= 0)) + 1, len(data)]
                else:
                    boundaries = np.array([0, len(data)])
            else:
                boundaries = np.array([0, len(data)])
            for begin, end in zip(boundaries[:-1], boundaries[1:]):
                if end - begin >= 2 * WINDOW - 1:
                    yield data[begin:end, -6:].astype(np.float32), label, subject, pace, session
                    session += 1

    with ZipFile(path) as release:
        archive = inner_archive(release, lambda name: selfback_metadata(name) is not None)
        result = pack_sequences(sequences(archive))
    if len(np.unique(result.subject)) != 33 or set(result.y) != set(range(7)) or set(result.pace) != set(PACES):
        raise ValueError("SELFBACK participant, class, or pace coverage did not match")
    return result


LOADERS = {"uci_har": load_uci_har, "pamap2": load_pamap2, "selfback": load_selfback}


def prepare_dataset(name, root):
    root = Path(root)
    source = download(name, root / "raw")
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    with FileLock(str(prepared / f"{name}.lock")):
        return prepare_locked(name, source, prepared)


def prepare_locked(name, source, prepared):
    destination = prepared / f"{name}.npz"
    provenance = prepared / f"{name}.json"
    archive_hash = sha256(source)
    if destination.is_file() and provenance.is_file():
        metadata = json.loads(provenance.read_text())
        if metadata.get("cache_version") == CACHE_VERSION and metadata.get("archive_sha256") == archive_hash:
            return metadata
    data = LOADERS[name](source)
    if not np.isfinite(data.x).all():
        raise ValueError(f"Processed {name} has nonfinite inputs")
    temporary = prepared / f"{name}.part.npz"
    np.savez(temporary, **data.__dict__)
    temporary.replace(destination)
    saved_url = source.with_suffix(".source.txt")
    metadata = {
        "dataset": name, "citation_doi": RELEASES[name][1], "archive_sha256": archive_hash,
        "archive_bytes": source.stat().st_size, "cache_version": CACHE_VERSION,
        "source_url": saved_url.read_text().strip() if saved_url.is_file() else UCI_ROOT + RELEASES[name][0],
        "shape": list(data.x.shape), "sample_rate": SAMPLE_RATE, "window_samples": WINDOW,
        "stride_samples": STRIDE, "subjects": [int(s) for s in np.unique(data.subject)],
        "classes": {label: int((data.y == i).sum()) for i, label in enumerate(CLASSES[name])},
        "pace_counts": {label: int((data.pace == i).sum()) for i, label in PACES.items()} if name == "selfback" else {},
    }
    provenance.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{name}: prepared {data.x.shape} from {len(metadata['subjects'])} participants", flush=True)
    return metadata


def load_prepared(name, root):
    with np.load(Path(root) / "prepared" / f"{name}.npz", allow_pickle=False) as arrays:
        return Windows(**{key: arrays[key] for key in Windows.__dataclass_fields__})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", choices=RELEASES, default=list(RELEASES))
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    work = ((lambda name: download(name, args.data_dir / "raw")) if args.download_only
            else (lambda name: prepare_dataset(name, args.data_dir)))
    with ThreadPoolExecutor(max_workers=len(args.datasets)) as executor:
        futures = [executor.submit(work, name) for name in dict.fromkeys(args.datasets)]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
