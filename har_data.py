"""UCI-HAR data preparation for the SKNet / MTHARS comparison.

Two views of the same recordings are needed:

* **Windows** -- the pre-cut 128-timestep windows the dataset ships, one activity
  label each. This is what Gao's SK network consumes.
* **Streams** -- continuous signal with activity boundaries. MTHARS segments and
  recognises at the same time, so it needs unsegmented data (Duan Sec. III-A).

UCI-HAR does not ship the continuous signal, only the windows. It does ship them
in temporal order with 50% overlap, so the stream can be reconstructed by taking
the first window of a run in full and then the second half of every window after
it. ``verify_overlap`` checks that assumption holds on the data as downloaded
rather than trusting it -- run it before relying on any stream built here.

The alternative would be the HAPT release (UCI dataset 341), which ships the raw
continuous recordings with labelled boundaries. Reconstructing from the windows
keeps both models on byte-identical signals, which is what makes the comparison
meaningful.
"""

import zipfile
from pathlib import Path
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
import torch

UCI_URL = ("https://archive.ics.uci.edu/static/public/240/"
           "human+activity+recognition+using+smartphones.zip")
# Alternative mirror:
# https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

SIGNALS = [f"{signal}_{axis}"
           for signal in ("body_acc", "body_gyro", "total_acc")
           for axis in ("x", "y", "z")]

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]

WINDOW = 128        # timesteps per shipped window (2.56 s at 50 Hz)
STEP = WINDOW // 2  # 50% overlap


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def download_uci_har(data_dir: str = "data") -> Path:
    """Download and extract UCI-HAR, returning the dataset root."""
    import requests

    data_dir = Path(data_dir)
    root = data_dir / "UCI HAR Dataset"
    if root.is_dir():
        return root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "uci_har.zip"

    response = requests.get(UCI_URL, timeout=300)
    response.raise_for_status()
    archive.write_bytes(response.content)

    with zipfile.ZipFile(archive) as z:
        z.extractall(data_dir)

    # The UCI download wraps the dataset in a second zip.
    for nested in data_dir.glob("*.zip"):
        if nested != archive:
            with zipfile.ZipFile(nested) as z:
                z.extractall(data_dir)

    if not root.is_dir():
        raise RuntimeError(f"expected {root} after extraction; found "
                           f"{[p.name for p in data_dir.iterdir()]}")
    return root


# --------------------------------------------------------------------------
# Windows (the SKNet view)
# --------------------------------------------------------------------------

class Split(NamedTuple):
    windows: torch.Tensor   # (N, 1, 128, 9)
    labels: torch.Tensor    # (N,) in 0..5
    subjects: torch.Tensor  # (N,)


def load_split(root: Path, split: str) -> Split:
    """Load one UCI-HAR split, keeping the row order the files use."""
    signal_dir = Path(root) / split / "Inertial Signals"

    channels = [np.loadtxt(signal_dir / f"{name}_{split}.txt") for name in SIGNALS]
    x = np.stack(channels, axis=-1)                       # (N, 128, 9)

    y = np.loadtxt(Path(root) / split / f"y_{split}.txt").astype(int) - 1
    subjects = np.loadtxt(Path(root) / split / f"subject_{split}.txt").astype(int)

    return Split(torch.from_numpy(x).float().unsqueeze(1),
                 torch.from_numpy(y).long(),
                 torch.from_numpy(subjects).long())


def normalize(train: Split, *others: Split) -> Tuple[Split, ...]:
    """Zero mean / unit variance per sensor channel, using training statistics only."""
    mean = train.windows.mean(dim=(0, 2), keepdim=True)
    std = train.windows.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)

    scaled = [Split((s.windows - mean) / std, s.labels, s.subjects)
              for s in (train,) + others]
    return tuple(scaled)


# --------------------------------------------------------------------------
# Streams (the MTHARS view)
# --------------------------------------------------------------------------

def _runs(split: Split) -> List[Tuple[int, int]]:
    """Maximal spans of consecutive rows sharing a subject and an activity."""
    spans = []
    start = 0
    for i in range(1, len(split.labels) + 1):
        ends = (i == len(split.labels)
                or split.labels[i] != split.labels[start]
                or split.subjects[i] != split.subjects[start])
        if ends:
            spans.append((start, i))
            start = i
    return spans


def verify_overlap(split: Split, max_runs: int = 50) -> float:
    """Mean absolute mismatch between the overlapping halves of adjacent windows.

    Row i covers timesteps [64i, 64i+128) of its run, so the second half of row i
    and the first half of row i+1 describe the same 64 timesteps. A value near
    zero confirms the rows really are consecutive overlapping windows; a large
    value means the reconstruction below is invalid for this download.
    """
    diffs = []
    for start, end in _runs(split)[:max_runs]:
        for i in range(start, end - 1):
            diffs.append((split.windows[i, 0, STEP:]
                          - split.windows[i + 1, 0, :STEP]).abs().mean())
    return float(torch.stack(diffs).mean()) if diffs else 0.0


class StreamRecording(NamedTuple):
    signal: torch.Tensor              # (T, 9)
    intervals: List[Tuple[int, int, int]]  # (start, end, label), end exclusive


def reconstruct_streams(split: Split) -> List[StreamRecording]:
    """Rebuild continuous per-subject signal with activity boundaries.

    Consecutive same-activity rows are de-overlapped into one contiguous segment;
    the segments of a subject are then concatenated in file order, which is the
    order they were recorded in.
    """
    by_subject: dict = {}

    for start, end in _runs(split):
        subject = int(split.subjects[start])
        label = int(split.labels[start])

        head = split.windows[start, 0]                   # (128, 9)
        tail = [split.windows[i, 0, STEP:] for i in range(start + 1, end)]
        segment = torch.cat([head] + tail, dim=0) if tail else head

        by_subject.setdefault(subject, []).append((segment, label))

    recordings = []
    for subject in sorted(by_subject):
        segments, intervals, cursor = [], [], 0
        for segment, label in by_subject[subject]:
            segments.append(segment)
            intervals.append((cursor, cursor + len(segment), label))
            cursor += len(segment)
        recordings.append(StreamRecording(torch.cat(segments, dim=0), intervals))
    return recordings


class StreamDataset(torch.utils.data.Dataset):
    """Fixed-length data streams with their activity boundaries.

    Duan Sec. III-F: "Since the length of the activity input to the network is
    fixed, we input a fixed length activity data stream at a time to recognize
    the start and end positions of each activity in each segment."

    Each item is ``(stream, boxes, labels, timestep_labels)`` where ``boxes`` are
    normalised (center, length) pairs in [0, 1] and ``timestep_labels`` is the
    per-sample ground truth used for scoring.
    """

    def __init__(self, recordings: Sequence[StreamRecording], stream_length: int,
                 stride: int = None, min_fraction: float = 0.05):
        self.stream_length = stream_length
        self.min_fraction = min_fraction
        stride = stride or stream_length

        self.items = []
        for recording in recordings:
            total = len(recording.signal)
            for start in range(0, max(total - stream_length + 1, 0), stride):
                self.items.append((recording, start))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        recording, start = self.items[index]
        end = start + self.stream_length

        signal = recording.signal[start:end]              # (T, 9)
        timestep_labels = torch.full((self.stream_length,), -1, dtype=torch.long)

        boxes, labels = [], []
        for interval_start, interval_end, label in recording.intervals:
            lo = max(interval_start, start)
            hi = min(interval_end, end)
            if hi <= lo:
                continue
            timestep_labels[lo - start:hi - start] = label

            fraction = (hi - lo) / self.stream_length
            if fraction < self.min_fraction:
                # Too small a sliver to be a detection target, but it stays in
                # timestep_labels so scoring still counts those samples.
                continue
            center = ((lo + hi) / 2 - start) / self.stream_length
            boxes.append([center, fraction])
            labels.append(label)

        return (signal.unsqueeze(0),                      # (1, T, 9)
                torch.tensor(boxes, dtype=torch.float32).reshape(-1, 2),
                torch.tensor(labels, dtype=torch.long),
                timestep_labels)


def collate_streams(batch):
    """Collate variable numbers of boundaries per stream into lists."""
    streams = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    timestep_labels = torch.stack([item[3] for item in batch])
    return streams, boxes, labels, timestep_labels


# --------------------------------------------------------------------------
# Sliding-window baseline decoding
# --------------------------------------------------------------------------

@torch.no_grad()
def sliding_window_predict(model: torch.nn.Module, stream: torch.Tensor,
                           window: int = WINDOW, stride: int = STEP,
                           device=None) -> torch.Tensor:
    """Label every timestep of a stream with a window classifier.

    This is the static sliding-window segmentation MTHARS is argued against
    (Duan Sec. IV-C): cut the stream into fixed windows, classify each, and give
    every timestep the label of the windows covering it. Overlapping windows vote
    by summed class probability, which is the standard way to turn a window
    classifier into a per-sample labelling.

    Args:
        model: window classifier taking (B, 1, window, S) to class logits.
        stream: (1, T, S) or (T, S) single stream.

    Returns:
        (T,) predicted class per timestep.
    """
    if stream.dim() == 3:
        stream = stream[0]
    total = stream.shape[0]

    starts = list(range(0, max(total - window + 1, 1), stride))
    if starts[-1] + window < total:
        starts.append(total - window)

    patches = torch.stack([stream[s:s + window] for s in starts]).unsqueeze(1)
    logits = model(patches.to(device) if device else patches)
    probs = torch.softmax(logits, dim=-1).cpu()

    votes = torch.zeros(total, probs.shape[-1])
    for i, s in enumerate(starts):
        votes[s:s + window] += probs[i]
    return votes.argmax(dim=1)


# --------------------------------------------------------------------------
# Synthetic data (for exercising the pipeline without the download)
# --------------------------------------------------------------------------

def synthetic_recordings(num_subjects: int = 8, num_classes: int = len(ACTIVITIES),
                         segments_per_subject: int = 6,
                         segment_length: int = 400, noise: float = 0.35,
                         seed: int = 0, signature_seed: int = 1234
                         ) -> List[StreamRecording]:
    """Continuous streams whose activities differ by frequency and amplitude.

    Not a substitute for UCI-HAR -- it exists so the training and evaluation
    path can be run end to end where the real download is unavailable. Each
    class gets its own per-channel frequency and offset, so the task is learnable
    but trivially so; numbers from it say nothing about either paper's results.
    """
    # Class signatures come from their own fixed seed so that every call shares
    # them; `seed` varies only the segment order, durations and noise. Without
    # this a train and a test set drawn with different seeds would describe
    # different activities and nothing could generalise between them.
    signature = torch.Generator().manual_seed(signature_seed)
    frequencies = torch.rand(num_classes, len(SIGNALS), generator=signature) * 0.3 + 0.02
    amplitudes = torch.rand(num_classes, len(SIGNALS), generator=signature) * 2 + 0.5
    offsets = torch.randn(num_classes, len(SIGNALS), generator=signature)

    generator = torch.Generator().manual_seed(seed)

    recordings = []
    for subject in range(num_subjects):
        segments, intervals, cursor = [], [], 0
        order = torch.randperm(num_classes, generator=generator).tolist()
        for i in range(segments_per_subject):
            label = order[i % num_classes]
            length = int(segment_length * (0.6 + 0.8 * torch.rand(1, generator=generator).item()))
            t = torch.arange(length, dtype=torch.float32)[:, None]
            signal = (amplitudes[label] * torch.sin(2 * torch.pi * frequencies[label] * t)
                      + offsets[label]
                      + noise * torch.randn(length, len(SIGNALS), generator=generator))
            segments.append(signal)
            intervals.append((cursor, cursor + length, label))
            cursor += length
        recordings.append(StreamRecording(torch.cat(segments, dim=0), intervals))
    return recordings


def windows_from_recordings(recordings: Sequence[StreamRecording],
                            window: int = WINDOW, stride: int = STEP) -> Split:
    """Cut recordings into labelled fixed windows, the view a window classifier needs.

    Only windows lying wholly inside one activity are kept, which is how the
    shipped UCI-HAR windows are built.
    """
    windows, labels, subjects = [], [], []
    for subject, recording in enumerate(recordings):
        for start, end, label in recording.intervals:
            for lo in range(start, end - window + 1, stride):
                windows.append(recording.signal[lo:lo + window])
                labels.append(label)
                subjects.append(subject)

    return Split(torch.stack(windows).unsqueeze(1),
                 torch.tensor(labels, dtype=torch.long),
                 torch.tensor(subjects, dtype=torch.long))


def normalize_recordings(train: Sequence[StreamRecording],
                         *others: Sequence[StreamRecording]):
    """Zero mean / unit variance per sensor channel over whole recordings.

    Used for synthetic data, where the streams are generated first and the
    windows are cut from them: normalising the recordings keeps both views on
    identical statistics, exactly as the UCI path does by normalising the
    windows before reconstructing streams from them.
    """
    stacked = torch.cat([recording.signal for recording in train], dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    std = stacked.std(dim=0, keepdim=True).clamp(min=1e-8)

    scaled = []
    for group in (train,) + others:
        scaled.append([StreamRecording((r.signal - mean) / std, r.intervals)
                       for r in group])
    return tuple(scaled)
