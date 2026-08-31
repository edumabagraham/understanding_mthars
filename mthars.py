"""MTHARS: multi-task human activity recognition and segmentation.

Implements:

    J. Duan, L. Zhang, et al., "A Multi-Task Deep Learning Approach for
    Sensor-based Human Activity Recognition and Segmentation," 2023.

on top of the selective kernel backbone of:

    W. Gao, L. Zhang, W. Huang, F. Min, J. He and A. Song, "Deep Neural
    Networks for Sensor-Based Human Activity Recognition Using Selective
    Kernel Convolution," IEEE TIM, vol. 70, 2021.

Section and equation references in the docstrings point at those two papers;
"Duan" is the MTHARS paper and "Gao" the SK paper.

Coordinate convention
---------------------
Activity boundaries and windows are (center, length) pairs normalised by the
length of the input data stream, so both live in [0, 1] regardless of how long
the stream is. Duan Sec. III-B: "we divide the center of the window by the
length of the feature sequence. Therefore, the value of x indicates the
relative position of the window in the feature sequence."

Tensor layout for the backbone is (B, C, T, S): time on the height axis, sensor
channels on the width axis.
"""

from typing import List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Backbone: the SK network (Gao et al.)
# --------------------------------------------------------------------------

class SKUnit(nn.Module):
    """conv -> BN -> ReLU, the transformation used inside each SK branch.

    Gao Sec. III-A-1: "the three transformations F1, F2 and F3 consist of
    grouped convolutions, batch normalization and ReLU activation in sequence."
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple, padding: tuple, dilation: tuple, groups: int = 1):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=False),  # the bias is cancelled by the BN that follows
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block_1(x)


class SKConv(nn.Module):
    """Selective kernel convolution: Split / Fuse / Select (Gao Sec. III-A).

    Args:
        M: branch number. Branch i uses a 3x1 kernel with dilation i, giving the
           effective receptive fields 3x1, 5x1, 7x1, ... along the time axis.
        G: group number of the grouped convolutions (Gao Table IV: G=32 best).
        r: reduction ratio, d = C / r (Gao Eq. 4; Table VI: r=32 best).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 M: int = 3, G: int = 32, r: int = 32):
        super().__init__()

        if in_channels % G or out_channels % G:
            raise ValueError(
                f"grouped convolution needs in_channels ({in_channels}) and "
                f"out_channels ({out_channels}) divisible by G ({G})")

        # ---- Split: M branches with different receptive fields along time.
        self.branches = nn.ModuleList([
            SKUnit(in_channels, out_channels,
                   kernel_size=(3, 1),
                   stride=(1, 1),
                   padding=(D, 0),   # keeps the temporal length unchanged
                   dilation=(D, 1),  # dilate the TIME axis only
                   groups=G)
            for D in range(1, M + 1)
        ])

        # ---- Fuse: global average pooling (Eq. 2) then an FC layer (Eq. 3).
        self.gap = nn.AdaptiveAvgPool2d(1)

        d = max(out_channels // r, 1)                            # Eq. (4)
        self.bottleneck = nn.Sequential(                          # Eq. (3)
            nn.Conv2d(out_channels, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )

        # ---- Select: one attention matrix per branch, A, B, C (Eqs. 5-7).
        self.attn = nn.ModuleList([
            nn.Conv2d(d, out_channels, kernel_size=1, stride=1) for _ in range(M)
        ])

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        feas = torch.stack([branch(x) for branch in self.branches], dim=1)

        fused = feas.sum(dim=1)                                   # Eq. (1)
        m = self.gap(fused)                                       # Eq. (2)
        n = self.bottleneck(m)                                    # Eq. (3)

        attn = torch.stack([a(n) for a in self.attn], dim=1)      # (B,M,C,1,1)
        attn = torch.softmax(attn, dim=1)                         # Eqs. (5)-(7)

        out = (feas * attn).sum(dim=1)                            # Eq. (8)

        if return_attention:
            return out, attn.flatten(start_dim=3).squeeze(-1)     # (B, M, C)
        return out


class SKNet(nn.Module):
    """Conv64-SKConv128-SKConv256-FC-Softmax (Gao Sec. III-B).

    With ``return_sequence=True`` the classification head is skipped and the
    backbone feature sequence (B, C, n) is returned instead -- the input the
    MTHARS Windows Generate module expects.
    """

    def __init__(self, in_channels: int = 1, ch1: int = 64, ch2: int = 128,
                 ch3: int = 256, num_classes: int = 6,
                 M: int = 3, G: int = 32, r: int = 32,
                 return_sequence: bool = False):
        super().__init__()
        self.return_sequence = return_sequence
        self.out_channels = ch3

        # MTHARS Table I, Layer1: Conv2D(5x1 / 3x1 / 1x0). The stride of 3 is
        # the only temporal downsampling, so it sets the feature sequence length.
        self.layer_1 = SKUnit(in_channels, ch1, kernel_size=(5, 1), stride=(3, 1),
                              padding=(1, 0), dilation=(1, 1), groups=1)
        self.layer_2 = SKConv(ch1, ch2, M=M, G=G, r=r)
        self.layer_3 = SKConv(ch2, ch3, M=M, G=G, r=r)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch3, num_classes),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature sequence (B, C, n), sensor axis averaged out."""
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        # Neither paper states how the sensor axis is reduced before the head;
        # averaging is the assumption made here.
        return x.mean(dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.return_sequence:
            return self.features(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.classifier(x)


def feature_sequence_length(stream_length: int) -> int:
    """Length n of the backbone feature sequence for a stream of T timesteps.

    Only Layer1 changes the temporal length (kernel 5, stride 3, padding 1);
    every SK branch uses padding == dilation with a 3x1 kernel, which is
    length-preserving.
    """
    return (stream_length + 2 * 1 - 5) // 3 + 1


# --------------------------------------------------------------------------
# Multiscale windows (Duan Sec. III-B)
# --------------------------------------------------------------------------

def generate_windows(n: int, scales: Sequence[float],
                     device=None, dtype=torch.float32) -> torch.Tensor:
    """Generate the n x m x 2 anchor windows for a feature sequence of length n.

    Duan Sec. III-B: windows are centred on each unit of the feature sequence,
    and for a scale s the two generated lengths are ``n*sqrt(s)`` and
    ``n/sqrt(s)``. Dividing by n puts everything in normalised stream
    coordinates, so the lengths become ``sqrt(s)`` and ``1/sqrt(s)``.

    Returns:
        (n * len(scales) * 2, 2) tensor of (center, length) pairs. Row order is
        centre-major: all windows for centre 0, then centre 1, and so on, which
        is the order the convolutional heads produce (Duan Sec. III-D).
    """
    if n < 1:
        raise ValueError(f"feature sequence length must be >= 1, got {n}")
    if not len(scales):
        raise ValueError("at least one scale is required")

    centers = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n

    lengths = []
    for s in scales:
        if s <= 0:
            raise ValueError(f"scales must be positive, got {s}")
        root = float(s) ** 0.5
        lengths.extend([root, 1.0 / root])
    lengths = torch.tensor(lengths, device=device, dtype=dtype)

    # (n, A, 2) -> (n*A, 2), centre-major.
    centers = centers[:, None].expand(n, len(lengths))
    lengths = lengths[None, :].expand(n, len(lengths))
    return torch.stack([centers, lengths], dim=-1).reshape(-1, 2)


def windows_per_center(scales: Sequence[float]) -> int:
    """Number of anchor windows generated at each feature-sequence unit."""
    return 2 * len(scales)


def to_edges(boxes: torch.Tensor) -> torch.Tensor:
    """(center, length) -> (start, end)."""
    center, length = boxes[..., 0], boxes[..., 1]
    return torch.stack([center - length / 2, center + length / 2], dim=-1)


def to_center_length(edges: torch.Tensor) -> torch.Tensor:
    """(start, end) -> (center, length)."""
    start, end = edges[..., 0], edges[..., 1]
    return torch.stack([(start + end) / 2, end - start], dim=-1)


def iou_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise 1-D IoU (the Jaccard index of Duan Sec. III-B).

    Args:
        a: (N, 2) windows as (center, length).
        b: (M, 2) boxes as (center, length).

    Returns:
        (N, M) IoU matrix.
    """
    a_edges, b_edges = to_edges(a), to_edges(b)

    inter_start = torch.maximum(a_edges[:, None, 0], b_edges[None, :, 0])
    inter_end = torch.minimum(a_edges[:, None, 1], b_edges[None, :, 1])
    inter = (inter_end - inter_start).clamp(min=0)

    union = a[:, None, 1] + b[None, :, 1] - inter
    return inter / union.clamp(min=1e-9)


# --------------------------------------------------------------------------
# Offset encoding / decoding (Duan Eqs. 1-4)
# --------------------------------------------------------------------------

def encode_offsets(targets: torch.Tensor, windows: torch.Tensor) -> torch.Tensor:
    """Ground-truth offsets of a target boundary relative to its window.

    Duan Eqs. (1)-(2):  fx = (tx - wx) / wl,  fl = log(tl / wl).
    """
    tx, tl = targets[..., 0], targets[..., 1]
    wx, wl = windows[..., 0], windows[..., 1]
    return torch.stack([(tx - wx) / wl, torch.log(tl.clamp(min=1e-9) / wl)], dim=-1)


def decode_offsets(offsets: torch.Tensor, windows: torch.Tensor) -> torch.Tensor:
    """Predicted boundary from predicted offsets.

    Duan Eqs. (3)-(4):  tx = fx * wl + wx,  tl = wl * exp(fl).
    """
    fx, fl = offsets[..., 0], offsets[..., 1]
    wx, wl = windows[..., 0], windows[..., 1]
    return torch.stack([fx * wl + wx, wl * torch.exp(fl)], dim=-1)


# --------------------------------------------------------------------------
# Window labelling and matching (Duan Sec. III-B)
# --------------------------------------------------------------------------

def match_windows(windows: torch.Tensor, targets: torch.Tensor,
                  labels: torch.Tensor, iou_threshold: float = 0.5
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign a ground-truth activity to every anchor window.

    Duan Sec. III-B ("Multiscale window labeling and matching"): repeatedly take
    the largest entry of the window-by-target IoU matrix and assign that pair,
    discarding its row and column, until every target has a window. Then every
    remaining window takes the target with which it has the highest IoU, if that
    IoU exceeds the threshold.

    The first phase guarantees each ground-truth activity is matched by at least
    one window even when no window clears the threshold.

    Args:
        windows: (A, 2) anchors as (center, length).
        targets: (T, 2) ground-truth boundaries as (center, length).
        labels: (T,) activity class of each target, in 0..K-1.
        iou_threshold: threshold for the second phase.

    Returns:
        matched_class: (A,) class per window, 0 = background, activity j -> j+1.
        matched_box: (A, 2) the assigned target boundary (rows for background
            windows are unused by the loss but are filled with the anchor).
    """
    num_anchors = windows.shape[0]
    matched_class = torch.zeros(num_anchors, dtype=torch.long, device=windows.device)
    matched_box = windows.clone()

    if targets.numel() == 0:
        return matched_class, matched_box

    ious = iou_1d(windows, targets)                     # (A, T)
    work = ious.clone()

    # Phase 1: bipartite greedy, one window per target.
    for _ in range(targets.shape[0]):
        best = torch.argmax(work)
        anchor_idx = int(best // work.shape[1])
        target_idx = int(best % work.shape[1])
        if work[anchor_idx, target_idx] < 0:            # everything discarded
            break
        matched_class[anchor_idx] = labels[target_idx] + 1
        matched_box[anchor_idx] = targets[target_idx]
        work[anchor_idx, :] = -1.0
        work[:, target_idx] = -1.0
        # Protect this pairing from being overwritten by phase 2.
        ious[anchor_idx, :] = -1.0
        ious[anchor_idx, target_idx] = 2.0

    # Phase 2: every other window takes its best target above the threshold.
    best_iou, best_target = ious.max(dim=1)
    positive = best_iou >= iou_threshold
    matched_class[positive] = labels[best_target[positive]] + 1
    matched_box[positive] = targets[best_target[positive]]

    return matched_class, matched_box


# --------------------------------------------------------------------------
# Recognition and segmentation module (Duan Sec. III-D)
# --------------------------------------------------------------------------

class RecognitionSegmentationHead(nn.Module):
    """Two convolutional branches: activity class and boundary offset.

    Duan Sec. III-D: a fully connected layer over n x m x 2 windows would be far
    too large, so each branch is a convolution that preserves the sequence
    length, keeping output and input coordinates aligned. The class branch emits
    ``m(k+1)`` channels with channel ``i(k+1)+j`` holding the score of class j
    for window i; the offset branch is the same with 2 channels per window.

    MTHARS Table I gives both as Conv1D(3 / 1 / 1) -- kernel 3, stride 1,
    padding 1.
    """

    def __init__(self, in_channels: int, num_classes: int, num_windows: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_windows = num_windows

        self.class_branch = nn.Conv1d(in_channels, num_windows * (num_classes + 1),
                                      kernel_size=3, stride=1, padding=1)
        self.offset_branch = nn.Conv1d(in_channels, num_windows * 2,
                                       kernel_size=3, stride=1, padding=1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """features: (B, C, n) -> logits (B, n*m, K+1), offsets (B, n*m, 2)."""
        batch, _, n = features.shape

        logits = self.class_branch(features)            # (B, m*(K+1), n)
        logits = logits.view(batch, self.num_windows, self.num_classes + 1, n)
        # -> centre-major (B, n, m, K+1), matching generate_windows' row order.
        logits = logits.permute(0, 3, 1, 2).reshape(batch, n * self.num_windows,
                                                    self.num_classes + 1)

        offsets = self.offset_branch(features)          # (B, m*2, n)
        offsets = offsets.view(batch, self.num_windows, 2, n)
        offsets = offsets.permute(0, 3, 1, 2).reshape(batch, n * self.num_windows, 2)

        return logits, offsets


class MTHARS(nn.Module):
    """SK backbone + Windows Generate + Recognition and Segmentation (Duan Sec. III-C)."""

    def __init__(self, num_classes: int, in_channels: int = 1,
                 scales: Sequence[float] = (2.0, 3.0, 4.0),
                 ch1: int = 64, ch2: int = 128, ch3: int = 256,
                 M: int = 3, G: int = 32, r: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.scales = tuple(scales)

        self.backbone = SKNet(in_channels=in_channels, ch1=ch1, ch2=ch2, ch3=ch3,
                              num_classes=num_classes, M=M, G=G, r=r,
                              return_sequence=True)
        self.head = RecognitionSegmentationHead(
            in_channels=ch3,
            num_classes=num_classes,
            num_windows=windows_per_center(self.scales),
        )

    def forward(self, x: torch.Tensor):
        """x: (B, 1, T, S) -> logits (B, A, K+1), offsets (B, A, 2), windows (A, 2)."""
        features = self.backbone.features(x)            # (B, C, n)
        logits, offsets = self.head(features)
        windows = generate_windows(features.shape[-1], self.scales,
                                   device=x.device, dtype=logits.dtype)
        return logits, offsets, windows


# --------------------------------------------------------------------------
# Multi-task loss (Duan Sec. III-E, Eqs. 5-8)
# --------------------------------------------------------------------------

def multitask_loss(logits: torch.Tensor, offsets: torch.Tensor,
                   windows: torch.Tensor,
                   targets: List[torch.Tensor], labels: List[torch.Tensor],
                   alpha: float = 1.0, beta: float = 1.0,
                   iou_threshold: float = 0.5, negative_ratio: int = 3):
    """L = (1/N) (alpha * L_conf + beta * L_loc), Duan Eq. (8).

    ``L_loc`` is SmoothL1 over the offsets of matched windows (Eq. 5) and
    ``L_conf`` is cross-entropy over the classes (Eq. 7). Negatives are mined:
    the unmatched windows are sorted by their loss and only the worst
    ``negative_ratio`` per positive contribute, "with the number of negative
    samples to the number of positive samples in the ratio of 3:1" (Sec. III-E).

    Args:
        logits: (B, A, K+1) class scores per window.
        offsets: (B, A, 2) predicted offsets per window.
        windows: (A, 2) anchors.
        targets: list of B tensors, each (T_b, 2), ground-truth boundaries.
        labels: list of B tensors, each (T_b,), ground-truth classes in 0..K-1.

    Returns:
        (total, conf_loss, loc_loss) -- all scalars.
    """
    batch = logits.shape[0]
    device = logits.device

    matched_classes, matched_boxes = [], []
    for b in range(batch):
        cls, box = match_windows(windows, targets[b].to(device), labels[b].to(device),
                                 iou_threshold=iou_threshold)
        matched_classes.append(cls)
        matched_boxes.append(box)
    matched_class = torch.stack(matched_classes)        # (B, A)
    matched_box = torch.stack(matched_boxes)            # (B, A, 2)

    positive = matched_class > 0
    num_positive = int(positive.sum())
    if num_positive == 0:
        zero = logits.sum() * 0.0
        return zero, zero, zero

    # ---- Localization loss over positives only (Eq. 5).
    true_offsets = encode_offsets(matched_box, windows.unsqueeze(0))
    loc_loss = F.smooth_l1_loss(offsets[positive], true_offsets[positive],
                                reduction="sum")

    # ---- Hard negative mining, then classification loss (Eq. 7).
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = matched_class.reshape(-1)
    all_conf = F.cross_entropy(flat_logits, flat_target, reduction="none")
    all_conf = all_conf.view(batch, -1)

    negative_conf = all_conf.clone()
    negative_conf[positive] = -1.0                      # rank negatives only
    _, order = negative_conf.sort(dim=1, descending=True)
    rank = order.argsort(dim=1)
    num_negative = torch.clamp(positive.sum(dim=1) * negative_ratio,
                               max=positive.shape[1] - 1)
    negative = rank < num_negative.unsqueeze(1)

    conf_loss = all_conf[positive].sum() + all_conf[negative].sum()

    total = (alpha * conf_loss + beta * loc_loss) / num_positive
    return total, conf_loss / num_positive, loc_loss / num_positive


# --------------------------------------------------------------------------
# Inference: NMS and segment concatenation (Duan Sec. III-B, Algorithm 1)
# --------------------------------------------------------------------------

def nms_1d(boxes: torch.Tensor, scores: torch.Tensor,
           iou_threshold: float = 0.45, top_k: int = 200) -> torch.Tensor:
    """Non-maximum suppression on 1-D boundaries (Duan Sec. III-B).

    Args:
        boxes: (N, 2) as (center, length).
        scores: (N,) class probability of each box.

    Returns:
        Indices of the kept boxes, highest score first.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)[:top_k]
    keep = []
    while order.numel() > 0:
        best = order[0]
        keep.append(best)
        if order.numel() == 1:
            break
        ious = iou_1d(boxes[order[1:]], boxes[best].unsqueeze(0)).squeeze(1)
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep)


@torch.no_grad()
def detect(logits: torch.Tensor, offsets: torch.Tensor, windows: torch.Tensor,
           score_threshold: float = 0.5, iou_threshold: float = 0.45,
           top_k: int = 200) -> List[dict]:
    """Turn raw head outputs into per-stream detections.

    Applies softmax over classes, drops the background class, thresholds, and
    runs NMS per class.

    Returns:
        One dict per batch element with keys ``boxes`` (N, 2) in normalised
        (center, length), ``labels`` (N,) in 0..K-1 and ``scores`` (N,).
    """
    probs = torch.softmax(logits, dim=-1)
    batch, _, num_classes_bg = probs.shape

    results = []
    for b in range(batch):
        boxes = decode_offsets(offsets[b], windows)
        keep_boxes, keep_labels, keep_scores = [], [], []

        for c in range(1, num_classes_bg):              # skip background
            class_scores = probs[b, :, c]
            mask = class_scores > score_threshold
            if not mask.any():
                continue
            candidate_boxes = boxes[mask]
            candidate_scores = class_scores[mask]
            kept = nms_1d(candidate_boxes, candidate_scores,
                          iou_threshold=iou_threshold, top_k=top_k)
            keep_boxes.append(candidate_boxes[kept])
            keep_scores.append(candidate_scores[kept])
            keep_labels.append(torch.full((len(kept),), c - 1, dtype=torch.long,
                                          device=logits.device))

        if keep_boxes:
            results.append({"boxes": torch.cat(keep_boxes),
                            "labels": torch.cat(keep_labels),
                            "scores": torch.cat(keep_scores)})
        else:
            results.append({"boxes": torch.zeros(0, 2, device=logits.device),
                            "labels": torch.zeros(0, dtype=torch.long,
                                                  device=logits.device),
                            "scores": torch.zeros(0, device=logits.device)})
    return results


def segments_to_labels(detection: dict, stream_length: int,
                       background: int = -1) -> torch.Tensor:
    """Paint detected segments onto a per-timestep label vector.

    Duan Algorithm 1 concatenates detections into contiguous (start, end)
    segments. Painting highest-score-last gives the same contiguous result while
    staying well defined when detections overlap or leave gaps, which lets the
    output be scored per timestep against the ground truth.
    """
    labels = torch.full((stream_length,), background, dtype=torch.long)

    order = detection["scores"].argsort()               # ascending: best paints last
    edges = to_edges(detection["boxes"][order]) * stream_length
    detected = detection["labels"][order]

    for (start, end), label in zip(edges.tolist(), detected.tolist()):
        lo = max(int(round(start)), 0)
        hi = min(int(round(end)), stream_length)
        if hi > lo:
            labels[lo:hi] = label
    return labels


def concatenate_segments(labels: torch.Tensor) -> List[Tuple[int, int, int]]:
    """Collapse a per-timestep label vector into (start, end, label) runs.

    This is the observable output of Duan Algorithm 1: consecutive positions
    carrying the same activity class become one segment.
    """
    if labels.numel() == 0:
        return []

    segments = []
    start = 0
    current = int(labels[0])
    for i in range(1, len(labels)):
        value = int(labels[i])
        if value != current:
            segments.append((start, i, current))
            start = i
            current = value
    segments.append((start, len(labels), current))
    return segments


# --------------------------------------------------------------------------
# Metrics (Duan Sec. IV-B)
# --------------------------------------------------------------------------

def levenshtein(a: Sequence, b: Sequence) -> int:
    """Levenshtein edit distance between two label sequences (Duan Eq. 10)."""
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, sym_a in enumerate(a, start=1):
        current = [i]
        for j, sym_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (sym_a != sym_b)))
        previous = current
    return previous[-1]


def normalized_edit_distance(predicted: Sequence, truth: Sequence) -> float:
    """NED = lev(predicted, truth) / len(truth), Duan Eq. (9).

    Lower is better: 0 means the predicted activity sequence is identical to the
    ground-truth sequence.
    """
    if len(truth) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    return levenshtein(predicted, truth) / len(truth)


def weighted_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    """Class-frequency-weighted F1 (Duan Eq. 11).

    "Since the activity classes in human activity data are mostly unbalanced,
    using classification accuracy is not an appropriate criterion" -- each
    class's F1 is weighted by Nc / Ntotal.
    """
    total = y_true.numel()
    if total == 0:
        return 0.0

    score = 0.0
    for c in range(num_classes):
        true_positive = int(((y_pred == c) & (y_true == c)).sum())
        false_positive = int(((y_pred == c) & (y_true != c)).sum())
        false_negative = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())
        if support == 0:
            continue

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        if precision + recall > 0:
            score += (support / total) * 2 * precision * recall / (precision + recall)
    return score
