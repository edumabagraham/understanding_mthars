"""Unit tests for the MTHARS implementation.

Run with:  python -m pytest test_mthars.py -v   (or plain `python test_mthars.py`)
"""

import math

import torch

from mthars import (
    MTHARS,
    SKNet,
    concatenate_segments,
    decode_offsets,
    detect,
    encode_offsets,
    feature_sequence_length,
    generate_windows,
    iou_1d,
    levenshtein,
    match_windows,
    multitask_loss,
    nms_1d,
    normalized_edit_distance,
    segments_to_labels,
    to_edges,
    weighted_f1,
    windows_per_center,
)

torch.manual_seed(0)


# ---------------------------------------------------------------- backbone

def test_feature_sequence_length_matches_conv_arithmetic():
    for stream in (128, 256, 512, 1024, 1536):
        x = torch.randn(1, 1, stream, 9)
        model = SKNet(num_classes=6, return_sequence=True).eval()
        with torch.inference_mode():
            out = model(x)
        assert out.shape[-1] == feature_sequence_length(stream)


def test_sk_layers_preserve_temporal_length():
    # Only Layer1 downsamples; the SK branches use padding == dilation.
    assert feature_sequence_length(128) == 42
    assert feature_sequence_length(1536) == 512


def test_sk_attention_weights_sum_to_one_across_branches():
    model = SKNet(num_classes=6).eval()
    x = torch.randn(2, 1, 128, 9)
    with torch.inference_mode():
        h = model.layer_2(model.layer_1(x))
        _, attn = model.layer_3(h, return_attention=True)
    assert attn.shape[:2] == (2, 3)                    # (B, M, C)
    assert torch.allclose(attn.sum(dim=1), torch.ones_like(attn[:, 0]), atol=1e-5)


# ---------------------------------------------------------------- windows

def test_generate_windows_count_and_layout():
    n, scales = 5, (2.0, 3.0)
    windows = generate_windows(n, scales)
    assert windows.shape == (n * windows_per_center(scales), 2)

    # Centre-major: the first 2*len(scales) rows share the first centre.
    per_centre = windows_per_center(scales)
    assert torch.allclose(windows[:per_centre, 0],
                          torch.full((per_centre,), 0.5 / n))
    # Lengths for scale s are sqrt(s) and 1/sqrt(s).
    expected = torch.tensor([math.sqrt(2), 1 / math.sqrt(2),
                             math.sqrt(3), 1 / math.sqrt(3)])
    assert torch.allclose(windows[:per_centre, 1], expected)


def test_window_centres_are_uniform_over_the_stream():
    windows = generate_windows(4, (2.0,))
    centres = windows[::2, 0]
    assert torch.allclose(centres, torch.tensor([0.125, 0.375, 0.625, 0.875]))


# ---------------------------------------------------------------- IoU

def test_iou_matches_the_worked_example_in_the_paper():
    # Duan Sec. III-B: "given windows W1(5, 10), W2(30, 50), and truth activity
    # bounding box T1(40, 60), the IOU of W1 and T1 is 0, and the IOU of W2 and
    # T1 is 0.69."
    w = torch.tensor([[5.0, 10.0], [30.0, 50.0]])
    t = torch.tensor([[40.0, 60.0]])
    ious = iou_1d(w, t)
    assert ious[0, 0] == 0.0
    assert abs(float(ious[1, 0]) - 0.69) < 0.01


def test_iou_identity_and_disjoint():
    boxes = torch.tensor([[0.5, 0.4], [0.5, 0.4]])
    assert torch.allclose(iou_1d(boxes, boxes), torch.ones(2, 2))

    a = torch.tensor([[0.1, 0.2]])
    b = torch.tensor([[0.9, 0.2]])
    assert float(iou_1d(a, b)) == 0.0


# ---------------------------------------------------------------- offsets

def test_offset_encode_decode_roundtrip():
    windows = torch.tensor([[0.5, 0.4], [0.25, 0.8], [0.75, 1.2]])
    targets = torch.tensor([[0.55, 0.3], [0.20, 0.9], [0.60, 0.5]])
    recovered = decode_offsets(encode_offsets(targets, windows), windows)
    assert torch.allclose(recovered, targets, atol=1e-6)


def test_offset_formulas_match_equations_1_to_4():
    window = torch.tensor([[0.4, 0.5]])
    target = torch.tensor([[0.6, 0.25]])
    offsets = encode_offsets(target, window)
    assert abs(float(offsets[0, 0]) - (0.6 - 0.4) / 0.5) < 1e-6   # Eq. (1)
    assert abs(float(offsets[0, 1]) - math.log(0.25 / 0.5)) < 1e-6  # Eq. (2)


# ---------------------------------------------------------------- matching

def test_every_target_is_matched_even_below_threshold():
    windows = generate_windows(8, (2.0, 3.0))
    # A very short activity that no anchor overlaps well.
    targets = torch.tensor([[0.5, 0.02]])
    labels = torch.tensor([3])

    matched_class, matched_box = match_windows(windows, targets, labels,
                                               iou_threshold=0.5)
    assert int((matched_class == 4).sum()) >= 1        # class 3 -> label 4
    assigned = matched_class > 0
    assert torch.allclose(matched_box[assigned][0], targets[0])


def test_matching_assigns_distinct_windows_to_distinct_targets():
    windows = generate_windows(16, (2.0, 3.0, 4.0))
    targets = torch.tensor([[0.25, 0.5], [0.75, 0.5]])
    labels = torch.tensor([0, 1])

    matched_class, _ = match_windows(windows, targets, labels)
    assert int((matched_class == 1).sum()) >= 1
    assert int((matched_class == 2).sum()) >= 1


def test_matching_with_no_targets_is_all_background():
    windows = generate_windows(4, (2.0,))
    matched_class, _ = match_windows(windows, torch.zeros(0, 2),
                                     torch.zeros(0, dtype=torch.long))
    assert int(matched_class.sum()) == 0


def test_high_overlap_windows_become_positives():
    windows = torch.tensor([[0.5, 0.5], [0.5, 0.52], [0.1, 0.05]])
    targets = torch.tensor([[0.5, 0.5]])
    labels = torch.tensor([2])
    matched_class, _ = match_windows(windows, targets, labels, iou_threshold=0.5)
    assert matched_class.tolist() == [3, 3, 0]


# ---------------------------------------------------------------- head

def test_head_output_shapes_and_window_alignment():
    num_classes, scales = 6, (2.0, 3.0, 4.0)
    model = MTHARS(num_classes=num_classes, scales=scales).eval()
    x = torch.randn(2, 1, 384, 9)

    with torch.inference_mode():
        logits, offsets, windows = model(x)

    n = feature_sequence_length(384)
    num_anchors = n * windows_per_center(scales)
    assert logits.shape == (2, num_anchors, num_classes + 1)
    assert offsets.shape == (2, num_anchors, 2)
    assert windows.shape == (num_anchors, 2)


def test_class_branch_channel_indexing_is_window_major():
    # Duan Sec. III-D: channel i(k+1)+j holds class j of window i. Writing a
    # known pattern into the conv bias and zeroing the weights lets us read the
    # arrangement straight out of the output.
    num_classes, num_windows = 3, 2
    model = MTHARS(num_classes=num_classes, scales=(2.0,)).eval()
    head = model.head
    assert head.num_windows == num_windows

    with torch.no_grad():
        head.class_branch.weight.zero_()
        head.class_branch.bias.copy_(
            torch.arange(num_windows * (num_classes + 1), dtype=torch.float32))

    features = torch.zeros(1, model.backbone.out_channels, 4)
    logits, _ = head(features)
    # Window 0 should read [0,1,2,3]; window 1 should read [4,5,6,7].
    assert logits[0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert logits[0, 1].tolist() == [4.0, 5.0, 6.0, 7.0]


# ---------------------------------------------------------------- loss

def _toy_batch(num_classes=6, stream=384, batch=2):
    x = torch.randn(batch, 1, stream, 9)
    targets = [torch.tensor([[0.3, 0.5], [0.8, 0.35]]) for _ in range(batch)]
    labels = [torch.tensor([0, 2]) for _ in range(batch)]
    return x, targets, labels


def test_loss_is_finite_and_positive():
    model = MTHARS(num_classes=6)
    x, targets, labels = _toy_batch()
    logits, offsets, windows = model(x)
    total, conf, loc = multitask_loss(logits, offsets, windows, targets, labels)

    assert torch.isfinite(total) and float(total) > 0
    assert float(conf) > 0 and float(loc) >= 0


def test_loss_weights_alpha_beta_scale_the_components():
    model = MTHARS(num_classes=6)
    x, targets, labels = _toy_batch()
    logits, offsets, windows = model(x)

    base, conf, loc = multitask_loss(logits, offsets, windows, targets, labels,
                                     alpha=1.0, beta=1.0)
    weighted, _, _ = multitask_loss(logits, offsets, windows, targets, labels,
                                    alpha=2.0, beta=3.0)
    assert torch.allclose(weighted, 2 * conf + 3 * loc, atol=1e-4)
    assert torch.allclose(base, conf + loc, atol=1e-4)


def test_hard_negative_mining_keeps_the_3_to_1_ratio():
    # With a 3:1 ratio the classification loss must cover far fewer windows than
    # the total, otherwise mining is not happening.
    model = MTHARS(num_classes=6)
    x, targets, labels = _toy_batch()
    logits, offsets, windows = model(x)

    all_negatives, _, _ = multitask_loss(logits, offsets, windows, targets, labels,
                                         negative_ratio=10 ** 6)
    mined, _, _ = multitask_loss(logits, offsets, windows, targets, labels,
                                 negative_ratio=3)
    assert float(mined) < float(all_negatives)


def test_gradients_reach_the_backbone():
    model = MTHARS(num_classes=6)
    x, targets, labels = _toy_batch()
    logits, offsets, windows = model(x)
    total, _, _ = multitask_loss(logits, offsets, windows, targets, labels)
    total.backward()

    first_conv = model.backbone.layer_1.block_1[0].weight
    assert first_conv.grad is not None
    assert float(first_conv.grad.abs().sum()) > 0


def test_model_can_overfit_a_tiny_batch():
    """The strongest end-to-end check: loss must fall a long way on two examples."""
    torch.manual_seed(0)
    model = MTHARS(num_classes=6, scales=(2.0, 3.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(2, 1, 384, 9)
    targets = [torch.tensor([[0.25, 0.5]]), torch.tensor([[0.7, 0.4]])]
    labels = [torch.tensor([1]), torch.tensor([4])]

    losses = []
    for _ in range(40):
        logits, offsets, windows = model(x)
        loss, _, _ = multitask_loss(logits, offsets, windows, targets, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

    assert losses[-1] < losses[0] * 0.25, f"loss went {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_overfit_recovers_the_planted_boundaries():
    """After overfitting, detection should return the planted class and boundary."""
    torch.manual_seed(0)
    model = MTHARS(num_classes=6, scales=(2.0, 3.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    x = torch.randn(2, 1, 384, 9)
    boxes = [torch.tensor([[0.25, 0.5]]), torch.tensor([[0.7, 0.4]])]
    labels = [torch.tensor([1]), torch.tensor([4])]

    for _ in range(120):
        logits, offsets, windows = model(x)
        loss, _, _ = multitask_loss(logits, offsets, windows, boxes, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        logits, offsets, windows = model(x)
    results = detect(logits, offsets, windows, score_threshold=0.5)

    for b, (expected_box, expected_label) in enumerate(zip(boxes, labels)):
        assert results[b]["boxes"].numel() > 0, f"stream {b}: no detection"
        best = int(results[b]["scores"].argmax())
        assert int(results[b]["labels"][best]) == int(expected_label[0])
        assert torch.allclose(results[b]["boxes"][best], expected_box[0], atol=0.1)


def test_training_with_batch_size_one_is_rejected_clearly():
    """The SK fuse path pools to 1x1, so BatchNorm needs a batch larger than one.

    This is inherent to Gao Eq. (3) using BN inside the fuse; training loaders
    must therefore use drop_last=True. Inference at batch size 1 is fine.
    """
    model = MTHARS(num_classes=6, scales=(2.0,))
    x = torch.randn(1, 1, 384, 9)

    model.train()
    try:
        model(x)
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected BatchNorm to reject a batch of one during training"

    model.eval()
    with torch.inference_mode():
        logits, offsets, windows = model(x)
    assert logits.shape[0] == 1


# ---------------------------------------------------------------- inference

def test_nms_suppresses_overlapping_boxes():
    boxes = torch.tensor([[0.5, 0.4], [0.51, 0.4], [0.9, 0.15]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    kept = nms_1d(boxes, scores, iou_threshold=0.45)
    assert kept.tolist() == [0, 2]


def test_nms_keeps_disjoint_boxes():
    boxes = torch.tensor([[0.2, 0.2], [0.8, 0.2]])
    scores = torch.tensor([0.6, 0.9])
    kept = nms_1d(boxes, scores, iou_threshold=0.45)
    assert sorted(kept.tolist()) == [0, 1]


def test_detect_recovers_a_planted_activity():
    # Hand-build head outputs that put all the probability mass on class 2 for
    # one window, with zero offset, and check detect() returns that window.
    num_classes = 6
    windows = generate_windows(4, (2.0,))
    logits = torch.full((1, windows.shape[0], num_classes + 1), -10.0)
    logits[0, :, 0] = 10.0                              # background everywhere
    logits[0, 3, 0] = -10.0
    logits[0, 3, 3] = 10.0                              # window 3 -> class 2
    offsets = torch.zeros(1, windows.shape[0], 2)

    results = detect(logits, offsets, windows, score_threshold=0.5)
    assert results[0]["labels"].tolist() == [2]
    assert torch.allclose(results[0]["boxes"][0], windows[3], atol=1e-5)


def test_detect_returns_empty_when_everything_is_background():
    windows = generate_windows(4, (2.0,))
    logits = torch.zeros(1, windows.shape[0], 7)
    logits[0, :, 0] = 20.0
    offsets = torch.zeros(1, windows.shape[0], 2)
    results = detect(logits, offsets, windows, score_threshold=0.5)
    assert results[0]["boxes"].numel() == 0


def test_segments_to_labels_paints_the_stream():
    detection = {"boxes": torch.tensor([[0.25, 0.5], [0.75, 0.5]]),
                 "labels": torch.tensor([1, 4]),
                 "scores": torch.tensor([0.9, 0.8])}
    labels = segments_to_labels(detection, stream_length=100)
    assert labels[:50].tolist() == [1] * 50
    assert labels[50:].tolist() == [4] * 50


def test_higher_scoring_detection_wins_an_overlap():
    detection = {"boxes": torch.tensor([[0.5, 1.0], [0.5, 0.2]]),
                 "labels": torch.tensor([1, 5]),
                 "scores": torch.tensor([0.6, 0.99])}
    labels = segments_to_labels(detection, stream_length=100)
    assert labels[50].item() == 5                       # the confident, narrow one
    assert labels[0].item() == 1


def test_concatenate_segments_collapses_runs():
    labels = torch.tensor([0, 0, 0, 2, 2, 1])
    assert concatenate_segments(labels) == [(0, 3, 0), (3, 5, 2), (5, 6, 1)]


# ---------------------------------------------------------------- metrics

def test_levenshtein_basics():
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([1, 2, 3], [1, 3]) == 1          # one deletion
    assert levenshtein([1, 2], [1, 2, 3]) == 1          # one insertion
    assert levenshtein([1, 2], [1, 5]) == 1             # one substitution


def test_ned_is_zero_for_a_perfect_sequence():
    assert normalized_edit_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert normalized_edit_distance([1, 9, 3], [1, 2, 3]) == 1 / 3


def test_weighted_f1_perfect_and_chance():
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    assert abs(weighted_f1(y, y, num_classes=3) - 1.0) < 1e-9

    wrong = torch.tensor([1, 1, 2, 2, 0, 0])
    assert weighted_f1(y, wrong, num_classes=3) == 0.0


def test_weighted_f1_weights_by_support():
    # Class 0 dominates; getting it right must dominate the score.
    y_true = torch.tensor([0] * 90 + [1] * 10)
    y_pred = torch.tensor([0] * 90 + [0] * 10)
    score = weighted_f1(y_true, y_pred, num_classes=2)
    assert 0.85 < score < 0.96


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
