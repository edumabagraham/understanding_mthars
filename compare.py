"""Train SKNet and MTHARS on UCI-HAR and compare them on the same test streams.

Run:  python compare.py --epochs 30

Comparison protocol
-------------------
Both models see byte-identical signal. They are scored on the same held-out
streams from the official UCI-HAR test subjects, at the same granularity:

* **SKNet** is Gao's window classifier. To label a stream it uses the static
  sliding window MTHARS argues against (Duan Sec. IV-C): cut into 128-timestep
  windows, classify each, and let overlapping windows vote per timestep.
* **MTHARS** predicts activity boundaries and classes directly on the stream,
  then NMS and concatenation give the per-timestep labelling.

Reported per timestep: accuracy and class-frequency-weighted F1 (Duan Eq. 11),
plus NED (Eq. 9) on the resulting activity sequence, which measures segmentation
quality -- how close the predicted run of activities is to the true one.

SKNet's own per-window accuracy on the shipped 128-timestep windows is reported
too, since that is the number Gao's paper quotes and it is not comparable to the
stream numbers.
"""

import argparse
import time
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import har_data
from har_data import (
    ACTIVITIES,
    StreamDataset,
    collate_streams,
    download_uci_har,
    load_split,
    normalize,
    reconstruct_streams,
    sliding_window_predict,
)
from mthars import (
    MTHARS,
    SKNet,
    concatenate_segments,
    decode_offsets,
    detect,
    multitask_loss,
    normalized_edit_distance,
    segments_to_labels,
    weighted_f1,
)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_sknet(train_windows, test_windows, num_classes: int, device,
                epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
                verbose: bool = True) -> SKNet:
    """Gao's setup: Adam with an exponentially decaying learning rate (Sec. IV)."""
    model = SKNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    loss_fn = nn.CrossEntropyLoss()

    # drop_last: the SK fuse pools to 1x1, so its BatchNorm cannot take a
    # trailing batch of one during training.
    loader = DataLoader(TensorDataset(train_windows.windows, train_windows.labels),
                        batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        model.train()
        total, correct, running = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * len(y)
            correct += int((logits.argmax(dim=1) == y).sum())
            total += len(y)
        scheduler.step()

        if verbose:
            accuracy = evaluate_windows(model, test_windows, device)["accuracy"]
            print(f"  [SKNet] epoch {epoch + 1:>3}/{epochs}  "
                  f"loss {running / total:.4f}  train acc {100 * correct / total:.2f}%  "
                  f"test acc {100 * accuracy:.2f}%")
    return model


def train_mthars(train_loader, num_classes: int, device, epochs: int = 30,
                 lr: float = 1e-3, alpha: float = 1.0, beta: float = 1.0,
                 scales=(2.0, 3.0, 4.0), verbose: bool = True) -> MTHARS:
    """Duan Sec. III-E. alpha=beta=1 is the paper's best setting (Table VII)."""
    model = MTHARS(num_classes=num_classes, scales=scales).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    for epoch in range(epochs):
        model.train()
        running_total = running_conf = running_loc = 0.0
        batches = 0

        for streams, boxes, labels, _ in train_loader:
            streams = streams.to(device)
            logits, offsets, windows = model(streams)
            loss, conf, loc = multitask_loss(logits, offsets, windows, boxes, labels,
                                             alpha=alpha, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_total += float(loss)
            running_conf += float(conf)
            running_loc += float(loc)
            batches += 1
        scheduler.step()

        if verbose:
            print(f"  [MTHARS] epoch {epoch + 1:>3}/{epochs}  "
                  f"loss {running_total / batches:.4f}  "
                  f"(conf {running_conf / batches:.4f}  loc {running_loc / batches:.4f})")
    return model


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_windows(model: SKNet, split, device, batch_size: int = 256) -> Dict:
    """SKNet's own task: classify the shipped 128-timestep windows."""
    model.eval()
    predictions = []
    for start in range(0, len(split.labels), batch_size):
        x = split.windows[start:start + batch_size].to(device)
        predictions.append(model(x).argmax(dim=1).cpu())
    predictions = torch.cat(predictions)

    return {
        "accuracy": float((predictions == split.labels).float().mean()),
        "f1": weighted_f1(split.labels, predictions, len(ACTIVITIES)),
        "predictions": predictions,
    }


def _detect_with_fallback(logits, offsets, windows, score_threshold, iou_threshold):
    """Detections for one stream, never empty.

    If nothing clears the score threshold the single most confident non-background
    window is kept, so a stream always receives a labelling and cannot be scored
    as a free pass.
    """
    results = detect(logits, offsets, windows,
                     score_threshold=score_threshold, iou_threshold=iou_threshold)
    probs = torch.softmax(logits, dim=-1)

    for b, result in enumerate(results):
        if result["boxes"].numel():
            continue
        foreground = probs[b, :, 1:]
        flat = int(foreground.argmax())
        anchor, cls = divmod(flat, foreground.shape[1])
        result["boxes"] = decode_offsets(offsets[b, anchor], windows[anchor]).unsqueeze(0)
        result["labels"] = torch.tensor([cls], device=logits.device)
        result["scores"] = foreground.reshape(-1)[flat].reshape(1)
    return results


@torch.no_grad()
def evaluate_streams(sknet: SKNet, mthars: MTHARS, dataset: StreamDataset, device,
                     batch_size: int = 16, score_threshold: float = 0.3,
                     iou_threshold: float = 0.45) -> Dict[str, Dict]:
    """Score both models per timestep on the same streams."""
    sknet.eval()
    mthars.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_streams)

    truth_all, sk_all, mt_all = [], [], []
    sk_ned, mt_ned = [], []

    for streams, _, _, timestep_labels in loader:
        streams_device = streams.to(device)
        logits, offsets, windows = mthars(streams_device)
        detections = _detect_with_fallback(logits, offsets, windows,
                                           score_threshold, iou_threshold)

        for b in range(len(streams)):
            truth = timestep_labels[b]
            valid = truth >= 0

            sk_pred = sliding_window_predict(sknet, streams[b], device=device)
            mt_pred = segments_to_labels({k: v.cpu() for k, v in detections[b].items()},
                                         stream_length=truth.numel())

            truth_all.append(truth[valid])
            sk_all.append(sk_pred[valid])
            mt_all.append(mt_pred[valid])

            truth_sequence = [label for _, _, label in concatenate_segments(truth[valid])]
            sk_ned.append(normalized_edit_distance(
                [label for _, _, label in concatenate_segments(sk_pred[valid])],
                truth_sequence))
            mt_ned.append(normalized_edit_distance(
                [label for _, _, label in concatenate_segments(mt_pred[valid])],
                truth_sequence))

    truth_all = torch.cat(truth_all)
    results = {}
    for name, predictions, neds in (("SKNet", torch.cat(sk_all), sk_ned),
                                    ("MTHARS", torch.cat(mt_all), mt_ned)):
        results[name] = {
            "accuracy": float((predictions == truth_all).float().mean()),
            "f1": weighted_f1(truth_all, predictions, len(ACTIVITIES)),
            "ned": sum(neds) / len(neds),
        }
    return results


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--stream-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stream-batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--synthetic", action="store_true",
                        help="run the whole pipeline on generated streams instead of "
                             "UCI-HAR; for checking the code path where the dataset "
                             "cannot be downloaded. The numbers mean nothing.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.synthetic:
        print("\n*** SYNTHETIC MODE: generated signal, not UCI-HAR. "
              "These numbers are a code-path check, not a result. ***\n")
        train_recordings, test_recordings = har_data.normalize_recordings(
            har_data.synthetic_recordings(num_subjects=12, seed=0),
            har_data.synthetic_recordings(num_subjects=4, seed=1))
        # Windows are cut from the already-normalised recordings, so both views
        # of the signal share statistics (as they do on the UCI path).
        train_split = har_data.windows_from_recordings(train_recordings)
        test_split = har_data.windows_from_recordings(test_recordings)
    else:
        root = download_uci_har(args.data_dir)
        train_split, test_split = normalize(load_split(root, "train"),
                                            load_split(root, "test"))

        mismatch = har_data.verify_overlap(train_split)
        print(f"Window overlap mismatch: {mismatch:.2e} "
              f"({'stream reconstruction valid' if mismatch < 1e-3 else 'RECONSTRUCTION INVALID'})")
        train_recordings = reconstruct_streams(train_split)
        test_recordings = reconstruct_streams(test_split)

    print(f"Train windows: {tuple(train_split.windows.shape)}  "
          f"Test windows: {tuple(test_split.windows.shape)}")

    train_streams = StreamDataset(train_recordings,
                                  stream_length=args.stream_length,
                                  stride=args.stream_length // 2)
    test_streams = StreamDataset(test_recordings, stream_length=args.stream_length)
    print(f"Train streams: {len(train_streams)}  Test streams: {len(test_streams)}")

    train_loader = DataLoader(train_streams, batch_size=args.stream_batch_size,
                              shuffle=True, collate_fn=collate_streams,
                              drop_last=True)  # see train_sknet on drop_last

    print("\nTraining SKNet...")
    start = time.time()
    sknet = train_sknet(train_split, test_split, len(ACTIVITIES), device,
                        epochs=args.epochs, batch_size=args.batch_size)
    sknet_time = time.time() - start

    print("\nTraining MTHARS...")
    start = time.time()
    mthars = train_mthars(train_loader, len(ACTIVITIES), device,
                          epochs=args.epochs, alpha=args.alpha, beta=args.beta)
    mthars_time = time.time() - start

    window_results = evaluate_windows(sknet, test_split, device)
    stream_results = evaluate_streams(sknet, mthars, test_streams, device,
                                      batch_size=args.stream_batch_size)

    dataset_name = "synthetic" if args.synthetic else "held-out UCI-HAR test"
    print("\n" + "=" * 62)
    print(f"Per-timestep results on the {dataset_name} streams")
    print("=" * 62)
    print(f"{'model':<10}{'accuracy':>12}{'weighted F1':>14}{'NED':>10}")
    for name in ("SKNet", "MTHARS"):
        r = stream_results[name]
        print(f"{name:<10}{r['accuracy']:>12.4f}{r['f1']:>14.4f}{r['ned']:>10.4f}")

    delta_f1 = stream_results["MTHARS"]["f1"] - stream_results["SKNet"]["f1"]
    delta_acc = stream_results["MTHARS"]["accuracy"] - stream_results["SKNet"]["accuracy"]
    print(f"\nMTHARS - SKNet:  F1 {delta_f1:+.4f}   accuracy {delta_acc:+.4f}")
    print("Duan Tables V/VI report, on UCI:  F1 +0.0165 (0.9558 -> 0.9723), "
          "accuracy +0.0226 (0.9406 -> 0.9632)")

    print(f"\nSKNet on the shipped 128-timestep windows (Gao's own task): "
          f"accuracy {window_results['accuracy']:.4f}, F1 {window_results['f1']:.4f}")
    print(f"Training time: SKNet {sknet_time:.0f}s, MTHARS {mthars_time:.0f}s")


if __name__ == "__main__":
    main()
