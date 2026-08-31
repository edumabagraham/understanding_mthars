"""Reproduce Duan Table VII: the class/offset loss weight ablation.

    TABLE VII
    ACTIVITY CLASSIFICATION F1 VALUE WITH DIFFERENT WEIGHT SETTINGS

    Model         OPPORTUNITY   WISDM
    SK [44]          0.9074     0.9725
    a=1, b=1         0.9213     0.9877
    a=1, b=2         0.9060     0.9796
    a=1, b=3         0.9174     0.9874
    a=2, b=1         0.9075     0.9783
    a=2, b=3         0.9154     0.9881

alpha weights the classification loss and beta the offset loss in Duan Eq. (8),
`L = (1/N)(alpha*L_conf + beta*L_loc)`. The SK row is Gao's window classifier,
the baseline every MTHARS row is measured against.

Run:
    python ablation.py --dataset wisdm --epochs 30
    python ablation.py --dataset opportunity --epochs 30
    python ablation.py --dataset both --epochs 30

What is being reproduced, and what is not
-----------------------------------------
Duan reports a single F1 per cell with no variance, and the spread across the
five weight settings is about 1 point on WISDM and 1.5 on OPPORTUNITY. That is
small enough that one run per cell cannot distinguish the settings from each
other -- so `--repeats` trains each cell several times with different seeds and
reports mean and standard deviation. Read the ordering, not the third decimal.

The F1 here is Duan Eq. (11), class-frequency-weighted, computed per timestep
over held-out streams. The paper does not say at what granularity it scores
MTHARS, so this is the reading that puts both models on identical footing.
"""

import argparse
import json
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

import har_data
import mthars
from compare import evaluate_streams, train_mthars, train_sknet
from har_data import StreamDataset, collate_streams, windows_from_recordings

# Duan Table VII, for comparison against whatever this script produces.
PAPER = {
    "OPPORTUNITY": {"SK": 0.9074, (1, 1): 0.9213, (1, 2): 0.9060,
                    (1, 3): 0.9174, (2, 1): 0.9075, (2, 3): 0.9154},
    "WISDM": {"SK": 0.9725, (1, 1): 0.9877, (1, 2): 0.9796,
              (1, 3): 0.9874, (2, 1): 0.9783, (2, 3): 0.9881},
}

WEIGHTS = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3)]


def prepare(bundle: har_data.DatasetBundle, scales, verbose: bool = True):
    """Build the window and stream views, and report anchor coverage."""
    train_windows = windows_from_recordings(bundle.train, window=bundle.window,
                                            stride=bundle.window // 2)
    test_windows = windows_from_recordings(bundle.test, window=bundle.window,
                                           stride=bundle.window // 2)

    train_streams = StreamDataset(bundle.train, bundle.stream_length,
                                  stride=bundle.stream_length // 2)
    test_streams = StreamDataset(bundle.test, bundle.stream_length)

    if verbose:
        print(f"\n{bundle.name}: {bundle.num_channels} channels, "
              f"{bundle.num_classes} classes, {bundle.rate} Hz")
        print(f"  windows  train {len(train_windows.labels):,}  "
              f"test {len(test_windows.labels):,}  (length {bundle.window})")
        print(f"  streams  train {len(train_streams):,}  "
              f"test {len(test_streams):,}  (length {bundle.stream_length})")

        n = mthars.feature_sequence_length(bundle.stream_length)
        coverage = mthars.anchor_coverage(
            [test_streams[i][1] for i in range(len(test_streams))], scales, n)
        print(f"  anchor coverage: {coverage['matched_fraction']:.1%} of "
              f"{coverage['targets']} activities reach IoU 0.5 "
              f"(mean best {coverage['mean_best_iou']:.3f})")
        print(f"  smallest anchor {coverage['smallest_anchor']:.3f} of the stream; "
              f"median activity {coverage['median_target_length']:.3f}, "
              f"10th pct {coverage['target_length_p10']:.3f}")
        if coverage["matched_fraction"] < 0.5:
            print("  *** most activities cannot be matched by any anchor. The "
                  "scale set or stream length does not suit this dataset; "
                  "results below reflect that, not the loss weights. ***")

    return train_windows, test_windows, train_streams, test_streams


def run_dataset(name: str, epochs: int, repeats: int, scales, device,
                data_dir: str, stream_batch_size: int = 16) -> Dict:
    bundle = har_data.load_dataset(name, data_dir=data_dir)
    train_windows, test_windows, train_streams, test_streams = prepare(bundle, scales)

    results: Dict = {"dataset": bundle.name, "cells": {}}

    # One SK baseline per repeat, exactly as the paper has one SK row. The last
    # trained one is reused when scoring the MTHARS cells, where only the MTHARS
    # column of the result is read.
    print(f"\n[{bundle.name}] SK baseline")
    sk_scores, sknet = [], None
    for repeat in range(repeats):
        torch.manual_seed(1000 + repeat)
        sknet = train_sknet(train_windows, test_windows, bundle.num_classes, device,
                            epochs=epochs, verbose=False)
        # An untrained MTHARS satisfies the shared signature; its column is ignored.
        placeholder = mthars.MTHARS(num_classes=bundle.num_classes,
                                    scales=scales).to(device)
        scored = evaluate_streams(sknet, placeholder, test_streams, device,
                                  batch_size=stream_batch_size)
        sk_scores.append(scored["SKNet"]["f1"])
        print(f"  repeat {repeat + 1}/{repeats}: F1 {sk_scores[-1]:.4f}")
    results["cells"]["SK"] = sk_scores

    for alpha, beta in WEIGHTS:
        print(f"\n[{bundle.name}] alpha={alpha}, beta={beta}")
        scores = []
        for repeat in range(repeats):
            torch.manual_seed(2000 + repeat)
            loader = DataLoader(train_streams, batch_size=stream_batch_size,
                                shuffle=True, collate_fn=collate_streams,
                                drop_last=True)
            model = train_mthars(loader, bundle.num_classes, device, epochs=epochs,
                                 alpha=float(alpha), beta=float(beta),
                                 scales=scales, verbose=False)
            scored = evaluate_streams(sknet, model, test_streams, device,
                                      batch_size=stream_batch_size)
            scores.append(scored["MTHARS"]["f1"])
            print(f"  repeat {repeat + 1}/{repeats}: F1 {scores[-1]:.4f}")
        results["cells"][f"{alpha},{beta}"] = scores

    return results


def print_table(all_results: List[Dict]) -> None:
    names = [r["dataset"] for r in all_results]

    print("\n" + "=" * (24 + 26 * len(names)))
    print("Duan Table VII reproduction  (ours | paper)")
    print("=" * (24 + 26 * len(names)))
    header = f"{'Model':<12}"
    for name in names:
        header += f"{name:>26}"
    print(header)

    def cell(result, key, paper_key):
        scores = result["cells"].get(key)
        if not scores:
            return f"{'-':>26}"
        mean = sum(scores) / len(scores)
        paper = PAPER.get(result["dataset"], {}).get(paper_key)
        if len(scores) > 1:
            spread = (sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)) ** 0.5
            ours = f"{mean:.4f}+-{spread:.4f}"
        else:
            ours = f"{mean:.4f}"
        return f"{ours:>16}{('| ' + f'{paper:.4f}') if paper else '':>10}"

    print(f"{'SK [44]':<12}" + "".join(cell(r, "SK", "SK") for r in all_results))
    for alpha, beta in WEIGHTS:
        label = f"a={alpha},b={beta}"
        print(f"{label:<12}"
              + "".join(cell(r, f"{alpha},{beta}", (alpha, beta)) for r in all_results))

    print("\nBest setting per dataset:")
    for result in all_results:
        rows = {k: sum(v) / len(v) for k, v in result["cells"].items() if k != "SK"}
        best = max(rows, key=rows.get)
        paper_best = max((k for k in PAPER[result["dataset"]] if k != "SK"),
                         key=lambda k: PAPER[result["dataset"]][k])
        print(f"  {result['dataset']:<12} ours a={best}  "
              f"(paper a={paper_best[0]},b={paper_best[1]})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["wisdm", "opportunity", "both"],
                        default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1,
                        help="training runs per cell; >1 gives a standard deviation, "
                             "which the paper's single numbers lack")
    parser.add_argument("--stream-batch-size", type=int, default=16)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out", type=str, default="ablation_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scales = (2.0, 3.0, 4.0)
    print(f"Device: {device}   scales: {scales}   epochs: {args.epochs}   "
          f"repeats: {args.repeats}")

    datasets = ["wisdm", "opportunity"] if args.dataset == "both" else [args.dataset]
    all_results = []
    for name in datasets:
        start = time.time()
        all_results.append(run_dataset(name, args.epochs, args.repeats, scales,
                                       device, args.data_dir, args.stream_batch_size))
        print(f"\n[{name}] done in {time.time() - start:.0f}s")

    print_table(all_results)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
