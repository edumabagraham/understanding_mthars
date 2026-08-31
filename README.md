# understanding_mthars

A PyTorch implementation of **MTHARS** — multi-task human activity recognition
and segmentation — on top of the **selective kernel (SK) network** it uses as a
backbone, with a like-for-like comparison of the two on UCI-HAR.

> W. Gao, L. Zhang, W. Huang, F. Min, J. He and A. Song, "Deep Neural Networks
> for Sensor-Based Human Activity Recognition Using Selective Kernel
> Convolution," *IEEE TIM*, vol. 70, 2021.

> J. Duan et al., "A Multi-Task Deep Learning Approach for Sensor-based Human
> Activity Recognition and Segmentation," 2023.

MTHARS Sec. III-C: *"The MTHARS approach consists of the SK network [44] as a
backbone network, the Windows Generate module and the recognition and
segmentation module."* The SK network is not a competitor to MTHARS — it is the
first third of it. What MTHARS adds is working on an **unsegmented stream**:
predicting where each activity starts and ends *and* what it is, in one pass,
instead of classifying a window somebody else cut.

## Layout

| File | Contents |
|---|---|
| `mthars.py` | `SKUnit`, `SKConv`, `SKNet`; window generation, 1-D IoU, offset coding, matching, the recognition/segmentation head, the multi-task loss, NMS, and the NED / weighted-F1 metrics |
| `har_data.py` | UCI-HAR download, the window view, stream reconstruction, the stream dataset, the sliding-window baseline decoder, and a synthetic generator |
| `compare.py` | Trains both models and scores them on identical test streams |
| `ablation.py` | Reproduces Duan Table VII, the class/offset loss weight sweep |
| `test_mthars.py` | 33 unit tests over the model, loss, inference and metrics |
| `test_har_data.py` | 17 tests over the dataset parsers, using fixtures in the real file formats |
| `MTHARS_vs_SKNet.ipynb` | Walkthrough: architecture, each MTHARS component, training, and the comparison |
| `SKNet_HAR.ipynb` | The backbone on its own — UCI-HAR classification and the per-branch attention analysis. Self-contained for Colab, so it carries its own copy of the model classes; `mthars.py` is the version the rest of the repository uses |

## Running

Locally — simplest, and avoids the Colab issue below:

```bash
git clone https://github.com/edumabagraham/understanding_mthars
cd understanding_mthars
pip install torch numpy matplotlib requests
python test_mthars.py test_har_data.py   # 50 tests, no dataset needed
python compare.py --epochs 30            # UCI-HAR: trains both, prints the table
python compare.py --dataset wisdm        # also: wisdm, opportunity
python compare.py --synthetic            # exercises the path without any download
python ablation.py --dataset both --epochs 30 --repeats 3   # Duan Table VII
```

In Colab, the notebook needs the four `.py` files in its working directory.
**This repository is private**, so `raw.githubusercontent.com` answers 404 to an
unauthenticated Colab runtime and the automatic fetch fails. Either make the
repository public, add a fine-grained PAT to Colab Secrets as `GITHUB_TOKEN`, or
upload the files when the setup cell offers a file picker. The setup cell
detects a saved 404 page and refuses to import it rather than failing later with
a confusing `SyntaxError`.

## The comparison

Both models are scored on the same held-out streams from the official
subject-disjoint UCI-HAR test split, per timestep:

- **SKNet** labels a stream with a static sliding window (128 timesteps, 50%
  overlap, overlapping windows vote) — the approach Duan Sec. IV-C argues against.
- **MTHARS** predicts boundaries directly; NMS and Duan Algorithm 1 give the
  per-timestep labelling.

Metrics are accuracy, class-frequency-weighted F1 (Duan Eq. 11) and NED
(Eq. 9), the segmentation metric. For reference, Duan Tables V/VI report on UCI:
SK F1 0.9558 / accuracy 0.9406, MTHARS F1 0.9723 / accuracy 0.9632.

## Datasets

| | channels | rate | classes | streams |
|---|---|---|---|---|
| UCI-HAR | 9 | 50 Hz | 6 | reconstructed from the shipped 50%-overlapping windows |
| WISDM v1.1 | 3 | 20 Hz | 6 | genuine continuous recordings |
| OPPORTUNITY | 113 | 30 Hz | 18 (17 gestures + NULL) | genuine continuous recordings |

WISDM and OPPORTUNITY suit MTHARS better than UCI-HAR, which ships pre-cut
windows and no continuous signal.

Before trusting any result on a dataset, look at the **anchor coverage** line
that `compare.py` and `ablation.py` print. The smallest window MTHARS generates
is `1/sqrt(max(scales))` — half the stream — so an activity much shorter than
that cannot be matched by any anchor however well the model trains. Duan's
conclusion names this as future work. It bites hardest on OPPORTUNITY, whose
gestures are short.

## Duan Table VII

`ablation.py` sweeps the loss weights of Eq. (8) and prints its numbers beside
the paper's:

| Model | OPPORTUNITY | WISDM |
|---|---|---|
| SK [44] | 0.9074 | 0.9725 |
| α=1, β=1 | **0.9213** | 0.9877 |
| α=1, β=2 | 0.9060 | 0.9796 |
| α=1, β=3 | 0.9174 | 0.9874 |
| α=2, β=1 | 0.9075 | 0.9783 |
| α=2, β=3 | 0.9154 | **0.9881** |

The spread across settings is about 1 point on WISDM and 1.5 on OPPORTUNITY,
which one run per cell cannot resolve — use `--repeats 3` or more, and read the
ordering rather than the third decimal.

Caveats that affect how the numbers should be read — in particular that UCI-HAR
ships windows rather than streams, so the continuous signal is reconstructed
from the shipped 50%-overlapping windows (checked by `verify_overlap`) — are
documented in the final section of the notebook.
