"""Tests for the dataset loaders.

The downloads are large and were not reachable from the machine this was written
on, so the parsers are exercised against fixtures written in the documented file
formats: the WISDM raw record format (including the malformed lines that file is
known for) and OPPORTUNITY's 250-column .dat layout.

That verifies the parsing logic and the channel/label bookkeeping. It does NOT
verify the assumptions about the real files -- that the OPPORTUNITY column
selection lands on the intended 113 sensors, or that the label column carries 18
codes. Both are checked at load time against the real data and will raise or
warn there.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

import har_data
import mthars
from har_data import (
    OPPORTUNITY_CHANNELS,
    WISDM_ACTIVITIES,
    StreamRecording,
    parse_opportunity,
    parse_wisdm,
    split_recordings,
    windows_from_recordings,
)


# ---------------------------------------------------------------- WISDM

WISDM_SAMPLE = (
    "33,Jogging,49105962326000,-0.6946377,12.680544,0.50395125;\n"
    "33,Jogging,49106062271000,5.012288,11.264028,0.95342433;\n"
    "33,Jogging,49106112167000,4.903325,10.882658,-0.08172209;\n"
    # two records run together on one line, as they do in the real file
    "33,Walking,49106222305000,-0.61291564,18.496431,3.0237172;"
    "33,Walking,49106332290000,-1.1849703,12.108489,7.205164;\n"
    "33,Walking,49106442306000,1.3756552,-2.4925237,-6.510526;\n"
    # malformed: truncated, empty field, unknown activity, stray double ;;
    "33,Walking,49106542312000,-0.61291564;\n"
    "33,Walking,49106652308000,,12.108489,7.205164;\n"
    "33,Napping,49106762304000,1.0,2.0,3.0;\n"
    ";;\n"
    "34,Sitting,49107062326000,1.0,2.0,3.0;\n"
    "34,Sitting,49107162326000,1.1,2.1,3.1;\n"
    "34,Standing,49107262326000,9.0,8.0,7.0;\n"
)


def _write(text: str, suffix: str = ".txt") -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    handle.write(text)
    handle.close()
    return Path(handle.name)


def test_wisdm_parses_records_and_skips_malformed():
    recordings = parse_wisdm(_write(WISDM_SAMPLE))

    assert len(recordings) == 2                     # subjects 33 and 34
    assert recordings[0].signal.shape == (6, 3)     # 3 Jogging + 3 Walking kept
    assert recordings[1].signal.shape == (3, 3)


def test_wisdm_splits_runs_into_intervals():
    recordings = parse_wisdm(_write(WISDM_SAMPLE))

    jogging = WISDM_ACTIVITIES.index("Jogging")
    walking = WISDM_ACTIVITIES.index("Walking")
    assert recordings[0].intervals == [(0, 3, jogging), (3, 6, walking)]

    sitting = WISDM_ACTIVITIES.index("Sitting")
    standing = WISDM_ACTIVITIES.index("Standing")
    assert recordings[1].intervals == [(0, 2, sitting), (2, 3, standing)]


def test_wisdm_reads_the_records_that_run_together_on_one_line():
    # The 4th and 5th records share a line; both must survive.
    recordings = parse_wisdm(_write(WISDM_SAMPLE))
    assert abs(float(recordings[0].signal[3, 1]) - 18.496431) < 1e-4
    assert abs(float(recordings[0].signal[4, 1]) - 12.108489) < 1e-4


def test_wisdm_rejects_a_file_with_nothing_usable():
    try:
        parse_wisdm(_write("garbage\nmore garbage\n"))
        raised = False
    except RuntimeError:
        raised = True
    assert raised


# ---------------------------------------------------------------- OPPORTUNITY

def _opportunity_file(rows: int = 40, label_plan=None) -> Path:
    """A .dat file in OPPORTUNITY's layout: 250 space-separated columns."""
    generator = np.random.default_rng(0)
    table = generator.normal(size=(rows, 250)).round(4)
    table[:, 0] = np.arange(rows) * 33          # MILLISEC

    label_plan = label_plan or [(0, rows // 2, 0), (rows // 2, rows, 406516)]
    for start, end, code in label_plan:
        table[start:end, har_data.OPPORTUNITY_LABEL_COLUMN] = code

    table[3:6, 10] = np.nan                     # missing sensor values to interpolate

    lines = "\n".join(" ".join("NaN" if math.isnan(v) else f"{v:g}" for v in row)
                      for row in table)
    return _write(lines + "\n", suffix=".dat")


def test_opportunity_selects_113_channels():
    path = _opportunity_file()
    recordings, class_names = parse_opportunity(path.parent, [path.name])

    assert recordings[0].signal.shape[1] == OPPORTUNITY_CHANNELS == 113
    assert class_names[0] == "NULL"


def test_opportunity_channel_selection_is_consistent():
    keep = [c for c in range(250) if c not in set(har_data.OPPORTUNITY_DROP)]
    assert len(keep) == 113
    assert 0 not in keep                                    # MILLISEC dropped
    assert har_data.OPPORTUNITY_LABEL_COLUMN not in keep     # labels are not inputs
    assert all(c < 134 for c in keep)                        # object/ambient dropped


def test_opportunity_interpolates_missing_values():
    path = _opportunity_file()
    recordings, _ = parse_opportunity(path.parent, [path.name])
    assert torch.isfinite(recordings[0].signal).all()


def test_opportunity_builds_intervals_from_the_label_column():
    path = _opportunity_file(rows=40, label_plan=[(0, 10, 0), (10, 25, 406516),
                                                  (25, 40, 404517)])
    recordings, class_names = parse_opportunity(path.parent, [path.name])

    assert len(class_names) == 3                             # NULL + 2 gestures
    # Codes map in sorted numeric order, not order of appearance, so the same
    # gesture gets the same index no matter which run it first shows up in:
    # 404517 -> 1, 406516 -> 2.
    assert class_names == ["NULL", "gesture_404517", "gesture_406516"]
    assert recordings[0].intervals == [(0, 10, 0), (10, 25, 2), (25, 40, 1)]


def test_opportunity_maps_null_to_class_zero():
    path = _opportunity_file(rows=20, label_plan=[(0, 10, 406516), (10, 20, 0)])
    recordings, class_names = parse_opportunity(path.parent, [path.name])
    assert class_names[0] == "NULL"
    # NULL occupies the second half, and must carry index 0.
    assert recordings[0].intervals[-1][2] == 0


def test_opportunity_rejects_a_file_with_the_wrong_column_count():
    path = _write("1 2 3\n4 5 6\n", suffix=".dat")
    try:
        parse_opportunity(path.parent, [path.name])
        raised = False
    except RuntimeError as exc:
        raised = "expected 250" in str(exc)
    assert raised


# ---------------------------------------------------------------- shared

def _toy_recordings(count: int = 10) -> list:
    return [StreamRecording(torch.randn(600, 3),
                            [(0, 300, 0), (300, 600, 1)]) for _ in range(count)]


def test_split_recordings_is_disjoint_and_proportional():
    recordings = _toy_recordings(10)
    train, test = split_recordings(recordings, train_fraction=0.7, seed=0)

    assert len(train) == 7 and len(test) == 3
    train_ids = {id(r) for r in train}
    assert not train_ids & {id(r) for r in test}


def test_split_recordings_is_deterministic():
    recordings = _toy_recordings(10)
    a, _ = split_recordings(recordings, seed=1)
    b, _ = split_recordings(recordings, seed=1)
    assert [id(r) for r in a] == [id(r) for r in b]


def test_windows_from_recordings_respects_activity_boundaries():
    # One recording, two activities of 300 samples, window 100 stride 50.
    recordings = _toy_recordings(1)
    split = windows_from_recordings(recordings, window=100, stride=50)

    assert split.windows.shape[1:] == (1, 100, 3)
    assert set(split.labels.tolist()) == {0, 1}
    # Windows never straddle the boundary, so each activity yields the same count.
    assert int((split.labels == 0).sum()) == int((split.labels == 1).sum())


def test_normalize_recordings_uses_training_statistics():
    train = [StreamRecording(torch.randn(500, 3) * 5 + 2, [(0, 500, 0)])]
    test = [StreamRecording(torch.randn(500, 3) * 5 + 2, [(0, 500, 0)])]
    scaled_train, scaled_test = har_data.normalize_recordings(train, test)

    signal = scaled_train[0].signal
    assert torch.allclose(signal.mean(dim=0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(signal.std(dim=0), torch.ones(3), atol=1e-4)
    # Test data is not separately standardised.
    assert not torch.allclose(scaled_test[0].signal.mean(dim=0), torch.zeros(3),
                              atol=1e-6)


# ---------------------------------------------------------------- anchor coverage

def test_anchor_coverage_flags_activities_that_are_too_short():
    # The smallest anchor is 1/sqrt(4) = 0.5 of the stream; a 5% activity is
    # unreachable and coverage must say so.
    short = [torch.tensor([[0.5, 0.05]]) for _ in range(20)]
    coverage = mthars.anchor_coverage(short, (2.0, 3.0, 4.0), n=64)

    assert coverage["matched_fraction"] == 0.0
    assert coverage["smallest_anchor"] == 0.5
    assert coverage["mean_best_iou"] < 0.2


def test_anchor_coverage_is_high_for_well_sized_activities():
    good = [torch.tensor([[0.5, 0.7]]) for _ in range(20)]
    coverage = mthars.anchor_coverage(good, (2.0, 3.0, 4.0), n=64)
    assert coverage["matched_fraction"] == 1.0
    assert coverage["mean_best_iou"] > 0.9


def test_anchor_coverage_reports_target_statistics():
    targets = [torch.tensor([[0.3, 0.4], [0.8, 0.6]])]
    coverage = mthars.anchor_coverage(targets, (2.0,), n=32)
    assert coverage["targets"] == 2
    assert 0.4 <= coverage["median_target_length"] <= 0.6


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
