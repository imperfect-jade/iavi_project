import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_split, validate_disjoint_objects


def sample(object_id="a"):
    return dict(
        points=np.zeros((2, 4, 3), np.float32),
        point_mask=np.ones((2, 4), bool),
        visited_mask=np.array([[True, False, False], [False, True, False]]),
        available_mask=np.ones((2, 3), bool),
        gain_target=np.array([[0, .4, .1], [.3, 0, .5]], np.float32),
        current_action_id=np.array([0, 1], np.int64),
        used_scans=np.array([2, 3], np.float32),
        object_id=np.array([object_id, object_id]),
    )


def test_load_derives_loss_mask(tmp_path):
    path = tmp_path / "train.npz"
    np.savez(path, **sample())
    result = load_split(path, 4, 3)
    assert result["gain_loss_mask"].tolist() == [[False, True, True], [True, False, True]]


def test_reject_nan_target(tmp_path):
    arrays = sample()
    arrays["gain_target"][0, 1] = np.nan
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="finite"):
        load_split(path, 4, 3)


def test_reject_object_leakage(tmp_path):
    paths = [tmp_path / f"{name}.npz" for name in ("train", "val")]
    for path in paths:
        np.savez(path, **sample("same"))
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_objects(*(load_split(p, 4, 3) for p in paths))


def test_reject_current_action_not_visited(tmp_path):
    arrays = sample()
    arrays["visited_mask"][0, 0] = False
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="current_action_id"):
        load_split(path, 4, 3)
