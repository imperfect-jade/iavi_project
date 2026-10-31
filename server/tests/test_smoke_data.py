import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_split, validate_disjoint_objects
from make_smoke_data import make_smoke_data


def test_smoke_splits_are_valid_and_disjoint(tmp_path):
    make_smoke_data(tmp_path, point_count=32, action_count=6)
    train = load_split(tmp_path / "train.npz", 32, 6)
    val = load_split(tmp_path / "val.npz", 32, 6)
    validate_disjoint_objects(train, val)
    assert train["points"].shape[0] > val["points"].shape[0]


def test_default_smoke_action_count_is_24(tmp_path):
    make_smoke_data(tmp_path, point_count=16)
    train = load_split(tmp_path / "train.npz", 16, 24)
    assert train["gain_target"].shape[1] == 24
