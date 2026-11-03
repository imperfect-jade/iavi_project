import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_dataset import build_dataset, coverage_table, marginal_gains, make_states
from data import load_split, validate_disjoint_objects


def test_coverage_and_gains():
    ref = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], np.float32)
    views = [ref[:2], ref[1:]]
    table = coverage_table(ref, views, 0.1)
    np.testing.assert_array_equal(table, [[1, 1, 0], [0, 1, 1]])
    np.testing.assert_allclose(marginal_gains(table, np.array([True, False])), [0, 1 / 3])


def test_build_separates_objects_and_reference(tmp_path):
    objects = []
    for split, oid in [("train", "a"), ("val", "b")]:
        views = []
        for action in range(5):
            filename = f"{oid}_{action}.npy"
            np.save(tmp_path / filename, np.array([[action * 10, 0, 0], [action * 10, 1, 0]], np.float32))
            views.append(filename)
        objects.append({"object_id": oid, "split": split, "views": views})
    manifest = {"unit": "mm", "coordinate_frame": "object_at_turntable_zero",
                "action_count": 5, "initial_views": [0, 1], "tolerance_mm": 2,
                "voxel_mm": 1, "point_count": 8, "reference_max_points": 100,
                "states_per_object": 3, "seed": 42, "objects": objects}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    build_dataset(path, tmp_path / "out")
    train = load_split(tmp_path / "out/train.npz", 8, 5)
    val = load_split(tmp_path / "out/val.npz", 8, 5)
    validate_disjoint_objects(train, val)
    assert len(train["points"]) == len(val["points"]) == 3
    assert "reference" not in train
    assert (tmp_path / "out/references/a.npz").exists()


def test_missing_view_fails_before_output(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"unit": "mm", "coordinate_frame": "object_at_turntable_zero",
                                "action_count": 2, "initial_views": [0], "tolerance_mm": 1,
                                "voxel_mm": 1, "point_count": 8, "reference_max_points": 10,
                                "states_per_object": 1, "seed": 1,
                                "objects": [{"object_id": "a", "split": "train", "views": ["missing.npy"]}]}))
    with pytest.raises(ValueError, match="views"):
        build_dataset(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_full_action_space_can_sample_200_states(tmp_path):
    objects = []
    for split, oid in (("train", "a"), ("val", "b")):
        views = []
        for action in range(24):
            name = f"{oid}_{action}.npy"
            np.save(tmp_path / name, np.array([[action, 0, 0]], np.float32))
            views.append(name)
        objects.append({"object_id": oid, "split": split, "views": views})
    manifest = {"unit": "mm", "coordinate_frame": "object_at_turntable_zero",
                "action_count": 24, "initial_views": [0, 6], "tolerance_mm": .1,
                "voxel_mm": .5, "point_count": 16, "reference_max_points": 100,
                "states_per_object": 200, "seed": 42, "objects": objects}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    build_dataset(path, tmp_path / "out")
    train = load_split(tmp_path / "out/train.npz", 16, 24)
    assert len(train["points"]) == 200
    assert len(np.unique(train["visited_mask"], axis=0)) == 200


def test_rollout_augmentation_includes_fixed_and_greedy_prefixes():
    count = 24
    views = [np.array([[i, 0, 0]], np.float32) for i in range(count)]
    table = np.eye(count, dtype=bool)
    table[15, 2:] = True  # greedy differs from fixed after initial actions
    cfg = {"action_count": count, "initial_views": [0, 6], "states_per_object": 30,
           "voxel_mm": .5, "point_count": 32, "augment_rollouts": True}
    rows = make_states({"object_id": "a"}, views, table, cfg, np.random.default_rng(3))
    assert len(rows) == 30
    assert len({tuple(np.flatnonzero(row[2])) for row in rows}) == 30
    fixed_four = {0, 6, 12, 18}
    greedy_three = {0, 6, 15}
    assert any(set(np.flatnonzero(row[2])) == fixed_four and row[-1] for row in rows)
    assert any(set(np.flatnonzero(row[2])) == greedy_three and row[-1] for row in rows)
    assert sum(bool(row[-1]) for row in rows) >= 8


def test_rebuilt_dataset_reuses_reference_tables(tmp_path):
    objects = []
    for split, oid in (("train", "a"), ("val", "b")):
        paths = []
        for action in range(5):
            name = f"{oid}_{action}.npy"
            np.save(tmp_path / name, np.array([[action, 0, 0]], np.float32))
            paths.append(name)
        objects.append({"object_id": oid, "split": split, "views": paths})
    manifest = {"unit": "mm", "coordinate_frame": "object_at_turntable_zero",
                "action_count": 5, "initial_views": [0, 1], "tolerance_mm": .1,
                "voxel_mm": .5, "point_count": 8, "reference_max_points": 10,
                "states_per_object": 6, "seed": 3, "objects": objects}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    build_dataset(path, tmp_path / "old")
    build_dataset(path, tmp_path / "new", augment_rollouts=True,
                  reference_dir=tmp_path / "old" / "references")
    for oid in ("a", "b"):
        with np.load(tmp_path / "old" / "references" / f"{oid}.npz") as old, \
             np.load(tmp_path / "new" / "references" / f"{oid}.npz") as new:
            np.testing.assert_array_equal(old["coverage"], new["coverage"])
            np.testing.assert_array_equal(old["reference"], new["reference"])
    assert load_split(tmp_path / "new" / "train.npz", 8, 5)["rollout_state"].any()
