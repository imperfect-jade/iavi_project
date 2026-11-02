import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulate_modelnet import action_pose, normalize_mesh_vertices, split_meshes


def test_24_actions_use_two_fixed_heights_and_twelve_angles():
    assert action_pose(0) == (0, 0)
    assert action_pose(6) == (0, 180)
    assert action_pose(12) == (1, 0)
    assert action_pose(23) == (1, 330)


def test_normalize_mesh_uses_fixed_max_dimension_and_floor():
    vertices = np.array([[-1, -2, -1], [1, 2, 3]], float)
    result = normalize_mesh_vertices(vertices, 100)
    np.testing.assert_allclose(result.max(axis=0) - result.min(axis=0), [50, 100, 100])
    assert result[:, 2].min() == 0


def test_split_meshes_keeps_source_test_holdout(tmp_path):
    for split in ("train", "test"):
        folder = tmp_path / "bottle" / split
        folder.mkdir(parents=True)
        for index in range(4 if split == "train" else 2):
            (folder / f"bottle_{index}.off").write_text("OFF\n")
    result = split_meshes(tmp_path, ["bottle"], val_fraction=.25, seed=42)
    assert [len(result[k]) for k in ("train", "val", "test")] == [3, 1, 2]
    assert all("/test/" in str(p).replace("\\", "/") for p in result["test"])
