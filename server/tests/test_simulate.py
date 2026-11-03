import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulate_modelnet import action_pose, normalize_mesh_vertices, split_meshes, diverse_order, read_modelnet_mesh


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


def test_diverse_order_interleaves_categories():
    paths = ([Path(f"bottle/train/bottle_{i}.off") for i in range(4)] +
             [Path(f"bowl/train/bowl_{i}.off") for i in range(4)])
    result = diverse_order(paths)
    assert [p.parent.parent.name for p in result[:4]] == ["bottle", "bowl", "bottle", "bowl"]


def test_read_modelnet_mesh_repairs_glued_off_header_without_editing_source(tmp_path, monkeypatch):
    source = tmp_path / "model.off"
    original = b"OFF3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
    source.write_bytes(original)
    seen = []

    def fake_read(path):
        seen.append(Path(path).read_bytes())
        return object()

    monkeypatch.setitem(sys.modules, "open3d", types.SimpleNamespace(
        io=types.SimpleNamespace(read_triangle_mesh=fake_read)))
    read_modelnet_mesh(source)

    assert seen == [b"OFF\n" + original[3:]]
    assert source.read_bytes() == original


def test_read_modelnet_mesh_leaves_valid_header_untouched(tmp_path, monkeypatch):
    source = tmp_path / "valid.off"
    source.write_bytes(b"OFF\n3 1 0\n0 0 0\n")
    seen = []

    def fake_read(path):
        seen.append(Path(path))
        return object()

    monkeypatch.setitem(sys.modules, "open3d", types.SimpleNamespace(
        io=types.SimpleNamespace(read_triangle_mesh=fake_read)))
    read_modelnet_mesh(source)

    assert seen == [source]
