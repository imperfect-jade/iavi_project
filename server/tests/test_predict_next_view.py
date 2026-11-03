import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import make_gain_policy
from predict_next_view import predict


def test_predict_loads_checkpoint_and_excludes_visited_actions(tmp_path):
    config = {"model": {"version": "v2", "point_count": 32, "action_count": 24,
                        "center_mm": [0, 0, 50], "scale_mm": 100,
                        "use_points": True},
              "runtime": {"device": "cpu", "output_dir": "run"}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "run").mkdir()
    torch.save(make_gain_policy(config["model"]).state_dict(),
               tmp_path / "run" / "best_model.pt")
    cloud_path = tmp_path / "observed.npy"
    np.save(cloud_path, np.array([[i, 0, 50] for i in range(20)], np.float32))

    result = predict(config_path, cloud_path, [0, 6], available=[0, 6, 13], voxel_mm=.5)

    assert result["action_id"] == 13
    assert result["arm_pose_id"] == 1
    assert result["turntable_deg"] == 30
    assert result["observed_points_used"] == 20


def test_predict_rejects_duplicate_visited_actions_before_loading_cloud(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"action_count": 24,
                                                      "use_points": True}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        predict(config_path, tmp_path / "missing.npy", [0, 0])
