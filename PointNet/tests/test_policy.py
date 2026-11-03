import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PointNet"))
sys.path.insert(0, str(ROOT / "server"))
from model import make_gain_policy
from policy import predict_scores, validate_checkpoint


def test_predict_scores_returns_structured_masked_action(tmp_path):
    config = {"model": {"version": "v2", "point_count": 32,
                        "action_count": 24, "center_mm": [0, 0, 50],
                        "scale_mm": 100, "use_points": True},
              "runtime": {"device": "cpu", "output_dir": "run"}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "run").mkdir()
    model = make_gain_policy(config["model"])
    for weight in model.parameters():
        weight.data.zero_()
    model.history_head[-1].bias.data.copy_(torch.arange(24, dtype=torch.float32))
    torch.save(model.state_dict(), tmp_path / "run" / "best_model.pt")
    cloud = tmp_path / "observed.npy"
    np.save(cloud, np.array([[i, 0, 50] for i in range(20)], np.float32))

    result = predict_scores(config_path, cloud, [0, 6], [0, 6, 12, 15], voxel_mm=0.5)

    assert (result["action_id"], result["arm_pose_id"], result["turntable_deg"]) == (15, 1, 90)
    assert result["best_score"] == pytest.approx(15)
    assert set(result["scores_by_action"]) == {"12", "15"}
    assert result["model_version"] == "v2"
    assert result["observed_points_used"] == 20


def test_validate_checkpoint_rejects_corrupt_weights_before_scan(tmp_path):
    config = {"model": {"version": "v2", "point_count": 2048,
                        "action_count": 24, "center_mm": [0, 0, 50],
                        "scale_mm": 100, "use_points": True},
              "runtime": {"device": "cpu", "output_dir": "run"}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "best_model.pt").write_bytes(b"broken")
    with pytest.raises((RuntimeError, ValueError, OSError)):
        validate_checkpoint(config_path)
