import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GainPolicy, masked_smooth_l1, select_action


def test_forward_backward_and_padding():
    model = GainPolicy(action_count=4)
    points = torch.randn(2, 8, 3)
    point_mask = torch.ones(2, 8, dtype=torch.bool)
    point_mask[:, 6:] = False
    visited = torch.tensor([[True, False, False, False], [False, True, False, False]])
    available = torch.ones(2, 4, dtype=torch.bool)
    current = torch.tensor([0, 1])
    used = torch.tensor([2., 2.])
    prediction = model(points, point_mask, visited, available, current, used)
    assert prediction.shape == (2, 4)
    assert ((prediction >= 0) & (prediction <= 1)).all()
    loss = masked_smooth_l1(prediction, torch.zeros_like(prediction), available & ~visited)
    loss.backward()
    assert model.point_encoder[0].weight.grad is not None
    model.eval()
    altered = points.clone()
    altered[:, 6:] = 1e6
    baseline = model(points, point_mask, visited, available, current, used)
    assert torch.allclose(baseline, model(altered, point_mask, visited, available, current, used), atol=1e-6)


def test_select_action_masks_visited_and_unavailable():
    score = torch.tensor([[.9, .8, .7, .6]])
    visited = torch.tensor([[True, False, False, False]])
    available = torch.tensor([[True, False, True, True]])
    assert select_action(score, visited, available).item() == 2
