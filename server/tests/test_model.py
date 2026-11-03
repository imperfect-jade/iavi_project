import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GainPolicy, GainPolicyV2, make_gain_policy, masked_smooth_l1, select_action


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


def test_action_only_model_ignores_points():
    model = GainPolicy(action_count=4, use_points=False).eval()
    visited = torch.tensor([[True, False, False, False]])
    available = torch.ones(1, 4, dtype=torch.bool)
    args = (torch.ones(1, 8, dtype=torch.bool), visited, available,
            torch.tensor([0]), torch.tensor([2.]))
    first = model(torch.zeros(1, 8, 3), *args)
    second = model(torch.full((1, 8, 3), 10000.), *args)
    assert torch.equal(first, second)
    assert model.point_encoder is None


def test_v2_point_branch_has_direct_gradient_path():
    torch.manual_seed(1)
    model = GainPolicyV2(action_count=4)
    points = torch.randn(3, 16, 3, requires_grad=True)
    mask = torch.ones(3, 16, dtype=torch.bool)
    visited = torch.tensor([[True, False, False, False]]).expand(3, -1)
    available = torch.ones(3, 4, dtype=torch.bool)
    scores = model(points, mask, visited, available, torch.zeros(3, dtype=torch.long),
                   torch.ones(3))
    target = torch.tensor([[0., .1, .2, .3], [0., .3, .2, .1], [0., .2, .3, .1]])
    masked_smooth_l1(scores, target, available & ~visited).backward()
    assert points.grad.norm().item() > 0
    assert model.point_encoder[0].weight.grad.norm().item() > 0
    assert model.point_head.weight.grad.norm().item() > 0


def test_factory_selects_versions_without_changing_v1_default():
    config = {"action_count": 4, "center_mm": [0., 0., 0.], "scale_mm": 100.}
    assert isinstance(make_gain_policy(config), GainPolicy)
    assert isinstance(make_gain_policy({**config, "version": "v2"}), GainPolicyV2)
    with pytest.raises(ValueError, match="version"):
        make_gain_policy({**config, "version": "unknown"})


def test_masked_loss_can_weight_rollout_states():
    prediction = torch.tensor([[0., 0.], [0., 1.]])
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    plain = masked_smooth_l1(prediction, target, mask)
    weighted = masked_smooth_l1(prediction, target, mask, torch.tensor([1., 3.]))
    assert plain.item() == pytest.approx(.125)
    assert weighted.item() == pytest.approx(.1875)
